"""The System 1 agent loop. Code owns the control flow; the decision model answers one typed question at a time.

    request ─► select a tool: one question over every tool of the catalogue plus "none of these" (which abstains)
            ─► when the two best tools are close, ask which of the two has parameters for the values the request gives
            ─► ask whether the chosen tool can do what the request asks (shown its description and parameter names);
               below CAPABLE_MIN the agent abstains
            ─► fill the arguments (s1agent.compiler)
            ─► a required argument missing or uncertain: ask the user
            ─► confirm, call the tool over MCP
            ─► stop or continue? (multi-step requests; earlier results become context and candidate sources)

The evaluation (scripts/run_s1.py) runs `plan` only: tool selection and argument filling, without calling the tool.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from s1agent.compiler import ArgumentFiller, short

NONE_FITS = "none of these: the request is not something these tools can do"
FIT_MARGIN = 0.3                     # the two best tools are this close: ask which one's parameters fit the request
CAPABLE_MIN = 0.5                    # abstain when P(the chosen tool can do what the request asks) is below this
STOP, CONTINUE = "the request is fully handled: stop", "another tool call is still needed"


@dataclass
class Selection:
    tool: object                         # ToolSpec or None
    confidence: float                    # product of the step confidences
    steps: list = field(default_factory=list)


@dataclass
class Plan:
    request: str
    selection: Selection
    filled: object = None                # compiler.Filled
    latency_s: float = 0.0

    @property
    def ready(self):
        return self.selection.tool is not None and self.filled is not None and not self.filled.missing and not self.filled.errors


def tool_option(t, n=220):
    return f"{t.name}: {short(t.description, n)}"


class S1Agent:
    """tools: the catalogue's ToolSpecs. select_backend answers the selection questions, args_backend (default: the
    same) the argument questions. select_min_conf: below this selection confidence the agent abstains instead of
    filling and calling (the evaluation sets it to 0 and scores the raw selection)."""

    def __init__(self, tools, select_backend, args_backend=None, select_min_conf=0.3, policy_defaults=None,
                 workspace="runs/workspace", optional_min_conf=0.7, list_min_conf=None):
        self.tools = tools
        self.sel = select_backend
        self.args = args_backend or select_backend
        self.select_min_conf = select_min_conf
        self.filler = ArgumentFiller(self.args, policy_defaults, workspace, optional_min_conf=optional_min_conf,
                                     list_min_conf=list_min_conf)

    # ---- selection -------------------------------------------------------------------------------------------------
    @staticmethod
    def _state(request, context):
        return f"User request: {request}" + (f"\nEarlier steps:\n{context}" if context else "")

    def select(self, request, context=""):
        sel = self._select(request, context)
        if sel.tool is not None:
            sel = self._capability(request, context, sel)
        return sel

    def _select(self, request, context=""):
        state = self._state(request, context)
        opts = [f"{t.server} / {tool_option(t)}" for t in self.tools] + [NONE_FITS]
        d = self.sel.choice(state, "Which tool should be called next for this request?", opts, tag="select:flat")
        if d.choice == NONE_FITS:
            return Selection(None, d.confidence, [d])
        return self._fit(state, d, self.tools, 1.0, [d])

    def _fit(self, state, d, tools, prior, steps):
        """The tool picked by decision d over `tools`. When the two best tools are within FIT_MARGIN, ask which of the two
        has parameters for the values the request gives (their parameter names are the options' text)."""
        ranked = sorted((i for i in range(len(tools))), key=lambda i: -d.probs[i])
        top = ranked[0]
        if len(ranked) > 1 and d.probs[ranked[0]] - d.probs[ranked[1]] < FIT_MARGIN:
            pair = [tools[i] for i in ranked[:2]]
            opts = [f"{t.name}: parameters {', '.join((t.input_schema or {}).get('properties', {}) or {}) or '(none)'}"
                    for t in pair]
            d3 = self.sel.choice(state, "Which of these two tools has parameters for the values the request gives?",
                                 opts, tag="select:fit")
            steps = steps + [d3]
            pick = ranked[d3.index]
            return Selection(tools[pick], prior * (d.probs[ranked[0]] + d.probs[ranked[1]]) * d3.confidence, steps)
        return Selection(tools[top], prior * d.probs[top], steps)

    def _capability(self, request, context, sel):
        """A yes/no check that the chosen tool can do what the request asks; below CAPABLE_MIN the agent abstains. The
        question shows the tool's whole description and its parameter names, so an option the request asks for ('skip
        the integrity check', 'export as DICOM') is visible where the tool offers it."""
        t = sel.tool
        params = ", ".join((t.input_schema or {}).get("properties", {}) or {}) or "(none)"
        q = f"Can the tool `{t.name}` do what the request asks? `{t.name}`: {short(t.description, 4000)} Parameters: {params}."
        p = self.sel.noul(self._state(request, context), q, tag="select:capable")
        steps = sel.steps + [self.sel.trace[-1]]
        if p < CAPABLE_MIN:
            return Selection(None, 1.0 - p, steps)
        return Selection(t, sel.confidence, steps)

    def plan(self, request, context=""):
        """Select a tool and, if the selection is confident enough, fill its arguments (nothing is called)."""
        t0 = time.perf_counter()
        sel = self.select(request, context)
        p = Plan(request, sel)
        if sel.tool is not None and sel.confidence >= self.select_min_conf:
            p.filled = self.filler.fill(request, sel.tool, context)
        p.latency_s = time.perf_counter() - t0
        return p

    def should_continue(self, request, context):
        d = self.sel.choice(self._state(request, context), "Is the request fully handled, or is another tool call needed?",
                            [STOP, CONTINUE], tag="loop")
        return d.choice == CONTINUE

    # ---- execution -------------------------------------------------------------------------------------------------
    async def run(self, request, confirm, ask, report, hub, max_steps=1, timeout=3600):
        """Plan, ask for what is missing, confirm, call the tool over MCP (hub: an entered s1agent.mcp_hub.MCPHub), and
        repeat while the model says another call is needed. confirm(plan) -> bool; ask(name, prop, current) -> str |
        None; report(event, payload) for the interface and the trace."""
        from s1agent.mcp_hub import result_text
        context, results = "", []
        for step in range(max_steps):
            plan = self.plan(request, context)
            report("plan", plan)
            tool = plan.selection.tool
            if tool is None or plan.selection.confidence < self.select_min_conf:
                report("abstain", plan); break
            for name in list(plan.filled.missing) + list(plan.filled.uncertain):
                prop = tool.input_schema.get("properties", {}).get(name, {})
                value = ask(name, prop, plan.filled.args.get(name))
                if value is None:
                    report("cancelled", name); return results
                plan.filled.args[name] = self.filler._coerce(value, prop) if prop.get("type") in ("integer", "number") else value
            if not confirm(plan):
                report("declined", plan); break
            res = await hub.call(tool.server, tool.name, plan.filled.args, timeout=timeout)
            text = result_text(res)
            results.append(dict(tool=f"{tool.server}/{tool.name}", args=plan.filled.args, result=text,
                                is_error=bool(getattr(res, "is_error", getattr(res, "isError", False)))))
            report("result", results[-1])
            context += (f"{step + 1}. called {tool.server}/{tool.name} with {json.dumps(plan.filled.args)}\n"
                        f"   result: {short(text, 1200)}\n")
            if step + 1 < max_steps and not self.should_continue(request, context):
                break
        return results


def save_trace(backends, path, extra=None):
    """Append every decision of the given backends (each backend once) to a JSON-lines file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.open("a") as f:
        for b in backends:
            if id(b) in seen:
                continue
            seen.add(id(b))
            for d in b.trace:
                f.write(json.dumps({**(extra or {}), **d.as_dict()}) + "\n")


def run_sync(coro):
    return asyncio.run(coro)
