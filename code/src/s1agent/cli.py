"""s1a: the command line of the System 1 agent.

    s1a serve decider-4b                                     run llama-server for a model of configs/models.yaml
    s1a decide "state" "question" A B C --model decider-4b   one raw decision
    s1a plan "request" --catalogue C --model decider-4b      select a tool and fill its arguments, without running it
    s1a run "request" --catalogue C --servers S              plan, ask for anything missing, confirm, call the tool
    s1a catalog --servers S --out C                          start every MCP server once and save its tools

C: a tool catalogue (data/catalogues/*.json, or one written by `s1a catalog`); S: an MCP servers file
(configs/servers.example.yaml). The model's llama-server must be running for plan, run and decide (`s1a serve`).
Every decision of plan and run (question, options, probabilities, latency) is appended to runs/traces/.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
TRACES = Path("runs") / "traces"


def show_plan(plan):
    sel = plan.selection
    for d in sel.steps:
        top = sorted(zip(d.options, d.probs), key=lambda x: -x[1])[:3]
        console.print(f"[bold]{d.meta.get('tag')}[/] -> {d.choice[:90]!r}  p={d.confidence:.2f}  ({d.latency_s:.2f}s)")
        for o, p in top[1:]:
            console.print(f"      {p:.2f}  {o[:90]}")
    if sel.tool is None:
        console.print("[yellow]No tool fits (or confidence too low): the agent abstains.[/]"); return
    f = plan.filled
    if f is None:
        return
    t = Table(title=f"{sel.tool.server}/{sel.tool.name}  (selection p={sel.confidence:.2f}, {f.model_calls} argument calls)")
    for col in ("argument", "kind", "status", "value", "p"):
        t.add_column(col)
    for d in f.decisions:
        if d.status == "default":
            continue
        t.add_row(d.name, d.kind, d.status, json.dumps(d.value) if d.value is not None else "",
                  f"{d.confidence:.2f}" if d.confidence is not None else "")
    console.print(t)
    if f.missing:
        console.print(f"[red]missing required:[/] {', '.join(f.missing)}")
    if f.uncertain:
        console.print(f"[yellow]uncertain required:[/] {', '.join(f.uncertain)}")
    if f.errors:
        console.print(f"[red]schema errors:[/] {f.errors}")
    console.print(f"[dim]{plan.latency_s:.2f}s total[/]")


def _agent(a):
    from s1agent import build_agent
    return build_agent(a.model, a.catalogue, settings=a.settings, models=a.models, args_model=a.args_model)


def cmd_serve(a):
    from s1agent.config import models_config, server_command
    cmd, env, _, _ = server_command(models_config(a.models), a.name, a.extra)
    console.print(" ".join(cmd))
    os.execvpe(cmd[0], cmd, env)


def cmd_decide(a):
    from s1agent.backends import make_backend
    from s1agent.config import models_config
    b = make_backend(a.model, None if a.model == "lexical" else models_config(a.models))
    d = b.choice(a.state, a.question, a.options)
    for o, p in sorted(zip(d.options, d.probs), key=lambda x: -x[1]):
        console.print(f"{p:.3f}  {o}")
    console.print(f"[dim]{d.latency_s:.3f}s, {d.prompt_tokens} prompt tokens, missing labels {d.missing_labels}[/]")


def cmd_plan(a):
    from s1agent.agent import save_trace
    agent = _agent(a)
    show_plan(agent.plan(a.request))
    save_trace([agent.sel, agent.args], TRACES / f"plan_{datetime.now():%Y%m%d}.jsonl", {"request": a.request})


def cmd_run(a):
    from s1agent.agent import save_trace
    from s1agent.mcp_hub import MCPHub, load_servers
    agent = _agent(a)

    def report(event, payload):
        if event == "plan":
            show_plan(payload)
        elif event == "result":
            style = "red" if payload["is_error"] else "green"
            console.print(f"[{style}]{payload['tool']} returned:[/]\n{payload['result'][:4000]}")
        elif event == "abstain":
            console.print("[yellow]Stopping: no tool selected.[/]")

    def ask(name, prop, current):
        hint = f" [{current}]" if current is not None else ""
        v = console.input(f"[bold]{name}[/] ({prop.get('description', '')[:120]}){hint}: ").strip()
        return v or current

    def confirm(plan):
        if a.yes:
            return True
        return console.input(f"Run {plan.selection.tool.server}/{plan.selection.tool.name} with these arguments? [y/N] "
                             ).strip().lower() == "y"

    async def go():
        async with MCPHub(load_servers(a.servers)) as hub:
            return await agent.run(a.request, confirm, ask, report, hub, max_steps=a.max_steps)

    try:
        asyncio.run(go())
    finally:
        save_trace([agent.sel, agent.args], TRACES / f"run_{datetime.now():%Y%m%d}.jsonl", {"request": a.request})


def cmd_catalog(a):
    from s1agent.mcp_hub import build_catalogue, load_servers
    cat = build_catalogue(load_servers(a.servers), a.out)
    console.print(f"{len(cat['tools'])} tools from {len(cat['servers'])} servers -> {a.out}")
    for sid, err in cat["errors"].items():
        console.print(f"[red]{sid}: {err}[/]")


def main(argv=None):
    p = argparse.ArgumentParser(prog="s1a", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def configs(sp):
        sp.add_argument("--models", help="decision models file (default configs/models.yaml)")

    def common(sp):
        configs(sp)
        sp.add_argument("--catalogue", required=True, help="tool catalogue (JSON)")
        sp.add_argument("--model", required=True, help="decision model for every question (a name of the models file, "
                        "or 'lexical')")
        sp.add_argument("--args-model", help="another decision model for the argument questions only")
        sp.add_argument("--settings", help="agent settings (default configs/agent.yaml)")

    sp = sub.add_parser("serve"); sp.add_argument("name"); configs(sp)
    sp.add_argument("extra", nargs=argparse.REMAINDER, help="more llama-server flags"); sp.set_defaults(fn=cmd_serve)
    sp = sub.add_parser("decide"); sp.add_argument("state"); sp.add_argument("question"); sp.add_argument("options", nargs="+")
    sp.add_argument("--model", required=True); configs(sp); sp.set_defaults(fn=cmd_decide)
    sp = sub.add_parser("plan"); sp.add_argument("request"); common(sp); sp.set_defaults(fn=cmd_plan)
    sp = sub.add_parser("run"); sp.add_argument("request"); common(sp)
    sp.add_argument("--servers", required=True, help="MCP servers file (configs/servers.example.yaml)")
    sp.add_argument("--yes", action="store_true", help="run without asking for confirmation")
    sp.add_argument("--max-steps", type=int, default=1); sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("catalog"); sp.add_argument("--servers", required=True); sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_catalog)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
