"""The evaluation record of one request for the System 1 agent (scripts/run_s1.py, scripts/replay_s1.py).

The agent plans with select_min_conf = 0 (the raw selection is scored; abstaining is choosing "none of these" or failing
the capability check). Fields:
  id, gold, pred         the labelled tool and the selected one ("server/tool", or None for no call)
  confidence             the selection's confidence; exact, name_only: pred equals gold (whole, or the tool name only)
  seconds, model_calls   time and number of model questions of the request
  steps                  the selection decisions (question, options, probabilities, choice)
  args_pred              the call's arguments (when the right tool was selected)
  args_extra             arguments set that the label does not have; args_missing: required arguments without a value
  arg_decisions          per argument: kind, status (set, default, policy, missing, low_conf, is_default, unsupported),
                         value and probability
  verdicts               s1eval.scoring.argument_verdicts (args_correct, args_equivalent, args_conditional, call_exact,
                         call_exact_equivalent, call_exact_conditional, call_correct)
  cost                   the request's measurement (s1eval.meter), when measured
"""
from s1eval.scoring import argument_verdicts


def calls(agent):
    return len(agent.sel.trace) + (len(agent.args.trace) if agent.args is not agent.sel else 0)


def s1_record(agent, r, cost):
    """Plan request row `r` (labelled) and score it; `cost` is a context manager that yields the cost dict (Meter.measure,
    or NoCost)."""
    n0 = calls(agent)
    with cost as c:
        plan = agent.plan(r["request"])
    sel, filled = plan.selection, plan.filled
    pred = f"{sel.tool.server}/{sel.tool.name}" if sel.tool else None
    rec = dict(id=r["id"], gold=r.get("tool"), pred=pred, confidence=sel.confidence,
               exact=pred == r.get("tool"), name_only=(pred or "").split("/")[-1] == (r.get("tool") or "").split("/")[-1],
               seconds=c["seconds"], model_calls=calls(agent) - n0, steps=[d.as_dict() for d in sel.steps])
    if r.get("ask"):
        rec["ask"] = r["ask"]
    if filled is not None and rec["exact"] and r.get("tool"):
        gold = r.get("args", {})
        setv = {d.name: d.value for d in filled.decisions if d.status == "set"}
        rec["args_pred"] = filled.args
        props = sel.tool.input_schema.get("properties", {})
        rec["args_extra"] = sorted(k for k in setv if k not in gold)
        rec["args_missing"] = filled.missing
        rec.update(argument_verdicts(sel.tool.name, props, gold, filled.args, rec["args_extra"], filled.missing,
                                     r.get("ask", [])))
        rec["arg_decisions"] = [dict(name=d.name, kind=d.kind, status=d.status, value=d.value, p=d.confidence)
                                for d in filled.decisions]
    rec["cost"] = c
    return rec, plan


class NoCost:
    """A cost context without a meter: only the seconds."""

    def __enter__(self):
        import time
        self.t0 = time.perf_counter()
        self.c = {}
        return self.c

    def __exit__(self, *a):
        import time
        self.c["seconds"] = time.perf_counter() - self.t0
        return False


def verdict(rec):
    """Fully correct: call_correct for an in-scope request, no call for an out-of-scope one."""
    return bool(rec.get("call_correct")) if rec["gold"] else rec["pred"] is None
