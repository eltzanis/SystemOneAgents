"""What an LLM agent's answer becomes before scoring, and its record (the fields of a System 1 record, plus the LLM's own).

  pred          the tool of the answer's first function call; None when the answer has no function call (the agent calls
                no tool: for an out-of-scope request that is the right outcome) or the server returned an error; a name
                that is no tool becomes "?<name>" (a wrong tool)
  args          the call's arguments as a schema-validating MCP server receives them (`coerce`: "true" -> True for a
                boolean, "5" -> 5 for an integer, a JSON string -> a list for an array; an optional argument declared
                as anyOf [<type>, null] is read as <type>); empty values ("" or null) count as not given. Then, as in
                the System 1 agent, the policy value of output_dir when the call leaves it out (configs/agent.yaml)
  missing       the tool's required parameters without a value; never a policy parameter (the agent fills it)
  extra         arguments the label does not have, except policy values and values equal to the documented default
                (the System 1 agent does not pass those either)
  verdicts      s1eval.scoring.argument_verdicts, as for the System 1 agent
The LLM gives no probability, so `confidence` is None. `cost` is the request's measurement (s1eval.meter), `tokens` the
server's own usage report.
"""
import json
import re

import jsonschema

from s1agent.compiler import _type, is_default
from s1eval.scoring import argument_verdicts

POLICY_VALUE = "{workspace}/{tool}_policy"


def parse_arguments(raw):
    """-> (dict, problem or None). llama-server returns the arguments as a JSON string; some models encode them twice."""
    if isinstance(raw, dict):
        return raw, None
    try:
        v = json.loads(raw or "{}")
    except (TypeError, ValueError) as e:
        return {}, f"arguments are not JSON: {e}"
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            pass
    return (v, None) if isinstance(v, dict) else ({}, "arguments are not a JSON object")


def declared_type(prop):
    """The argument's JSON type; an optional argument written anyOf [<type>, null] (pydantic's Optional) is <type>."""
    t = _type(prop)
    if t is None and isinstance(prop.get("anyOf"), list):
        ts = [x.get("type") for x in prop["anyOf"] if isinstance(x, dict) and x.get("type") != "null"]
        if len(ts) == 1 and len(prop["anyOf"]) == 2 and isinstance(ts[0], str):
            return ts[0]
    return t


def coerce(value, prop):
    """The value as a schema-validating server receives it (pydantic's lax mode): strings that spell the declared type."""
    t = declared_type(prop)
    if isinstance(value, str):
        s = value.strip()
        if t == "boolean" and s.lower() in ("true", "false"):
            return s.lower() == "true"
        if t == "integer" and re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if t == "number":
            try:
                return float(s)
            except ValueError:
                return value
        if t in ("array", "object") and s[:1] in "[{":
            try:
                v = json.loads(s)
                if isinstance(v, list if t == "array" else dict):
                    return v
            except ValueError:
                pass
    if t == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def resolve(name, by_name):
    """The ToolSpec a function name refers to (also written `server/tool` or, when unique, as the bare tool name)."""
    if name in by_name:
        return by_name[name]
    n = str(name or "").strip()
    for t in by_name.values():
        if n in (f"{t.server}/{t.name}", f"{t.server}__{t.name}"):
            return t
    bare = [t for t in by_name.values() if t.name == n]
    return bare[0] if len(bare) == 1 else None


def interpret(entry, by_name):
    """What the agent did: kind (call | none | error), the tool, its arguments."""
    out = dict(kind="none", fn=None, tool=None, args={}, n_calls=0, problems=[], error=None)
    t = entry["turns"][0]
    if t.get("error"):
        out.update(kind="error", error=t["error"])
        return out
    calls = (t.get("message") or {}).get("tool_calls") or []
    if not calls:
        return out
    fn = calls[0].get("function") or {}
    name, (args, problem) = fn.get("name"), parse_arguments(fn.get("arguments"))
    if problem:
        out["problems"].append(problem)
    out.update(kind="call", fn=name, n_calls=len(calls), tool=resolve(name, by_name), args=args)
    if out["tool"] is None:
        out["problems"].append(f"unknown function {name!r}")
        out["unknown_tool"] = str(name)
    return out


def usage(entry):
    tok = dict(prompt=0, prompt_processed=0, completion=0, reasoning_chars=0)
    for t in entry["turns"]:
        u, tm, m = t.get("usage") or {}, t.get("timings") or {}, t.get("message") or {}
        tok["prompt"] += u.get("prompt_tokens") or 0
        tok["prompt_processed"] += tm.get("prompt_n") or 0
        tok["completion"] += u.get("completion_tokens") or 0
        tok["reasoning_chars"] += len(m.get("reasoning_content") or "")
    return tok


def record(row, entry, by_name, policy, workspace):
    """The scored record of one request (row: the labelled request; entry: the stored completion)."""
    it = interpret(entry, by_name)
    tool = it["tool"]
    if tool is not None:
        pred = f"{tool.server}/{tool.name}"
    elif it.get("unknown_tool") is not None:
        pred = f"?{it['unknown_tool']}"                    # a tool that does not exist: a wrong tool, not a decline
    else:
        pred = None
    gold_tool = row.get("tool")
    rec = dict(id=row["id"], gold=gold_tool, pred=pred, confidence=None, exact=pred == gold_tool,
               name_only=(pred or "").split("/")[-1] == (gold_tool or "").split("/")[-1],
               seconds=sum(t["seconds"] for t in entry["turns"]), model_calls=len(entry["turns"]), steps=[])
    if row.get("ask"):
        rec["ask"] = row["ask"]
    rec.update(kind=it["kind"], fn=it["fn"], n_calls=it["n_calls"], problems=it["problems"], error=it["error"],
               finish=[t.get("finish_reason") for t in entry["turns"]], tokens=usage(entry),
               call=dict(fn=it["fn"], args=it["args"]), cost=entry.get("cost"))
    if tool is not None and rec["exact"] and gold_tool:
        schema = tool.input_schema or {}
        props, required = schema.get("properties", {}) or {}, schema.get("required", []) or []
        args = {k: coerce(v, props.get(k, {})) for k, v in it["args"].items() if v is not None and v != ""}
        filled = []
        for p in sorted(policy):
            if p in props and p not in args:
                args[p] = POLICY_VALUE.format(workspace=workspace, tool=tool.name)
                filled.append(p)
        missing = [p for p in required if p not in policy and p not in args]
        gold = row.get("args", {})
        extra = sorted(k for k in args if k not in gold and k not in filled and not is_default(args[k], props.get(k, {})))
        rec.update(args_pred=args, args_extra=extra, args_missing=missing,
                   **argument_verdicts(tool.name, props, gold, args, extra, missing, row.get("ask", [])))
        rec.update(args_policy=filled, unknown_params=sorted(k for k in args if k not in props),
                   schema_errors=[f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
                                  for e in jsonschema.Draft202012Validator(schema).iter_errors(args)])
    return rec


def summary(recs):
    """Counts for a run's summary and the console: over the complete in-scope requests (labelled tool, every required
    value given), the out-of-scope requests and the underspecified requests (apart)."""
    n = len(recs)
    complete = [r for r in recs if r["gold"] and not r.get("ask")]
    oos = [r for r in recs if not r["gold"]]
    ask = [r for r in recs if r["gold"] and r.get("ask")]
    ok = sum(bool(r.get("call_correct")) for r in complete)
    argrows = [r for r in complete if "call_correct" in r]
    return dict(
        n=n, complete=len(complete), oos=len(oos), ask_rows=len(ask),
        tool_right=sum(r["exact"] for r in complete), oos_declined=sum(r["pred"] is None for r in oos),
        correct=ok, all_correct=ok + sum(r["pred"] is None for r in oos),
        strict=sum(bool(r.get("call_exact")) for r in complete), extras=sum(len(r["args_extra"]) for r in argrows),
        ask_no_call=sum(r["kind"] == "none" for r in ask),
        **{f"kind_{k}": sum(r["kind"] == k for r in recs) for k in ("call", "none", "error")},
        truncated=sum("length" in r["finish"] for r in recs),
        seconds=sum(r["seconds"] for r in recs) / n if n else 0.0,
        completion_tokens=sum(r["tokens"]["completion"] for r in recs) / n if n else 0.0)
