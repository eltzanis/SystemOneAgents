"""The compiler oracle: for each labelled request, can the compiler express its labelled call at all?

    python scripts/oracle.py --data ../data/requests/imaging_test.jsonl --catalogue ../data/catalogues/imaging_tools.json
    python scripts/oracle.py --data my_requests.jsonl --catalogue my_tools.json --list      # also each failing request

A decision model can only pick among the options the compiler offers. The oracle is a backend that answers every
question the way the label says (the option equal to, or server-equivalent to, the labelled value of the argument the
question is about; "not specified" / "no further item" otherwise), driving the argument filler on the labelled tool. A
complete request (labelled tool, no `ask`) whose call it cannot make correct (s1eval.scoring.call_correct) has a stated
value that no question offers: no decision model can fill it. This finds such calls before any model is run, which is
how a compiler of this kind is developed: each unreachable value names a form of writing values that the candidate
readers (s1agent.candidates) do not cover yet. Out-of-scope and underspecified requests are not checked (declining and
asking need no value).

--list prints, per unreachable request, its id and the labelled values out of reach (label and what the oracle got).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

from s1agent.backends.base import Backend, Decision  # noqa: E402
from s1agent.catalog import load_catalogue  # noqa: E402
from s1agent.compiler import OFF, ON, ArgumentFiller  # noqa: E402
from s1agent.config import agent_settings  # noqa: E402
from s1eval.equivalence import equivalent  # noqa: E402
from s1eval.scoring import argument_verdicts, same  # noqa: E402

_NAME = re.compile(r"`([^`]+)`")
_OWNER = re.compile(r"The request gives the value '(.*)'\. Which argument does it belong to\?$", re.S)
_FIELD_VALUE = re.compile(r"Which value does the request give for the field `([^`]+)` of `([^`]+)`\?")
_FIELD = re.compile(r"Which field does the request want in `([^`]+)`")


def _items(v):
    if isinstance(v, list):
        return v
    return [x.strip() for x in str(v).split(",") if x.strip()] if isinstance(v, str) else [v]


class Oracle(Backend):
    """Picks the option that matches the label of the argument the question is about; else the last option, which is
    always the "not specified" / "no further item" / "not mentioned" option."""
    name = "oracle"

    def __init__(self):
        super().__init__()
        self.gold, self.tool = {}, None

    def _match(self, name, option, gold):
        if isinstance(gold, dict) and str(option).startswith("{"):          # a dictionary option
            option = json.loads(option)
        if isinstance(gold, list) and str(option).startswith(("[[", "[{")):  # a written list of lists or of objects
            option = json.loads(option)
        return same(option, gold) or equivalent(self.tool, name, option, gold)

    def _choice(self, state, question, options, kind):
        names = _NAME.findall(question)
        name = names[0] if names else None
        g = self.gold.get(name)
        pick = len(options) - 1
        owner = _OWNER.match(question)
        field_value, field = _FIELD_VALUE.match(question), _FIELD.match(question)
        if field_value:                                                    # the value of an object's field
            key, obj = field_value.groups()
            gv = self.gold.get(obj)
            gv = gv.get(key) if isinstance(gv, dict) else None
            if gv is not None:
                pick = next((i for i, o in enumerate(options) if same(o, gv) or equivalent(self.tool, key, o, gv)), pick)
        elif field:                                                        # which field of an object comes next
            gv = self.gold.get(field.group(1))
            chosen = re.search(r"Already chosen: (.*)\.$", question)
            done = {x.strip() for x in chosen.group(1).split(",")} if chosen else set()
            if isinstance(gv, dict):
                pick = next((i for i, o in enumerate(options) if o in gv and o not in done), pick)
        elif owner:                                                        # which argument a value belongs to
            v = owner.group(1)
            args = [_NAME.findall(o)[0] for o in options]
            pick = next((i for i, a in enumerate(args) if a in self.gold
                         and any(self._match(a, v, x) for x in _items(self.gold[a]))), 0)
        elif kind == "noul":                                               # gate: does the request give items for `name`?
            pick = 1 if name in self.gold else 0
        elif "turn the option" in question:
            pick = options.index(ON) if g is True else options.index(OFF) if g is False else len(options) - 1
        elif question.startswith("Which item") and g is not None:
            chosen = re.search(r"Already chosen: (.*)\.$", question)
            done = [x.strip() for x in chosen.group(1).split(",")] if chosen else []
            todo = [x for x in _items(g) if not any(self._match(name, d, x) for d in done)]
            pick = next((i for i, o in enumerate(options) for x in todo if self._match(name, o, x)), pick)
        elif g is not None:
            pick = next((i for i, o in enumerate(options) if self._match(name, o, g)), pick)
        probs = [0.97 if i == pick else 0.03 / (len(options) - 1) for i in range(len(options))]
        return Decision(question, options, probs, kind)


def reach(rows, catalogue, settings=None):
    """{id of a complete request: (reachable, {argument: (label, oracle's value)} out of reach, missing required)}."""
    cfg = settings if isinstance(settings, dict) else agent_settings(settings) if settings else agent_settings()
    tools = {t.qualified: t for t in load_catalogue(catalogue)}
    oracle = Oracle()
    filler = ArgumentFiller(oracle, policy_defaults=cfg.get("policy_defaults"), workspace=cfg["workspace"],
                            optional_min_conf=float(cfg["optional_min_conf"]), list_min_conf=float(cfg["list_min_conf"]))
    out = {}
    for r in rows:
        if not r.get("tool") or r.get("ask"):
            continue
        tool = tools[r["tool"]]
        oracle.gold, oracle.tool = r["args"], tool.name
        filled = filler.fill(r["request"], tool)
        props = tool.input_schema.get("properties", {})
        setv = {d.name for d in filled.decisions if d.status == "set"}
        v = argument_verdicts(tool.name, props, r["args"], filled.args, sorted(k for k in setv if k not in r["args"]),
                              filled.missing, r.get("ask", []))
        bad = {k: (r["args"][k], filled.args.get(k)) for k, ok in v["args_equivalent"].items()
               if not ok and k not in v["args_conditional"]}
        out[r["id"]] = (bool(v["call_correct"]), bad, filled.missing)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="labelled request file")
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--settings", help="agent settings (default configs/agent.yaml)")
    ap.add_argument("--list", action="store_true", help="print each unreachable request's id and values out of reach")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    res = reach(rows, a.catalogue, a.settings)
    fails = {i: x for i, x in res.items() if not x[0]}
    print(f"{Path(a.data).name}: {len(res) - len(fails)} of {len(res)} complete in-scope requests reachable")
    if a.list:
        for rid, (_, bad, missing) in fails.items():
            print(f"  {rid}: out of reach: " + "; ".join(f"{k} = {g!r} (got {p!r})" for k, (g, p) in bad.items())
                  + (f"; required missing: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
