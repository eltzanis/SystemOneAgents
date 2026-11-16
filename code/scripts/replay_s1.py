"""Rebuild System 1 records from recorded answer stores, without any model, and compare them with the run's records.

    python scripts/replay_s1.py --catalogue ../data/catalogues/imaging_tools.json \
        --data ../data/requests/imaging_test.jsonl --answers ../results/system1/answers \
        --records ../results/system1/records [--models decider-4b ...] [--report replay.json]

Each model is replaced by a backend that only reads its answer store (<answers>/<model>.jsonl) and fails on any question
the store has no answer for (s1agent.backends.StoreBackend); everything else is the agent's code. Every question the
compiler asks (state, question text, options) is hashed into the store's key, so a request whose questions differ in a
single character from the run's cannot be replayed: it shows up as a missing answer.

Each rebuilt record (s1eval.records.s1_record) is compared with the run's record (<records>/<data name>/<model>.jsonl)
field by field: the selected tool, the selection's confidence (within 1e-9) and steps (question, options, probabilities,
choice), the number of model questions, the arguments of the call, the extras and missing arguments, every argument
decision (kind, status, value, probability) and the verdicts. Masked: the value the policy gives an argument the request
leaves open ("{workspace}/{tool}_{timestamp}") holds the run's folder and the time of the call; its folder and time
stamp are replaced by placeholders in both records (the tool name is kept). Timings and costs are not compared.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

from s1agent import build_agent
from s1agent.backends import MissingAnswer, store_backend
from s1agent.config import models_config
from s1eval.records import NoCost, s1_record

MODELS = ["decider-4b", "pplx-decider-flash", "decider-35b-a3b", "openjev", "clef", "pplx-decider-27b-embed"]
VERDICTS = ("exact", "name_only", "args_correct", "args_equivalent", "args_conditional", "call_exact",
            "call_exact_equivalent", "call_exact_conditional", "call_correct", "args_extra", "args_missing")
STAMP = re.compile(r"^.+/([^/]+)_\d{8}_\d{6}$")


def _json(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def masked(rec, full_steps=True):
    """The fields compared, with policy values masked: {field: value}. full_steps: compare the selection steps' question,
    options and probabilities; otherwise (released records keep compact steps) their tag, choice and confidence."""
    policy = {d["name"] for d in rec.get("arg_decisions") or [] if d["status"] == "policy"}

    def mask(name, v):
        return STAMP.sub(r"<workspace>/\1_<time>", v) if name in policy and isinstance(v, str) else v
    out = {k: _json(rec.get(k)) for k in ("pred",) + VERDICTS}
    out["model_calls"] = rec.get("model_calls")
    out["args_pred"] = None if rec.get("args_pred") is None else {k: mask(k, v) for k, v in _json(rec["args_pred"]).items()}
    out["arg_decisions"] = None if rec.get("arg_decisions") is None else [
        (d["name"], d["kind"], d["status"], mask(d["name"], _json(d["value"]))) for d in rec["arg_decisions"]]
    out["steps"] = [(s.get("tag"), s["kind"], s["question"], s["options"], s["probs"], s["choice"]) if full_steps
                    else (s.get("tag"), s["kind"], s["choice"], s["confidence"]) for s in _json(rec.get("steps") or [])]
    return out


def differences(new, old):
    full = all("question" in s for s in old.get("steps") or [])
    a, b = masked(new, full), masked(old, full)
    diff = sorted(k for k in a if a[k] != b[k])
    if abs((new.get("confidence") or 0) - (old.get("confidence") or 0)) > 1e-9:
        diff.append("confidence")
    pa = [d.get("p") for d in new.get("arg_decisions") or []]
    pb = [d.get("p") for d in old.get("arg_decisions") or []]
    if len(pa) != len(pb) or any((x is None) != (y is None) or (x is not None and abs(x - y) > 1e-9) for x, y in zip(pa, pb)):
        diff.append("arg_decisions.p")
    return diff


def records_file(root, stem, model):
    for name in (f"{model}.jsonl", f"{model}_flat.jsonl"):
        p = Path(root) / stem / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no records of {model} for {stem} in {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="labelled request files (all with this catalogue)")
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--answers", required=True, help="folder of the answer stores (<model>.jsonl)")
    ap.add_argument("--records", required=True, help="folder of the run's records (<data name>/<model>.jsonl)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--report", help="write the per-request results as JSON here")
    a = ap.parse_args()
    cfg = models_config()
    report, all_same = {}, True
    for name in a.models:
        backend = store_backend(name, Path(a.answers) / f"{name}.jsonl", cfg)
        agent = build_agent(name, a.catalogue, backends=(backend, backend))
        agent.select_min_conf = 0.0
        for data in a.data:
            stem = Path(data).stem
            rows = [json.loads(l) for l in Path(data).read_text().splitlines() if l.strip()]
            run = {r["id"]: r for r in map(json.loads, records_file(a.records, stem, name).read_text().splitlines())}
            same, missing, differ, absent = 0, [], [], []
            for r in rows:
                if r["id"] not in run:
                    absent.append(r["id"]); continue
                try:
                    rec, _ = s1_record(agent, r, NoCost())
                except MissingAnswer:
                    missing.append(r["id"]); continue
                d = differences(rec, run[r["id"]])
                if d:
                    differ.append([r["id"], d])
                else:
                    same += 1
            all_same &= not (missing or differ or absent)
            report[f"{name}|{stem}"] = dict(requests=len(rows), identical=same, different=len(differ),
                                            missing_answers=len(missing), no_record=len(absent),
                                            differ=differ, missing=missing, absent=absent)
            print(f"{name:24s} {stem:18s} requests {len(rows):5d}  identical {same:5d}  different {len(differ):3d}  "
                  f"missing answers {len(missing):3d}" + (f"  no record {len(absent)}" if absent else ""), flush=True)
            for i, d in differ[:10]:
                print(f"    {i}: {', '.join(d)}")
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1) + "\n")
    print("all identical" if all_same else "SOME RECORDS DIFFER OR CANNOT BE REPLAYED")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
