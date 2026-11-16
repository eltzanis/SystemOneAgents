"""Fabricated content in tool calls: can an agent send a tool, argument or value that nothing grounds?

    python scripts/fabrication.py                 # ../data + ../results -> ../results/fabrication/
    python scripts/fabrication.py --check-sample  # also re-draw the review sample and compare it with the released one

Scope: every call each agent made on the evaluation set (all requests, underspecified ones included). System 1: the
recorded calls; LLM agents: the first tool call of each completion. The output folder (`output_dir`, which System 1
fills by policy) is left out for every agent. Categories, per call:

  F1 nonexistent tool       the function name is not a tool of the catalogue
  F2 nonexistent argument   an argument name that is not a property of the called tool's schema
  F3 invalid value          a JSON-schema validation error on the call's arguments, after the scorer's lax coercion
                            (llm_agent.score.coerce, the optional-anyOf rule included); missing required arguments are
                            not counted. Reported as schema validity, apart from fabrication
  F4 ungrounded value       the unit is each value of an argument that exists in the schema: a string or number, each
                            list item, each object key and value (booleans are not judged). A value is grounded when
                            (a) the scorer judges it to be the labelled value of that argument; (b) it is a string that
                            occurs in the request, ignoring case, spaces, '_', '-' and '/'; (c) it is a number written
                            in the request, as the compiler reads numbers; (d) it is one of the options the compiler
                            offers for that argument of that tool on that request (derived forms such as unit
                            conversions and joined paths included); (e) it is a value of the schema (an enum value, the
                            default, text of the argument's or the tool's description). A comma-separated string that is
                            not grounded as a whole is judged item by item. Otherwise the value is ungrounded. The test
                            is generous: a value written anywhere in the request counts as grounded, even in the wrong
                            argument
  F5 invented required value  underspecified requests only: the call sets a required argument the request does not
                            give; per request, "ungrounded" when any such value is ungrounded, else "grounded" (a value
                            written elsewhere in the request, misassigned)

Precision of F4: an independent reviewer judged a random sample of 200 LLM values flagged F4 (drawn with seed 20261110,
random.sample over the flagged values in the order this script lists them), as grounded, invented or unclear, without
reporting request text (../results/fabrication_review/). Verdicts "invented" of the kind "mis-derived stated quantity"
count as "engine", not as the model's fabrication, only when the raw completion writes the value with a truncated
exponent (llama.cpp's tool-call parser dropped the exponent's digits); otherwise they stay "invented". Lower bound of the precision: unclear counted grounded; upper bound: unclear counted invented. The adjusted
counts per LLM setting are the flagged counts times these bounds (descriptive; no test).

Writes fabrication.json and fabrication.md, and, when the review is present, precision.json and precision.md. Counts
only are printed, never request text.
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE / "src"))  # run from a checkout without installing

import jsonschema  # noqa: E402

import analyse as A  # noqa: E402

POLICY = "output_dir"
SEED = 20261110


def squash(s):
    return re.sub(r"[\s_\-/]+", "", str(s).lower())


def as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", ""))
    except ValueError:
        return None


def leaves(v):
    """(kind, value) of every judged leaf: strings and numbers, list items, object keys and values; booleans skipped."""
    if isinstance(v, dict):
        for k, x in v.items():
            yield "key", k
            yield from leaves(x)
    elif isinstance(v, list):
        for x in v:
            yield from leaves(x)
    elif isinstance(v, bool) or v is None:
        return
    elif isinstance(v, (int, float, str)):
        if str(v).strip():
            yield "value", v


class Grounding:
    """Where a value comes from, for one request and tool: the request, its numbers, the compiler's options, the schema."""

    def __init__(self, filler, numbers_of):
        self.filler, self.numbers_of = filler, numbers_of
        self.cache = {}

    def context(self, request, tool):
        key = (request, tool.qualified)
        if key not in self.cache:
            opts = defaultdict(set)
            for name, spec, _ in self.filler.questions(request, tool):
                if not spec:
                    continue
                for o in spec["options"]:
                    for src in (spec.get("arrays") or {}, spec.get("dicts") or {}):
                        if o in src:
                            for _, x in leaves(src[o]):
                                opts[name].add(squash(x))
                    opts[name].add(squash(o))
            self.cache[key] = (dict(opts), squash(request),
                               [float(n) for n in self.numbers_of(request) if as_number(n) is not None])
        return self.cache[key]

    def grounded(self, leaf, arg, prop, tool, request):
        """The source that grounds `leaf` (request, request number, compiler option, schema, empty), or None."""
        opts, sreq, nums = self.context(request, tool)
        s = squash(leaf)
        if not s:
            return "empty"
        if s in sreq:
            return "request"
        n = as_number(leaf)
        if n is not None and any(abs(n - m) <= 1e-9 * max(1.0, abs(m)) for m in nums):
            return "request number"
        if s in opts.get(arg, set()) or (n is not None and any(as_number(o) == n for o in opts.get(arg, set()))):
            return "compiler option"
        schema_text = squash(json.dumps(prop, ensure_ascii=False)) + squash(tool.description or "")
        if s in schema_text:
            return "schema"
        if n is not None:
            desc_nums = self.numbers_of(json.dumps(prop) + " " + (tool.description or ""))
            if any(as_number(d) == n for d in desc_nums):
                return "schema"
        return None


def judge_call(tool, args, row, labelled_ok, G, policy):
    """-> dict(F2=[names], F3=[messages], F4=[(arg, leaf)], judged=int, sources=Counter, F5=[(arg, grounded)])"""
    props = (tool.input_schema or {}).get("properties", {}) or {}
    out = dict(F2=[], F3=[], F4=[], judged=0, sources=Counter(), F5=[])
    out["F2"] = sorted(k for k in args if k not in props)
    for e in jsonschema.Draft202012Validator(tool.input_schema or {}).iter_errors(args):
        if e.validator != "required":
            out["F3"].append(f"{'/'.join(map(str, e.path)) or '(root)'}: {e.validator}")
    ask = set(row.get("ask") or [])
    for arg, v in args.items():
        if arg not in props or arg in policy or v is None or v == "":
            continue
        if labelled_ok.get(arg):
            out["sources"]["labelled value"] += 1
            out["judged"] += 1
            continue
        bad = []
        for _, leaf in leaves(v):
            src = G.grounded(leaf, arg, props[arg], tool, row["request"])
            parts = [leaf]
            if src is None and isinstance(leaf, str) and "," in leaf:
                # a list written as one comma-separated string (string-typed list arguments): each item is a value
                parts = [p.strip() for p in leaf.split(",") if p.strip()]
            for p in parts:
                out["judged"] += 1
                s = src if len(parts) == 1 else G.grounded(p, arg, props[arg], tool, row["request"])
                if s is None:
                    bad.append(p)
                else:
                    out["sources"][s] += 1
        out["F4"] += [(arg, b) for b in bad]
        if arg in ask:
            out["F5"].append((arg, not bad))
    return out


def _add(t, j, row):
    t["F2"] += bool(j["F2"]); t["F3"] += bool(j["F3"]); t["F4"] += bool(j["F4"])
    t["any"] += bool(j["F2"] or j["F3"] or j["F4"])
    t["judged"] += j["judged"]; t["ungrounded"] += len(j["F4"]); t["sources"].update(j["sources"])
    if row.get("ask") and row.get("tool") and j["F5"]:
        # per request: invented if any missing required value it fills is ungrounded, else a misassigned written value
        t["F5_ungrounded" if not all(g for _, g in j["F5"]) else "F5_grounded"] += 1


def fabrication(a):
    """-> (results by agent, flagged LLM values in order, rows, tools)."""
    from llm_agent import score, tools as T
    from llm_agent.agent import all_models
    from s1agent import candidates as C
    from s1agent.backends.lexical import LexicalBackend
    from s1agent.catalog import load_catalogue
    from s1agent.compiler import ArgumentFiller
    from s1agent.config import agent_settings

    data, res = Path(a.data), Path(a.results)
    rows = {(k, r["id"]): r for k in A.SETS for r in A.load(data / "requests" / f"{k}.jsonl")}
    cats = {k: data / "catalogues" / c for k, c in A.SETS.items()}
    tools = {k: {t.qualified: t for t in load_catalogue(c)} for k, c in cats.items()}
    built = {k: T.build(c) for k, c in cats.items()}
    cfg = agent_settings()
    filler = ArgumentFiller(LexicalBackend(), cfg.get("policy_defaults"), cfg["workspace"],
                            optional_min_conf=float(cfg["optional_min_conf"]), list_min_conf=float(cfg["list_min_conf"]))
    G = Grounding(filler, C.numbers)
    ws = str(Path(cfg["workspace"]).resolve())
    policy = {POLICY}
    results, flagged = {}, []

    def tally(key, label, family):
        results[key] = dict(label=label, family=family, calls=0, F1=0, F2=0, F3=0, F4=0, any=0, judged=0,
                            ungrounded=0, F5_grounded=0, F5_ungrounded=0, ask_calls=0, sources=Counter())
        return results[key]

    # System 1: the recorded calls
    for m in A.S1:
        t = tally(f"s1:{m}", f"System 1 {A.S1_LABEL[m]}", "s1")
        for k in A.SETS:
            for rec in A.load(res / "system1" / "records" / k / f"{m}.jsonl"):
                u = (k, rec["id"])
                if u not in rows or not rec.get("pred"):
                    continue
                t["calls"] += 1
                tool = tools[k].get(rec["pred"])
                if tool is None:
                    t["F1"] += 1; t["any"] += 1; continue
                row = rows[u]
                ok = {}
                if rec["pred"] == row.get("tool"):
                    ok = {x: bool(v) for x, v in (A.rescore_s1(rec, row, tools[k]).get("args_equivalent") or {}).items()}
                _add(t, judge_call(tool, rec.get("args_pred") or {}, row, ok, G, policy), row)

    # LLM agents: the first call of each completion
    cfgl = all_models()["models"]
    for m in cfgl:
        for c in A.MODES:
            t = tally(f"{c}:{m}", f"{cfgl[m].get('label', m)} [{c}]", c)
            for k in A.SETS:
                key = json.loads((res / "llm" / k / f"{m}__{c}.summary.json").read_text())["settings"]
                ans = {e["id"]: e for e in A.load(res / "llm" / k / f"{m}__{c}.jsonl") if e.get("settings") == key}
                _, by_name, pol = built[k]
                for (kk, i), row in rows.items():
                    if kk != k:
                        continue
                    it = score.interpret(ans[i], by_name)
                    if it["kind"] != "call":
                        continue
                    t["calls"] += 1
                    if it["tool"] is None:
                        t["F1"] += 1; t["any"] += 1; continue
                    tool = it["tool"]
                    props = (tool.input_schema or {}).get("properties", {}) or {}
                    args = {x: score.coerce(v, props.get(x, {})) for x, v in (it["args"] or {}).items()
                            if v is not None and v != ""}
                    ok = {}
                    if f"{tool.server}/{tool.name}" == row.get("tool"):
                        r = score.record(row, ans[i], by_name, pol, ws)
                        ok = {x: bool(v) for x, v in (r.get("args_equivalent") or {}).items()}
                    j = judge_call(tool, args, row, ok, G, policy)
                    _add(t, j, row)
                    for arg, leaf in j["F4"]:
                        flagged.append(dict(agent=f"{c}:{m}", source=k, id=i, tool=f"{tool.server}/{tool.name}",
                                            argument=arg, value=leaf))
    return results, flagged, rows, tools


def review_sample(flagged, rows, tools, n=200):
    """The review sample: (items as in sample.jsonl, key lines {n, agent})."""
    random.seed(SEED)
    pick = random.sample(flagged, min(n, len(flagged)))
    items = []
    for i, f in enumerate(pick, 1):
        tool = tools[f["source"]][f["tool"]]
        items.append(dict(n=i, request=rows[(f["source"], f["id"])]["request"], tool=f["tool"],
                          tool_description=tool.description, argument=f["argument"],
                          argument_schema=(tool.input_schema or {}).get("properties", {}).get(f["argument"]),
                          value=f["value"]))
    return items, [dict(n=i, agent=f["agent"]) for i, f in enumerate(pick, 1)]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


TRUNCATED_EXPONENT = r'"%s"\s*:\s*"?-?\d+(?:\.\d+)?[eE][-+]?0"?\s*[,}]'


def raw_arguments(res, f):
    """The raw `arguments` string of the first tool call of the completion behind a flagged value."""
    c, m = f["agent"].split(":", 1)
    base = Path(res) / "llm" / f["source"]
    key = json.loads((base / f"{m}__{c}.summary.json").read_text())["settings"]
    for line in (base / f"{m}__{c}.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("id") == f["id"] and e.get("settings") == key:
            calls = ((e["turns"][0].get("message") or {}).get("tool_calls")) or []
            a = (calls[0].get("function") or {}).get("arguments") if calls else None
            return a if isinstance(a, str) else json.dumps(a)
    return ""


def precision(D, review, out, pick, res):
    """The review's precision per thinking mode and the precision-adjusted counts per agent. A value the reviewer judged
    a mis-derived stated quantity counts as an engine error, not as the model's invention, only when the raw completion
    writes it with a truncated exponent (e.g. 1e-0 for 1e-05: llama.cpp's tool-call parser dropped the exponent's
    digits); otherwise it stays invented."""
    V = {json.loads(l)["n"]: json.loads(l) for l in (review / "verdicts.jsonl").read_text().splitlines() if l.strip()}
    K = {json.loads(l)["n"]: json.loads(l)["agent"] for l in (review / "review_key.jsonl").read_text().splitlines()
         if l.strip()}
    assert set(V) == set(K) and len(pick) == len(V)
    fam, engine = defaultdict(Counter), []
    for n, v in V.items():
        c = v["verdict"]
        if c == "invented" and "mis-derived" in (v.get("kind") or ""):
            f = pick[n - 1]
            assert f["agent"] == K[n]
            if re.search(TRUNCATED_EXPONENT % re.escape(f["argument"]), raw_arguments(res, f)):
                c = "engine"; engine.append(n)
        fam[K[n].split(":")[0]][c] += 1
    print(f"engine errors (truncated exponent in the raw completion): sample items {engine}")
    P, L = {}, ["# Precision of the ungrounded-value check", "",
                "Independent review of 200 random flagged LLM values (sample drawn with seed 20261110).", "",
                "| thinking mode | sample | invented | unclear | grounded | engine | lower bound (95% CI) | unclear left out | upper bound |",
                "|---|---|---|---|---|---|---|---|---|"]
    for f in ("think", "nothink"):
        c = fam[f]; n = sum(c.values()); inv, unc = c["invented"], c["unclear"]
        lo, hi = wilson(inv, n)
        P[f] = dict(strict=inv / n, mid=inv / (n - unc), high=(inv + unc) / n, n=n)
        L.append(f"| {'on' if f == 'think' else 'off'} | {n} | {inv} | {unc} | {c['grounded']} | {c['engine']} | "
                 f"{inv / n:.3f} ({lo:.3f}–{hi:.3f}) | {inv / (n - unc):.3f} | {(inv + unc) / n:.3f} |")
    L += ["", "## Per agent (adjusted = flagged × precision, lower–upper bound)", "",
          "| agent | calls | nonexistent tool | nonexistent argument | calls with a flagged value (%) | calls with an invented value, adjusted (%) | "
          "invented values, adjusted | underspecified requests with a required value filled: written elsewhere / no source | schema-invalid calls |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in D.items():
        n = v["calls"]
        if v["family"] == "s1":
            adj, vals = "0 (0.0)", "0"
        else:
            p = P[v["family"]]
            lo, hi = v["F4"] * p["strict"], v["F4"] * p["high"]
            adj = f"{lo:.0f}–{hi:.0f} ({100 * lo / n:.1f}–{100 * hi / n:.1f})"
            vals = f"{v['ungrounded'] * p['strict']:.0f}–{v['ungrounded'] * p['high']:.0f}"
        L.append(f"| {v['label']} | {n} | {v['F1']} | {v['F2']} | {v['F4']} ({100 * v['F4'] / n:.1f}) | {adj} | {vals} | "
                 f"{v['F5_grounded']} / {v['F5_ungrounded']} | {v['F3']} |")
    (out / "precision.md").write_text("\n".join(L) + "\n")
    (out / "precision.json").write_text(json.dumps(P, indent=1))
    print("\n".join(L[:8]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(CODE.parent / "data"))
    ap.add_argument("--results", default=str(CODE.parent / "results"))
    ap.add_argument("--review", default=str(CODE.parent / "results" / "fabrication_review"),
                    help="the review folder (sample.jsonl, review_key.jsonl, verdicts.jsonl)")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "fabrication"))
    ap.add_argument("--check-sample", action="store_true", help="re-draw the review sample and compare it")
    a = ap.parse_args()
    results, flagged, rows, tools = fabrication(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = {k: {**v, "sources": dict(v["sources"])} for k, v in results.items()}
    (out / "fabrication.json").write_text(json.dumps(D, indent=1))
    L = ["| agent | calls | any fabrication (F1-F4) | F1 nonexistent tool | F2 nonexistent argument | F3 invalid value | "
         "F4 calls with an ungrounded value | ungrounded values / values judged | underspecified: required value invented "
         "(grounded / ungrounded) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        pc = lambda x: f"{x} ({100 * x / v['calls']:.1f}%)" if v["calls"] else "0"
        L.append(f"| {v['label']} | {v['calls']} | {pc(v['any'])} | {v['F1']} | {v['F2']} | {v['F3']} | {v['F4']} | "
                 f"{v['ungrounded']} / {v['judged']} | {v['F5_grounded']} / {v['F5_ungrounded']} |")
    (out / "fabrication.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    srcs = Counter()
    for v in results.values():
        if v["family"] == "s1":
            srcs.update(v["sources"])
    print("System 1 grounding sources:", dict(srcs))
    print(f"LLM values flagged ungrounded: {len(flagged)}")
    review = Path(a.review)
    if a.check_sample:
        items, key = review_sample(flagged, rows, tools)
        load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        same_items = items == load(review / "sample.jsonl")
        same_key = key == load(review / "review_key.jsonl")
        print(f"review sample re-drawn (seed {SEED}, {len(items)} of {len(flagged)}): sample.jsonl "
              f"{'identical' if same_items else 'DIFFERS'}, review_key.jsonl {'identical' if same_key else 'DIFFERS'}")
    if (review / "verdicts.jsonl").exists():
        random.seed(SEED)
        precision(D, review, out, random.sample(flagged, min(200, len(flagged))), a.results)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
