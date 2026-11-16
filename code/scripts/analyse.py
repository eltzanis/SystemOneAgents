"""The comparison of the paper: the System 1 agent (six decision models) against the LLM agents (nine LLMs, reasoning on
and off), on the released evaluation set and results.

    python scripts/analyse.py [--data ../data] [--results ../results] [--out ../results/analysis]
    python scripts/analyse.py --sensitivity          # also the scoring-rule variants -> <out>/sensitivity.md

Every record is scored again here, by one scorer (s1eval.scoring), against the labels of data/requests:
  System 1, lexical  from the call each record keeps (tool, arguments, arguments set, missing required arguments)
  LLM agents         from the saved completions (llm_agent.score.record)
Requests: data/requests/imaging_test.jsonl, external.jsonl (imaging catalogue) and biomedical_test.jsonl (biomedical
catalogue), pooled. Measures:
  tool decisions   complete in-scope requests (labelled tool, every required value given) and out-of-scope requests:
                   the right tool for an in-scope request, no call for an out-of-scope one
  stated values    every stated value of the complete requests: right when the right tool was called and the value is
                   right (server-equivalent) or omitted where the server auto-detects it; a wrong tool fills none
  fully correct    the same requests as tool decisions: call_correct, or out of scope declined (the primary measure)
Tests: each System 1 model against each LLM in both modes, 6 x 9 x 2 = 108 pairs per measure, one Holm correction per
measure; exact McNemar (tool decisions, fully correct), exact paired sign-flip test with the request as the unit (stated
values). Also: each LLM in its better mode (chosen on the results, which favours the LLMs), same-backbone pairs,
subcategories (source, task group, out-of-scope kind, phrasing tag; descriptive), the underspecified requests apart,
cost per request and per fully correct request, and the scorer check (rescored against stored System 1 verdicts).
Sensitivity (--sensitivity): the strict search-text rule (s1eval.equivalence.SEARCH_TEXT_RULE off) and PyCaret's own
model names for the regression server's model lists.
Writes <out>/tables.md and <out>/metrics.json. No request text is printed.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))  # run from a checkout without installing

import numpy as np  # noqa: E402

S1 = ["decider-4b", "pplx-decider-flash", "decider-35b-a3b", "openjev", "clef", "pplx-decider-27b-embed"]
S1_LABEL = {"decider-4b": "decider-4b", "pplx-decider-flash": "pplx-decider-flash", "decider-35b-a3b": "decider-35b-a3b",
            "openjev": "OpenJev", "clef": "Clef", "pplx-decider-27b-embed": "pplx-decider-27b"}
MODES = ("think", "nothink")
SETS = {"imaging_test": "imaging_tools.json", "external": "imaging_tools.json", "biomedical_test": "biomedical_tools.json"}
ASK = ["no tool call", "wrong tool", "right tool, missing value invented", "right tool, missing value left out (correct)",
       "right tool, other"]
VERDICT_KEYS = ("exact", "call_correct", "args_equivalent", "args_conditional")


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------------------------------------------------ scoring
def rescore_s1(rec, row, tools):
    """A System 1 (or lexical) record scored again against `row`'s label, from the call it keeps."""
    from s1eval.scoring import argument_verdicts
    gold = row.get("tool")
    out = {k: rec[k] for k in ("id", "pred", "confidence", "seconds", "model_calls", "cost") if k in rec}
    out.update(gold=gold, exact=rec["pred"] == gold)
    if row.get("ask"):
        out["ask"] = row["ask"]
    if gold and out["exact"]:
        props = tools[gold].input_schema.get("properties", {})
        labels = row.get("args", {})
        decided = rec.get("arg_decisions")
        extra = (sorted(d["name"] for d in decided if d["status"] == "set" and d["name"] not in labels)
                 if decided is not None else rec.get("args_extra", []))
        out.update(args_pred=rec["args_pred"], args_extra=extra, args_missing=rec.get("args_missing", []),
                   **argument_verdicts(tools[gold].name, props, labels, rec["args_pred"], extra,
                                       rec.get("args_missing", []), row.get("ask", [])))
    return out


def differs(a, b):
    return any(a.get(k) != b.get(k) for k in VERDICT_KEYS)


def systems(a, rows):
    """{key: dict(label, family, model, recs={(set, id): record}, check={set: n differing})}."""
    from llm_agent import tools as T
    from llm_agent.agent import all_models
    from llm_agent.score import record
    from s1agent.catalog import load_catalogue
    from s1agent.config import agent_settings
    cats = {k: Path(a.data) / "catalogues" / c for k, c in SETS.items()}
    tools = {k: {t.qualified: t for t in load_catalogue(c)} for k, c in cats.items()}
    ids = {k: {i for (kk, i) in rows if kk == k} for k in SETS}
    out = {}
    res = Path(a.results)
    for m in S1 + ["lexical"]:
        recs, check = {}, {}
        for k in SETS:
            stored = {r["id"]: r for r in load(res / "system1" / "records" / k / f"{m}.jsonl")}
            if set(stored) != ids[k]:
                sys.exit(f"STOP: {m} has {len(stored)} records on {k}, the data {len(ids[k])}")
            for i, r in stored.items():
                recs[(k, i)] = rescore_s1(r, rows[(k, i)], tools[k])
            check[k] = sum(differs(recs[(k, i)], stored[i]) for i in stored)
        label = "lexical baseline (no model)" if m == "lexical" else f"System 1 {S1_LABEL[m]}"
        out[f"s1:{m}"] = dict(label=label, family="lexical" if m == "lexical" else "s1", model=m, recs=recs, check=check)

    cfg = all_models()["models"]
    built = {k: T.build(c) for k, c in cats.items()}
    ws = str(Path(agent_settings()["workspace"]).resolve())
    for m, e in cfg.items():
        for c in MODES:
            recs = {}
            for k in SETS:
                base = res / "llm" / k
                key = json.loads((base / f"{m}__{c}.summary.json").read_text())["settings"]
                ans = {x["id"]: x for x in load(base / f"{m}__{c}.jsonl") if x.get("settings") == key}
                if set(ans) != ids[k]:
                    sys.exit(f"STOP: {m} [{c}] has {len(set(ans) & ids[k])} of {len(ids[k])} completions on {k}")
                _, by_name, policy = built[k]
                for i in ids[k]:
                    recs[(k, i)] = record(rows[(k, i)], ans[i], by_name, policy, ws)
            out[f"{c}:{m}"] = dict(label=f"{e.get('label', m)} [{c}]", family=c, model=m,
                                   backbone_of=e.get("backbone_of", []), recs=recs, check={})
    return out


# ----------------------------------------------------------------------------------------------------------- measures
def value_ok(rec, arg):
    if not rec.get("exact"):
        return False
    eq = rec.get("args_equivalent")
    if eq is None:
        return False
    return bool(eq.get(arg)) or arg in rec.get("args_conditional", [])


def mcnemar(a, b):
    """Exact two-sided McNemar on paired verdicts: (only a right, only b right, p)."""
    x = sum(1 for u in a if a[u] and not b[u]); y = sum(1 for u in a if b[u] and not a[u]); n = x + y
    if n == 0:
        return x, y, 1.0
    return x, y, min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n)


def signflip(da, db):
    """Exact paired sign-flip (randomisation) test on per-request counts {u: int}: under the null hypothesis each
    request's difference a - b is equally likely to have either sign; two-sided p = P(|sum| >= |observed sum|), from the
    exact distribution of the sum (convolution over the requests with a nonzero difference)."""
    d = [abs(da[u] - db[u]) for u in da if da[u] != db[u]]
    if not d:
        return 1.0
    obs = abs(sum(da[u] - db[u] for u in da))
    tot = sum(d)
    dist = np.zeros(2 * tot + 1); dist[tot] = 1.0                 # index = sum + tot
    for x in d:
        dist = 0.5 * (np.roll(dist, x) + np.roll(dist, -x))      # no wrap-around: |partial sum| <= tot
    sums = np.arange(-tot, tot + 1)
    return float(min(1.0, dist[np.abs(sums) >= obs].sum()))


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pct(a, b):
    return 100 * a / b if b else float("nan")


def ask_behaviour(rec, r):
    if rec["pred"] is None:
        return "no tool call"
    if not rec["exact"]:
        return "wrong tool"
    if rec.get("call_correct"):
        return "right tool, missing value left out (correct)"
    if any(k in (rec.get("args_pred") or {}) and k not in rec.get("args_policy", []) for k in r["ask"]):
        return "right tool, missing value invented"
    return "right tool, other"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def tally(D, metric, llms, s1):
    return Counter("S1" if D[(metric, l, s_)]["p_holm"] < 0.05 and D[(metric, l, s_)]["diff_pp"] > 0 else
                   "LLM" if D[(metric, l, s_)]["p_holm"] < 0.05 else "ns" for l in llms for s_ in s1)


# --------------------------------------------------------------------------------------------------------------- main
def analyse(a):
    from s1eval import equivalence
    notes = []
    if a.strict_search_text:
        equivalence.SEARCH_TEXT_RULE = False
        notes.append("scoring variant: strict search-text rule")
    if a.pycaret_lists:
        equivalence.use_pycaret_lists()
        notes.append("scoring variant: PyCaret's model names for the regression server's model lists")
    rows = {(k, r["id"]): r for k in SETS for r in load(Path(a.data) / "requests" / f"{k}.jsonl")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    SY = systems(a, rows)

    COMPLETE = [u for u, r in rows.items() if r.get("tool") and not r.get("ask")]
    OOS = [u for u, r in rows.items() if not r.get("tool")]
    ASKS = [u for u, r in rows.items() if r.get("tool") and r.get("ask")]
    MAIN = COMPLETE + OOS
    VALUES = [(k, i, x) for (k, i) in COMPLETE for x in rows[(k, i)].get("args", {})]

    def verdicts(s):
        R = s["recs"]
        tool = {u: (bool(R[u]["exact"]) if rows[u].get("tool") else R[u]["pred"] is None) for u in MAIN}
        full = {u: (bool(R[u].get("call_correct")) if rows[u].get("tool") else R[u]["pred"] is None) for u in MAIN}
        vals = {v: value_ok(R[v[:2]], v[2]) for v in VALUES}
        per_req = {u: sum(vals[(u[0], u[1], x)] for x in rows[u].get("args", {})) for u in COMPLETE}
        return dict(tool=tool, full=full, vals=vals, per_req=per_req)

    V = {k: verdicts(s) for k, s in SY.items()}
    s1 = [k for k, s in SY.items() if s["family"] == "s1"]
    think = [k for k, s in SY.items() if s["family"] == "think"]
    nothink = [k for k, s in SY.items() if s["family"] == "nothink"]
    lexical = [k for k, s in SY.items() if s["family"] == "lexical"]
    M, T, D = {}, [], {}
    nM, nV = len(MAIN), len(VALUES)

    # ---- each system
    for k, s in SY.items():
        v, R = V[k], s["recs"]
        m = dict(label=s["label"], family=s["family"], check=s["check"])
        m["tool"] = sum(v["tool"].values()); m["tool_ci"] = wilson(m["tool"], nM)
        m["tool_complete"] = sum(v["tool"][u] for u in COMPLETE); m["tool_oos"] = sum(v["tool"][u] for u in OOS)
        m["wrong_tool"] = sum(1 for u in COMPLETE if R[u]["pred"] is not None and not R[u]["exact"])
        m["no_call_in_scope"] = sum(1 for u in COMPLETE if R[u]["pred"] is None)
        m["vals"] = sum(v["vals"].values())
        m["extras"] = sum(len(R[u].get("args_extra") or []) for u in COMPLETE if R[u]["exact"])
        m["full"] = sum(v["full"].values()); m["full_ci"] = wilson(m["full"], nM)
        m["ask"] = dict(Counter(ask_behaviour(R[u], rows[u]) for u in ASKS))
        cs = [r.get("cost") or {} for r in R.values()]
        if any(cs):
            secs = [c.get("seconds") for c in cs]
            main_e = sum((R[u].get("cost") or {}).get("gpu_energy_j") or 0 for u in MAIN)
            main_s = sum((R[u].get("cost") or {}).get("seconds") or 0 for u in MAIN)
            m["cost"] = dict(
                n=sum(1 for c in cs if c), seconds_mean=mean(secs),
                seconds_median=statistics.median([x for x in secs if x is not None]) if any(x is not None for x in secs) else None,
                energy_j=mean(c.get("gpu_energy_j") for c in cs), above_idle_j=mean(c.get("above_idle_j") for c in cs),
                power_w=mean(c.get("power_w") for c in cs), idle_w=mean(c.get("idle_w") for c in cs),
                mem_peak_mib=max((c.get("mem_used_mib") or 0) for c in cs),
                prompt_tokens=mean(c.get("prompt_tokens") for c in cs), generated_tokens=mean(c.get("predicted_tokens") for c in cs),
                reasoning_tokens=mean(c.get("reasoning_tokens") for c in cs),
                context_tokens=mean((r.get("tokens") or {}).get("prompt") for r in R.values()) if s["family"] in MODES else None,
                model_calls=mean(r.get("model_calls") for r in R.values()) if s["family"] not in MODES else None,
                cpu_energy_j=mean(c.get("cpu_energy_j") for c in cs),
                total_wh=sum(c.get("gpu_energy_j") or 0 for c in cs) / 3600, total_h=sum(x or 0 for x in secs) / 3600,
                wh_per_full=main_e / 3600 / m["full"] if m["full"] else None,
                s_per_full=main_s / m["full"] if m["full"] else None)
        if s["family"] in MODES:
            m["errors"] = sum(1 for r in R.values() if r.get("kind") == "error")
            m["no_call"] = sum(1 for r in R.values() if r.get("kind") == "none")
        M[k] = m

    def stars(q):
        return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""

    def ptab(title, keys):
        T.extend([f"\n### {title}\n",
                  f"| system | tool decisions right /{nM} (95% CI) | in scope, right tool /{len(COMPLETE)} | out of scope "
                  f"declined /{len(OOS)} | wrong tool | no call (in scope) | stated values right /{nV} | fully correct "
                  f"/{nM} (95% CI) | extras (right tool) |", "|" + "---|" * 9])
        for k in keys:
            m = M[k]
            T.append(f"| {m['label']} | {m['tool']} ({pct(m['tool'], nM):.1f}; {m['tool_ci'][0]:.1f}–{m['tool_ci'][1]:.1f}) | "
                     f"{m['tool_complete']} | {m['tool_oos']} | {m['wrong_tool']} | {m['no_call_in_scope']} | "
                     f"{m['vals']} ({pct(m['vals'], nV):.1f}) | {m['full']} ({pct(m['full'], nM):.1f}; "
                     f"{m['full_ci'][0]:.1f}–{m['full_ci'][1]:.1f}) | {m['extras']} |")

    T.append("# The System 1 agent against the LLM agents\n")
    if notes:
        T.append("**" + "; ".join(notes) + "**\n")
    T.append(f"Requests, pooled over both fields and every source: {len(COMPLETE)} complete in scope + {len(OOS)} out of "
             f"scope = {nM} (tool decisions, fully correct); {nV} stated values; {len(ASKS)} underspecified apart. "
             f"Sets: " + ", ".join(f"{k} {sum(u[0] == k for u in MAIN)}" for k in SETS) + ".")
    T.append("\n## 1. Each system")
    ptab("System 1 (six decision models) and the lexical baseline", s1 + lexical)
    ptab("LLM agents, thinking on", think)
    ptab("LLM agents, thinking off", nothink)

    # ---- primary: both modes, 108 pairs, one Holm per measure
    T.append("\n## 2. Each System 1 model against each LLM in both modes, one Holm correction per measure over the "
             f"{len(s1) * len(think + nothink)} pairs (System 1 minus LLM in percentage points; Holm-adjusted * < 0.05, "
             "** < 0.01, *** < 0.001)")
    for metric, unit in (("full", f"fully correct requests (/{nM}, exact McNemar): the primary measure"),
                         ("tool", f"tool decisions (/{nM}, exact McNemar)"),
                         ("vals", f"stated values (/{nV}, exact paired sign-flip test by request)")):
        pairs, ps = [], []
        for l in think + nothink:
            for s_ in s1:
                if metric == "vals":
                    p = signflip(V[s_]["per_req"], V[l]["per_req"]); diff = pct(M[s_]["vals"] - M[l]["vals"], nV); extra = {}
                else:
                    x, y, p = mcnemar(V[s_][metric], V[l][metric]); diff = pct(M[s_][metric] - M[l][metric], nM)
                    extra = dict(s1_only=x, llm_only=y)
                pairs.append((l, s_, diff, extra)); ps.append(p)
        for (l, s_, diff, extra), p, q in zip(pairs, ps, holm(ps)):
            D[(metric, l, s_)] = dict(diff_pp=diff, p=p, p_holm=q, **extra)
        T += [f"\n### {unit}\n", "| LLM \\ System 1 | " + " | ".join(SY[s_]["label"].replace("System 1 ", "") for s_ in s1)
              + " | S1 ahead / LLM ahead / n.s. |", "|" + "---|" * (len(s1) + 2)]
        for l in think + nothink:
            cells, t = [], Counter()
            for s_ in s1:
                d = D[(metric, l, s_)]; st = stars(d["p_holm"])
                cells.append(f"{d['diff_pp']:+.1f}{st}")
                t["S1" if st and d["diff_pp"] > 0 else "LLM" if st else "ns"] += 1
            T.append(f"| {SY[l]['label']} | " + " | ".join(cells) + f" | {t['S1']} / {t['LLM']} / {t['ns']} |")
        tot = tally(D, metric, think + nothink, s1)
        by_mode = {"think": tally(D, metric, think, s1), "nothink": tally(D, metric, nothink, s1)}
        T.append(f"\n**All {len(pairs)} pairs: System 1 ahead {tot['S1']}, LLM ahead {tot['LLM']}, no significant "
                 f"difference {tot['ns']}** (within this correction: thinking on {by_mode['think']['S1']} / "
                 f"{by_mode['think']['LLM']} / {by_mode['think']['ns']}; thinking off {by_mode['nothink']['S1']} / "
                 f"{by_mode['nothink']['LLM']} / {by_mode['nothink']['ns']}).")
        M[f"primary|{metric}"] = dict(tot)
        M[f"primary|{metric}|by_mode"] = {f: dict(c) for f, c in by_mode.items()}

    # ---- better mode (favours the LLMs)
    llm_models = sorted({SY[l]["model"] for l in think + nothink})
    best = {m: max(MODES, key=lambda c: (M[f"{c}:{m}"]["full"], c == "think")) for m in llm_models}
    T.append("\n## 3. Each LLM in its better mode (chosen on fully correct, which favours the LLMs; Holm over the "
             f"{len(s1) * len(llm_models)} pairs of each measure)\n")
    T.append("Better mode: " + ", ".join(f"{m} {best[m]} ({M[f'think:{m}']['full']} vs {M[f'nothink:{m}']['full']})"
                                         for m in llm_models) + ".\n")
    for metric in ("full", "tool", "vals"):
        res = []
        for m in llm_models:
            l = f"{best[m]}:{m}"
            for s_ in s1:
                if metric == "vals":
                    res.append((sum(V[s_]["per_req"].values()) - sum(V[l]["per_req"].values()),
                                signflip(V[s_]["per_req"], V[l]["per_req"])))
                else:
                    x, y, p = mcnemar(V[s_][metric], V[l][metric]); res.append((x - y, p))
        tot = Counter("S1" if q < 0.05 and d > 0 else "LLM" if q < 0.05 else "ns"
                      for (d, _), q in zip(res, holm([p for _, p in res])))
        M[f"better_mode|{metric}"] = dict(tot)
        T.append(f"- {metric}: System 1 ahead {tot['S1']}, LLM ahead {tot['LLM']}, no significant difference {tot['ns']}")

    # ---- same backbone
    T.append("\n## 4. Same backbone (each LLM against the System 1 model built on it; p unadjusted)\n")
    for l in think + nothink:
        for b in SY[l].get("backbone_of", []):
            s_ = f"s1:{b}"
            parts = []
            for metric in ("tool", "vals", "full"):
                p = signflip(V[s_]["per_req"], V[l]["per_req"]) if metric == "vals" else mcnemar(V[s_][metric], V[l][metric])[2]
                n = nV if metric == "vals" else nM
                parts.append(f"{metric} {M[s_][metric]} vs {M[l][metric]} ({pct(M[s_][metric] - M[l][metric], n):+.1f} pp, p={p:.2g})")
            T.append(f"- {SY[s_]['label']} vs {SY[l]['label']}: " + "; ".join(parts))

    # ---- subcategories
    def by(pred):
        return [u for u in MAIN if pred(u)], [v for v in VALUES if pred(v[:2])]
    SUB = {"source: generated (imaging and biomedical test sets)": by(lambda u: u[0] in ("imaging_test", "biomedical_test")),
           "source: mAIstro (external)": by(lambda u: u[0] == "external" and rows[u].get("source") == "maistro"),
           "source: ReCLAIM (external)": by(lambda u: u[0] == "external" and rows[u].get("source") == "reclaim")}
    for g in sorted({rows[u]["group"] for u in COMPLETE if rows[u].get("group")}):
        SUB[f"group: {g}"] = by(lambda u, g=g: rows[u].get("tool") and rows[u].get("group") == g)
    for kind in ("oos-near", "oos-far", "oos-modality"):
        SUB[f"out of scope: {kind}"] = by(lambda u, kind=kind: not rows[u].get("tool") and kind in rows[u].get("tags", []))
    tags = Counter(t for u in COMPLETE for t in rows[u].get("tags", []))
    for t, n in sorted(tags.items(), key=lambda x: -x[1]):
        if n >= 10:
            SUB[f"tag: {t}"] = by(lambda u, t=t: rows[u].get("tool") and t in rows[u].get("tags", []))
    T.append("\n## 5. Subcategories (mean [min–max] in % over the systems of each family; Δ = System 1 mean minus LLM "
             "mean, pp; descriptive, no test)\n")
    T.append("| subcategory | requests | values | tool: System 1 | tool: LLM think | Δ | tool: LLM nothink | Δ | values: "
             "System 1 | values: LLM think | Δ | values: LLM nothink | Δ |"); T.append("|" + "---|" * 13)
    SM = {}

    def rates(keys, metric, units):
        return [100 * sum(V[k][metric][u] for u in units) / len(units) for k in keys] if units and keys else None

    def f(r):
        return "—" if r is None else f"{statistics.fmean(r):.1f} [{min(r):.0f}–{max(r):.0f}]"

    def dlt(x, y):
        return "—" if x is None or y is None else f"{statistics.fmean(x) - statistics.fmean(y):+.1f}"
    for name, (tu, vu) in SUB.items():
        if not tu:
            continue
        cell = {(metric, fam): rates(keys, metric, units) for metric, units in (("tool", tu), ("vals", vu))
                for fam, keys in (("s1", s1), ("think", think), ("nothink", nothink))}
        SM[name] = dict(n_requests=len(tu), n_values=len(vu), **{f"{x}|{y}": c for (x, y), c in cell.items()})
        T.append(f"| {name} | {len(tu)} | {len(vu)} | {f(cell[('tool', 's1')])} | {f(cell[('tool', 'think')])} | "
                 f"{dlt(cell[('tool', 's1')], cell[('tool', 'think')])} | {f(cell[('tool', 'nothink')])} | "
                 f"{dlt(cell[('tool', 's1')], cell[('tool', 'nothink')])} | {f(cell[('vals', 's1')])} | "
                 f"{f(cell[('vals', 'think')])} | {dlt(cell[('vals', 's1')], cell[('vals', 'think')])} | "
                 f"{f(cell[('vals', 'nothink')])} | {dlt(cell[('vals', 's1')], cell[('vals', 'nothink')])} |")

    # ---- underspecified
    T.append(f"\n## 6. Underspecified requests ({len(ASKS)}, pooled; correct = right tool, stated values right, the "
             "missing values left unfilled)\n")
    T.append("| system | " + " | ".join(ASK) + " |"); T.append("|" + "---|" * (len(ASK) + 1))
    for k in s1 + lexical + think + nothink:
        T.append(f"| {SY[k]['label']} | " + " | ".join(str(M[k]["ask"].get(c, 0)) for c in ASK) + " |")

    # ---- cost
    T.append(f"\n## 7. Cost per request (every request of every set, {len(rows)}; whole-board GPU energy) and per fully "
             f"correct request (over the {nM})\n")
    T.append("| system | s/request mean (median) | GPU energy J/request (above idle) | context tokens (LLM) | prompt tokens "
             "computed | generated tokens (reasoning) | model calls | Wh total | hours total | Wh per fully correct | "
             "s per fully correct | mean power W (idle W) | peak GPU memory MiB |"); T.append("|" + "---|" * 13)

    def fm(x, d=0):
        return "—" if x is None else f"{x:.{d}f}"
    for k in s1 + think + nothink:
        c = M[k].get("cost")
        if c:
            T.append(f"| {SY[k]['label']} | {fm(c['seconds_mean'], 2)} ({fm(c['seconds_median'], 2)}) | {fm(c['energy_j'])} "
                     f"({fm(c['above_idle_j'])}) | {fm(c['context_tokens'])} | {fm(c['prompt_tokens'])} | "
                     f"{fm(c['generated_tokens'])} ({fm(c['reasoning_tokens'])}) | {fm(c['model_calls'], 1)} | "
                     f"{fm(c['total_wh'], 1)} | {fm(c['total_h'], 2)} | {fm(c['wh_per_full'], 3)} | {fm(c['s_per_full'], 2)} | "
                     f"{fm(c['power_w'])} ({fm(c['idle_w'])}) | {fm(c['mem_peak_mib'])} |")

    # ---- checks
    T.append("\n## 8. Checks\n")
    T.append("Scorer check: System 1 records whose rescored verdicts (tool, call_correct, value verdicts) differ from the "
             "verdicts stored with the run, per set.\n")
    T.append("| system | " + " | ".join(SETS) + " |"); T.append("|" + "---|" * (len(SETS) + 1))
    for k in s1 + lexical:
        T.append(f"| {SY[k]['label']} | " + " | ".join(str(SY[k]["check"].get(x, "—")) for x in SETS) + " |")
    T.append("\nLLM completions (all requests of every set): " + "; ".join(
        f"{SY[k]['label']}: errors {M[k]['errors']}, no call {M[k]['no_call']}" for k in think + nothink))

    (out / "tables.md").write_text("\n".join(T) + "\n")
    json.dump(dict(notes=notes, n=dict(main=nM, values=nV, complete=len(COMPLETE), oos=len(OOS), asks=len(ASKS)),
                   systems=M, differences={f"{x}|{y}|{z}": v for (x, y, z), v in D.items()}, subcategories=SM),
              open(out / "metrics.json", "w"), indent=1, default=str)
    print("\n".join(T))
    print(f"\n-> {out / 'tables.md'}, {out / 'metrics.json'}")


def sensitivity(a):
    """The main analysis and each scoring-rule variant, each a run of this script; sensitivity.md compares the tallies."""
    out = Path(a.out)
    runs = {"main": (out, []), "strict search text": (out / "variants" / "strict_search_text", ["--strict-search-text"]),
            "PyCaret model lists": (out / "variants" / "pycaret_lists", ["--pycaret-lists"])}
    L = ["# Sensitivity of the comparison to the scoring rules", "",
         "Each row is a full run of scripts/analyse.py; tallies are System 1 ahead / LLM ahead / no significant "
         "difference (108 pairs per measure, one Holm correction per measure).", "",
         "| variant | fully correct | tool decisions | stated values |", "|---|---|---|---|"]
    for name, (d, flags) in runs.items():
        r = subprocess.run([sys.executable, "-I", __file__, "--data", a.data, "--results", a.results, "--out", str(d)]
                           + flags, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"STOP: variant {name!r} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        m = json.loads((d / "metrics.json").read_text())["systems"]

        def t3(c):
            return f"{c.get('S1', 0)} / {c.get('LLM', 0)} / {c.get('ns', 0)}"
        L.append(f"| {name} | {t3(m['primary|full'])} | {t3(m['primary|tool'])} | {t3(m['primary|vals'])} |")
        print(f"{name}: done", flush=True)
    (out / "sensitivity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(CODE.parent / "data"), help="the evaluation set (requests/, catalogues/)")
    ap.add_argument("--results", default=str(CODE.parent / "results"), help="the released results (system1/, llm/)")
    ap.add_argument("--out", default=str(CODE.parent / "results" / "analysis"))
    ap.add_argument("--strict-search-text", action="store_true", help="scoring variant: no search-text rule")
    ap.add_argument("--pycaret-lists", action="store_true", help="scoring variant: PyCaret's regression model names")
    ap.add_argument("--sensitivity", action="store_true", help="the main analysis and both scoring variants")
    a = ap.parse_args()
    sensitivity(a) if a.sensitivity else analyse(a)


if __name__ == "__main__":
    main()
