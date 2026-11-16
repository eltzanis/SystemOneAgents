"""Table 1 of the manuscript: System 1 agents against LLM agents, category by category.

    python scripts/scorecard.py [--out runs/scorecard]

Reads ../results/analysis/metrics.json (scripts/analyse.py) and ../results/fabrication/{fabrication,precision}.json
(scripts/fabrication.py). For each measure it gives the range over the six System 1 agents and over the 18 LLM-agent
settings (nine models, reasoning on and off), for the three accuracy measures the 108-pair tally (System 1 ahead / LLM
ahead / no significant difference; Holm-adjusted P < 0.05), and the values of three leading agents: the most accurate
System 1 agent (pplx-decider-27b), a fast System 1 agent (OpenJev) and the most accurate LLM setting (Qwen3.5-27B,
reasoning off). Invented content of an LLM is the precision-adjusted lower bound. Writes scorecard.md and scorecard.json.
"""
import argparse
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
RES = CODE.parent / "results"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "runs" / "scorecard"))
    out = Path(ap.parse_args().out)
    M = json.loads((RES / "analysis" / "metrics.json").read_text())
    F = json.loads((RES / "fabrication" / "fabrication.json").read_text())
    P = json.loads((RES / "fabrication" / "precision.json").read_text())
    S, n = M["systems"], M["n"]
    s1 = [k for k, v in S.items() if v.get("family") == "s1"]
    llm = [k for k, v in S.items() if v.get("family") in ("think", "nothink")]
    assert len(s1) == 6 and len(llm) == 18

    def pct(x, d):
        return 100 * x / d

    measures = [
        # (group, name, unit, function key -> value, higher is better, tally key)
        ("Accuracy", "Fully correct requests", f"% of {n['main']:,}", lambda k: pct(S[k]["full"], n["main"]), True, "full"),
        ("Accuracy", "Tool calling", f"% of {n['main']:,}", lambda k: pct(S[k]["tool"], n["main"]), True, "tool"),
        ("Accuracy", "Argument filling", f"% of {n['values']:,} values", lambda k: pct(S[k]["vals"], n["values"]), True, "vals"),
        ("Reliability", "Calls with invented content", "% of calls (LLM: adjusted, lower bound)",
         lambda k: 0.0 if S[k]["family"] == "s1" else pct(F[k]["F4"] * P[S[k]["family"]]["strict"], F[k]["calls"]), False, None),
        ("Reliability", "Calls with argument names absent from the schema", "calls",
         lambda k: F[k]["F2"], False, None),
        ("Reliability", "Underspecified requests handled correctly", f"% of {n['asks']}",
         lambda k: pct(S[k]["ask"].get("right tool, missing value left out (correct)", 0), n["asks"]), True, None),
        ("Reliability", "Underspecified requests with an invented required value", f"% of {n['asks']}",
         lambda k: pct(F[k]["F5_ungrounded"], n["asks"]), False, None),
        ("Reliability", "Out-of-scope requests declined", f"% of {n['oos']}",
         lambda k: pct(S[k]["tool_oos"], n["oos"]), True, None),
        ("Reliability", "Complete requests left without a call", f"% of {n['complete']:,}",
         lambda k: pct(S[k]["no_call_in_scope"], n["complete"]), False, None),
        ("Reliability", "Calls with a schema-invalid value", "% of calls",
         lambda k: pct(F[k]["F3"], F[k]["calls"]), False, None),
        ("Efficiency", "GPU energy per fully correct request", "Wh", lambda k: S[k]["cost"]["wh_per_full"], False, None),
        ("Efficiency", "Time per request", "s (mean)", lambda k: S[k]["cost"]["seconds_mean"], False, None),
        ("Efficiency", "Generated tokens per request", "tokens", lambda k: S[k]["cost"]["generated_tokens"], False, None),
        ("Efficiency", "Peak GPU memory", "MiB", lambda k: S[k]["cost"]["mem_peak_mib"], False, None),
    ]
    fmt = {"Wh": "{:.3f}", "s (mean)": "{:.1f}", "MiB": "{:,.0f}", "tokens": "{:.0f}", "calls": "{:.0f}"}
    lead = ["s1:pplx-decider-27b-embed", "s1:openjev", "nothink:qwen3.5-27b"]   # most accurate S1, fast S1, most accurate LLM
    rows, js = [], []
    for group, name, unit, f, higher, tally in measures:
        a = {k: f(k) for k in s1}
        b = {k: f(k) for k in llm}
        form = fmt.get(unit, "{:.1f}")
        t = S.get(f"primary|{tally}") if tally else None
        rec = dict(group=group, measure=name, unit=unit, s1_min=min(a.values()), s1_max=max(a.values()),
                   llm_min=min(b.values()), llm_max=max(b.values()), pairs=t, **{k: f(k) for k in lead})
        js.append(rec)
        rng = lambda lo, hi: form.format(lo) if form.format(lo) == form.format(hi) else f"{form.format(lo)}–{form.format(hi)}"
        rows.append(f"| {group} | {name} ({unit}) | {rng(rec['s1_min'], rec['s1_max'])} | {rng(rec['llm_min'], rec['llm_max'])} | "
                    f"{'%d / %d / %d' % (t['S1'], t['LLM'], t['ns']) if t else '—'} | "
                    + " | ".join(form.format(rec[k]) for k in lead) + " |")
    out.mkdir(parents=True, exist_ok=True)
    head = ["| category | measure | System 1, six agents | LLM, 18 settings | pairs: S1 / LLM / n.s. | pplx-decider-27b | OpenJev | Qwen3.5-27B, reasoning off |",
            "|---|---|---|---|---|---|---|---|"]
    (out / "scorecard.md").write_text("\n".join(head + rows) + "\n")
    (out / "scorecard.json").write_text(json.dumps(js, indent=1))
    print("\n".join(head + rows))


if __name__ == "__main__":
    main()
