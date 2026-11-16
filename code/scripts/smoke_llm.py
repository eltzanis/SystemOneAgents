"""Technical check of one LLM behind its running llama-server: does a tool call come back parsed, in both conditions?

    python scripts/smoke_llm.py qwen3.8-27b --data R --catalogue C

Three requests of R (the first complete in-scope request of each of three servers) are asked once per condition,
outside the answer store. Printed per answer: its kind and function, reasoning length, tokens, prompt tokens processed
(a small number on the second request means the tool prefix was reused from the cache), seconds.

Pass (exit 0): no server error, and in each condition at least one answer carries a parsed function call. This is not an
accuracy test: it finds models that do not run, or whose calls llama.cpp leaves unparsed in the answer text. Warnings:
fewer than 3 parsed calls, no reasoning text with thinking on, reasoning text with thinking off, the tool prefix not
reused, an answer that hit max_tokens.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

from llm_agent import tools as T  # noqa: E402
from llm_agent.agent import LLMAgent, model_config  # noqa: E402
from llm_agent.score import interpret, usage  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--catalogue", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    picks, servers = [], set()
    for r in rows:
        if r.get("tool") and not r.get("ask") and r["tool"].split("/")[0] not in servers and len(picks) < 3:
            picks.append(r); servers.add(r["tool"].split("/")[0])
    fns, by_name, _ = T.build(a.catalogue)
    m = model_config(a.model)
    ok, warn = True, []
    for cond in ("think", "nothink"):
        agent = LLMAgent(m, cond, fns)
        calls = 0
        for i, r in enumerate(picks):
            e = agent.run(r["request"])
            it, tok = interpret(e, by_name), usage(e)
            turn = e["turns"][0]
            print(f"[{cond}] {r['id']}: kind {it['kind']}, function {it['fn']}, reasoning {tok['reasoning_chars']} chars, "
                  f"{tok['completion']} tokens, prompt {tok['prompt']} tokens ({tok['prompt_processed']} processed), "
                  f"{turn['seconds']:.1f} s, finish {turn.get('finish_reason')}"
                  f"{'; problems ' + str(it['problems']) if it['problems'] else ''}", flush=True)
            if it["kind"] == "error":
                print(f"   SERVER ERROR: {it['error'][:2000]}"); ok = False
            elif it["kind"] == "call":
                calls += 1
            if cond == "think" and tok["reasoning_chars"] == 0:
                warn.append(f"{cond} {r['id']}: no reasoning text with thinking on")
            if cond == "nothink" and tok["reasoning_chars"] > 0:
                warn.append(f"{cond} {r['id']}: reasoning text with thinking off ({tok['reasoning_chars']} chars)")
            if turn.get("finish_reason") == "length":
                warn.append(f"{cond} {r['id']}: hit max_tokens")
            if i > 0 and tok["prompt"] and tok["prompt_processed"] > 0.5 * tok["prompt"]:
                warn.append(f"{cond} {r['id']}: {tok['prompt_processed']} of {tok['prompt']} prompt tokens processed "
                            "(the tool prefix was not reused: slow)")
        if calls == 0:
            print(f"!! [{cond}] no answer carries a parsed function call"); ok = False
        elif calls < len(picks):
            warn.append(f"{cond}: {calls} of {len(picks)} answers carry a parsed function call")
    for w in warn:
        print("warning:", w)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
