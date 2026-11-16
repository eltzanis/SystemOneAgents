"""The context each LLM needs for a request file: every prompt counted with the model's own template and tokenizer.

    python scripts/measure_context.py --catalogue ../data/catalogues/biomedical_tools.json \
        --data ../data/requests/biomedical_test.jsonl [--models qwen3.5-4b ...] [--out runs/llm_context.json]

For each model (one llama-server at a time, its flags of configs/llm_models.yaml, context 65,536): every request, in both
conditions, is rendered exactly as the agent sends it (llm_agent.agent.LLMAgent.body: the request as the only message,
the catalogue's functions, the condition's template switches) by the server's own chat template (POST /apply-template)
and counted by the server's own tokenizer (POST /tokenize, special tokens added and parsed as for a completion). The
count is checked against the server's usage report of a real completion of the same body with max_tokens = 1 for the
longest requests (--check, default 3 per condition): the difference must be the same constant for each of them (it is
added to every count); otherwise every request is counted by a real completion.

Rule: longest prompt + max_tokens (16,384) <= 65,536: the context stays 65,536; otherwise the smallest multiple of
8,192 that fits (the value of `ctx_by_catalogue` in configs/llm_models.yaml). Prints counts only, never a request.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

import httpx  # noqa: E402
from run_llm import stop, wait_healthy  # noqa: E402

from llm_agent import tools as T  # noqa: E402
from llm_agent.agent import LLMAgent, all_models, model_config  # noqa: E402
from llm_agent.cli import server_command  # noqa: E402

BASE, STEP = 65536, 8192


def rule(longest, max_tokens):
    need = longest + max_tokens
    ctx = BASE if need <= BASE else -(-need // STEP) * STEP
    return need, ctx


def template_count(url, body):
    p = httpx.post(f"{url}/apply-template", json=body, timeout=600)
    p.raise_for_status()
    t = httpx.post(f"{url}/tokenize", json={"content": p.json()["prompt"], "add_special": True, "parse_special": True},
                   timeout=600)
    t.raise_for_status()
    return len(t.json()["tokens"])


def completion_count(url, body):
    r = httpx.post(f"{url}/v1/chat/completions", json={**body, "max_tokens": 1}, timeout=3600)
    r.raise_for_status()
    return int(r.json()["usage"]["prompt_tokens"])


def measure(name, rows, fns, n_check, logs):
    m = model_config(name)
    cmd, env, gguf, port = server_command(name)               # the default context, 65,536
    if not gguf.exists():
        return {"error": f"{gguf} not found"}
    import subprocess
    with open(logs / f"{name}_server.log", "w") as log:
        log.write(" ".join(cmd) + "\n\n"); log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_healthy(proc, port, 900):
            return {"error": f"the server did not come up ({logs / f'{name}_server.log'})"}
        url = f"http://127.0.0.1:{port}"
        out = {"conditions": {}}
        for cond in ("think", "nothink"):
            agent = LLMAgent(m, cond, fns, url=url)
            t0 = time.time()
            counts = {r["id"]: template_count(url, agent.body([{"role": "user", "content": r["request"]}])) for r in rows}
            top = sorted(counts, key=counts.get, reverse=True)[:n_check]
            by_id = {r["id"]: r for r in rows}
            diffs = [completion_count(url, agent.body([{"role": "user", "content": by_id[i]["request"]}])) - counts[i]
                     for i in top]
            method = "apply-template + tokenize"
            if len(set(diffs)) == 1:
                longest = max(counts.values()) + diffs[0]
            else:                                         # the template count does not track the server's: count all
                method = "real completion for every request (template counts did not match)"
                longest = max(completion_count(url, agent.body([{"role": "user", "content": r["request"]}])) for r in rows)
            out["conditions"][cond] = dict(longest_prompt=longest, method=method, check_offsets=diffs,
                                           seconds=round(time.time() - t0, 1))
            print(f"  {name} [{cond}]: longest prompt {longest} tokens ({method}; offsets on the {len(top)} longest "
                  f"{diffs}) in {time.time() - t0:.0f} s", flush=True)
        longest = max(c["longest_prompt"] for c in out["conditions"].values())
        need, ctx = rule(longest, int(m["max_tokens"]))
        out.update(longest_prompt=longest, max_tokens=int(m["max_tokens"]), needed=need, ctx=ctx)
        return out
    finally:
        stop(proc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--models", nargs="+", default=list(all_models()["models"]))
    ap.add_argument("--check", type=int, default=3, help="longest requests checked by a real completion, per condition")
    ap.add_argument("--out", default="runs/llm_context.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    fns, _, _ = T.build(a.catalogue)
    print(f"catalogue {Path(a.catalogue).name}: {len(fns)} functions; {len(rows)} requests of {Path(a.data).name}", flush=True)
    out = Path(a.out)
    logs = out.parent / "llm_context_logs"
    logs.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    for name in a.models:
        print(f"=== {name}  {time.strftime('%F %T')}", flush=True)
        r = measure(name, rows, fns, a.check, logs)
        r.update(catalogue=Path(a.catalogue).name, data=Path(a.data).name, n_requests=len(rows))
        res[name] = r
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"  -> {r['error']}" if "error" in r else
              f"  -> longest {r['longest_prompt']} + {r['max_tokens']} = {r['needed']}: context {r['ctx']}", flush=True)
    sys.exit(1 if any("error" in res.get(n, {"error": 1}) for n in a.models) else 0)


if __name__ == "__main__":
    main()
