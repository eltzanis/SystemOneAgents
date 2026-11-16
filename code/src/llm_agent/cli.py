"""s1llm: the command line of the LLM agent baseline.

    s1llm serve qwen3.8-27b [--catalogue C]                                     llama-server in the foreground
    s1llm run   --model qwen3.8-27b --cond think --data R --catalogue C         ask the running server, then score
    s1llm score --model qwen3.8-27b --cond think --data R --catalogue C         score the saved answers only
    s1llm show  --model qwen3.8-27b --cond think --data R --catalogue C --id ID one saved answer

R: a labelled request file (data/requests/*.jsonl); C: its tool catalogue (data/catalogues/*.json).
Answers: <answers>/<R's name>/<model>__<cond>.jsonl (every completion, reused when the settings are the same; default
runs/llm_answers). Records: <out>/<R's name>/<model>__<cond>.jsonl, one scored record per request, and the summary next
to them (.summary.json; default runs/llm_eval). Each new completion is measured (s1eval.meter: seconds, GPU energy and
utilisation, tokens from the server's counters) after a 20 s idle baseline; the reasoning text is counted in tokens by
the server's /tokenize after the measurement. A request is printed as its id and verdict only, never its text.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

from llm_agent import tools as T
from llm_agent.agent import AnswerStore, LLMAgent, all_models, model_config
from llm_agent.score import record, summary
from s1agent.config import agent_settings


def paths(a):
    stem = Path(a.data).stem
    seed = f"__seed{a.seed}" if a.seed is not None else ""
    store = Path(a.answers) / stem / f"{a.model}__{a.cond}.jsonl"
    out = Path(a.out) / stem / f"{a.model}__{a.cond}{seed}.jsonl"
    return store, out


def with_seed(m, seed):
    return {**m, "seed": seed} if seed is not None else m


def rows_of(a):
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids)]
    return rows[: a.limit] if a.limit else rows


def workspace():
    return str(Path(agent_settings()["workspace"]).resolve())


def score(a, rows, agent, store, by_name, policy):
    _, out = paths(a)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs, ws = [], workspace()
    for r in rows:
        e = store.get(r["id"])
        if e is not None:
            recs.append(record(r, e, by_name, policy, ws))
    with out.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    s = {"model": a.model, "condition": a.cond, "seed": agent.m["seed"], "data": Path(a.data).name, "settings": agent.key,
         "settings_detail": agent.settings, "answered": len(recs), "rows": len(rows), **summary(recs)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
    print(f"\n{a.model} [{a.cond}] {Path(a.data).stem}: {s['answered']}/{s['rows']} answered | complete in-scope requests: "
          f"right tool {s['tool_right']}/{s['complete']}, correct {s['correct']}/{s['complete']} | out of scope "
          f"declined {s['oos_declined']}/{s['oos']} | no call {s['kind_none']} | errors {s['kind_error']} | "
          f"{s['seconds']:.1f} s, {s['completion_tokens']:.0f} tokens per request\nrecords: {out}", flush=True)
    return s


def count_tokens(url, text):
    if not text:
        return 0
    try:
        return len(httpx.post(f"{url}/tokenize", json={"content": text}, timeout=120).json()["tokens"])
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def cmd_run(a, generate=True):
    fns, by_name, policy = T.build(a.catalogue)
    agent = LLMAgent(with_seed(model_config(a.model), a.seed), a.cond, fns, url=a.url)
    store_path, _ = paths(a)
    store = AnswerStore(store_path, agent.key)
    rows = rows_of(a)
    meter = None
    if generate:
        from s1eval.meter import Meter
        ws = workspace()
        todo = [r for r in rows if store.get(r["id"]) is None]
        print(f"{a.model} [{a.cond}] settings {agent.key}: {len(rows) - len(todo)} saved, {len(todo)} to ask", flush=True)
        meter = Meter(agent.url, gpu_index=int(os.environ.get("S1_GPU", "0"))) if todo else None
        if meter:
            print(f"meter: idle {meter.idle()}; {'; '.join(meter.problems) or 'all counters available'}", flush=True)
        for i, r in enumerate(todo, 1):
            with meter.measure() as cost:
                e = agent.run(r["request"])
            reasoning = (e["turns"][0].get("message") or {}).get("reasoning_content")
            cost.update(reasoning_tokens=count_tokens(agent.url, reasoning), idle_w=meter.idle_w)
            e = store.add(r["id"], {**e, "cost": cost})
            rec = record(r, e, by_name, policy, ws)
            ok = bool(rec.get("call_correct")) if r.get("tool") else rec["pred"] is None
            print(f"{'✓' if ok else '✗'} {i:4d}/{len(todo)} {r['id']:14s} {rec['kind']:7s} {rec['seconds']:6.1f}s "
                  f"{rec['tokens']['completion']:6d} tok {cost.get('gpu_energy_j') or 0:7.0f} J"
                  f"{' ERROR ' + rec['error'][:200] if rec['error'] else ''}", flush=True)
    if meter:
        meter.close()
    return score(a, rows, agent, store, by_name, policy)


def cmd_show(a):
    fns, _, _ = T.build(a.catalogue)
    agent = LLMAgent(with_seed(model_config(a.model), a.seed), a.cond, fns)
    store_path, _ = paths(a)
    e = AnswerStore(store_path, agent.key).get(a.id[0])
    print(json.dumps(e, indent=1, ensure_ascii=False) if e else f"no saved answer for {a.id[0]} with settings {agent.key}")


def server_command(name, catalogue=None):
    """The llama-server command of an LLM -> (argv, env, gguf path, port). With the catalogue the model will be asked
    with, the context of `ctx_by_catalogue` replaces the default one."""
    from llm_agent import LLM_MODELS_YAML
    cfg, m = all_models(), model_config(name)
    root = (LLM_MODELS_YAML.parent / cfg.get("root", ".")).resolve()

    def res(p):
        p = Path(p).expanduser()
        return p if p.is_absolute() else root / p
    exe = res(cfg["llama_server"])
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join([str(exe.parent), str(res(cfg["cuda_libs"])),
                                                      os.environ.get("LD_LIBRARY_PATH", "")])}
    gguf = res(m["dir"]) / m["file"]
    args = list(m["server_args"])
    ctx = (m.get("ctx_by_catalogue") or {}).get(Path(catalogue).name) if catalogue else None
    if ctx:
        args[args.index("--ctx-size") + 1] = str(ctx)
    cmd = [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(m["port"]), "--alias", name, *args]
    return cmd, env, gguf, m["port"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s1llm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "score", "show"):
        p = sub.add_parser(c)
        p.add_argument("--model", required=True)
        p.add_argument("--cond", required=True, choices=["think", "nothink"], help="reasoning on (think) or off (nothink)")
        p.add_argument("--data", required=True, help="labelled request file")
        p.add_argument("--catalogue", required=True, help="the tool catalogue of the requests")
        p.add_argument("--answers", default="runs/llm_answers", help="answer store root")
        p.add_argument("--out", default="runs/llm_eval", help="records root")
        p.add_argument("--limit", type=int)
        p.add_argument("--ids", nargs="+")
        p.add_argument("--url", help="server URL (default: the model's port on 127.0.0.1)")
        p.add_argument("--seed", type=int, help="another sampling seed; records get a __seed<N> suffix")
        if c == "show":
            p.add_argument("--id", nargs=1, required=True)
    p = sub.add_parser("serve")
    p.add_argument("model")
    p.add_argument("--catalogue", help="the catalogue the model will be asked with (sets its context)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "score":
        cmd_run(a, generate=False)
    elif a.cmd == "show":
        cmd_show(a)
    else:
        cmd, env, gguf, _ = server_command(a.model, a.catalogue)
        print(" ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
