"""Run the LLM agents over labelled request files, one llama-server at a time, without starting servers by hand.

    python scripts/run_llm.py --catalogue ../data/catalogues/imaging_tools.json \
        --data ../data/requests/imaging_test.jsonl ../data/requests/external.jsonl
    python scripts/run_llm.py --catalogue ../data/catalogues/biomedical_tools.json \
        --data ../data/requests/biomedical_test.jsonl --models qwen3.5-4b --conds think

For each condition (outer loop, so that every model's reasoning-on run finishes first), for each model of
configs/llm_models.yaml: start its llama-server (its flags, the context of the catalogue), wait until /health answers,
ask every request of each data file once (`s1llm run`: one completion per request, measured), score the answers, stop
the server. Every completion is saved (<answers>/<data name>/<model>__<cond>.jsonl), so after an interruption the same
command continues where it stopped. Records and summaries: <out>/<data name>/<model>__<cond>.jsonl, .summary.json.
A model whose GGUF is missing or whose server does not come up is skipped and reported. Logs: <logs>/.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

import httpx  # noqa: E402

from llm_agent.agent import all_models  # noqa: E402
from llm_agent.cli import cmd_run, server_command  # noqa: E402


def wait_healthy(proc, port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(3)
    return False


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    time.sleep(5)                                       # let the driver release the GPU memory


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(all_models()["models"]))
    ap.add_argument("--conds", nargs="+", default=["think", "nothink"], choices=["think", "nothink"])
    ap.add_argument("--data", nargs="+", required=True, help="labelled request files")
    ap.add_argument("--catalogue", required=True, help="the tool catalogue of the requests")
    ap.add_argument("--answers", default="runs/llm_answers")
    ap.add_argument("--out", default="runs/llm_eval")
    ap.add_argument("--logs", default="runs/llm_logs")
    ap.add_argument("--seed", type=int, help="another sampling seed")
    ap.add_argument("--startup-timeout", type=int, default=900)
    a = ap.parse_args()
    logs = Path(a.logs)
    logs.mkdir(parents=True, exist_ok=True)
    notes = []
    for cond in a.conds:
        for name in a.models:
            print(f"\n=== {name} [{cond}]  {time.strftime('%F %T')}", flush=True)
            cmd, env, gguf, port = server_command(name, a.catalogue)
            if not gguf.exists():
                notes.append(f"{name} [{cond}]: skipped, {gguf} not found"); print(notes[-1]); continue
            with open(logs / f"{name}_server_{cond}.log", "w") as log:
                log.write(" ".join(cmd) + "\n\n"); log.flush()
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if not wait_healthy(proc, port, a.startup_timeout):
                    notes.append(f"{name} [{cond}]: skipped, the server did not come up ({name}_server_{cond}.log)")
                    print(notes[-1]); continue
                for data in a.data:
                    cmd_run(Namespace(model=name, cond=cond, data=data, catalogue=a.catalogue, answers=a.answers,
                                      out=a.out, limit=None, ids=None, url=None, seed=a.seed))
            finally:
                stop(proc)
    for n in notes:
        print("note:", n)
    (logs / "notes.json").write_text(json.dumps(notes, indent=1))


if __name__ == "__main__":
    main()
