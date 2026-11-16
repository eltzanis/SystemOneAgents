"""Run the System 1 agent with each decision model over labelled request files: accuracy and cost in one run.

    python scripts/run_s1.py --catalogue ../data/catalogues/imaging_tools.json \
        --data ../data/requests/imaging_test.jsonl ../data/requests/external.jsonl
    python scripts/run_s1.py --catalogue ../data/catalogues/biomedical_tools.json --data ../data/requests/biomedical_test.jsonl
    python scripts/run_s1.py --models decider-4b --catalogue ... --data ...

For each model (configs/models.yaml): start its llama-server (with --metrics, which only adds counters), wait for
/health, measure 20 s of idle power, then plan every request (tool selection and argument filling; nothing is executed)
under the cost meter (s1eval.meter: seconds, GPU energy and utilisation, tokens counted by the server), and stop the
server. Records: <out>/<data name>/<model>.jsonl (s1eval.records) and <model>.summary.json.

Every answer of the model is recorded in <answers>/<model>.jsonl (s1agent.backends.base.AnswerStore format) without
being replayed: every question goes to the model, so times and energies are the model's own, and the run can later be
rebuilt exactly without the model (scripts/replay_s1.py). After an interruption the same command continues: requests
already recorded are skipped.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

import httpx

from s1agent import build_agent
from s1agent.backends import RecordOnlyStore, make_backend
from s1agent.config import models_config, server_command
from s1eval.meter import Meter
from s1eval.records import s1_record, verdict

MODELS = ["decider-4b", "pplx-decider-flash", "decider-35b-a3b", "openjev", "clef", "pplx-decider-27b-embed"]


def healthy(proc, port, timeout=900):
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
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--data", nargs="+", required=True, help="labelled request files")
    ap.add_argument("--catalogue", required=True, help="the tool catalogue of the requests")
    ap.add_argument("--models-file", help="decision models file (default configs/models.yaml)")
    ap.add_argument("--out", default="runs/s1", help="records folder")
    ap.add_argument("--answers", default="runs/s1_answers", help="answer store folder")
    a = ap.parse_args()
    cfg = models_config(a.models_file) if a.models_file else models_config()
    out, answers = Path(a.out), Path(a.answers)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        print(f"\n=== {name}  {time.strftime('%F %T')}", flush=True)
        cmd, env, gguf, port = server_command(cfg, name, ["--metrics"])
        if not gguf.exists():
            print(f"skipped: {gguf} not found"); continue
        with open(out / f"{name}_server.log", "a") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not healthy(proc, port):
                print(f"skipped: the server did not come up ({out / f'{name}_server.log'})"); continue
            backend = make_backend(name, cfg)
            backend.answers = RecordOnlyStore(answers / f"{name}.jsonl")
            agent = build_agent(name, a.catalogue, backends=(backend, backend))
            agent.select_min_conf = 0.0                       # score the raw selection
            meter = Meter(f"http://127.0.0.1:{port}", gpu_index=int(os.environ.get("S1_GPU", "0")))
            idle = meter.idle()
            print(f"meter: idle {idle}; {'; '.join(meter.problems) or 'all counters available'}", flush=True)
            for data in a.data:
                stem = Path(data).stem
                rows = [json.loads(l) for l in Path(data).read_text().splitlines() if l.strip()]
                path = out / stem / f"{name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
                with path.open("a") as f:
                    for i, r in enumerate(rows, 1):
                        if r["id"] in done:
                            continue
                        rec, _ = s1_record(agent, r, meter.measure())
                        rec["cost"]["idle_w"] = meter.idle_w
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); f.flush()
                        print(f"{'✓' if verdict(rec) else '✗'} {i:4d}/{len(rows)} {r['id']:16s} {rec['seconds']:6.2f}s "
                              f"{rec['cost'].get('gpu_energy_j') or 0:6.0f} J {rec['cost'].get('prompt_tokens') or 0:7.0f} tok",
                              flush=True)
                recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                s = dict(model=name, data=Path(data).name, n=len(recs), n_rows=len(rows), idle=idle,
                         meter_problems=meter.problems, correct=sum(verdict(r) for r in recs),
                         seconds=sum(r["seconds"] for r in recs),
                         gpu_energy_j=sum(r["cost"].get("gpu_energy_j") or 0 for r in recs),
                         model_calls=sum(r["model_calls"] for r in recs))
                path.with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
                print(f"{stem}: {s['correct']}/{s['n']} fully correct, {s['seconds'] / max(1, len(recs)):.2f} s and "
                      f"{s['gpu_energy_j'] / max(1, len(recs)):.0f} J per request", flush=True)
            meter.close()
        finally:
            stop(proc)


if __name__ == "__main__":
    sys.exit(main())
