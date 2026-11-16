"""Replay of the released runs: the compiler must ask exactly the questions the recorded answers were given to.

Every question (state, question text, options) is hashed into the answer-store key, so a change of one character in a
question or option makes the replay fail with MissingAnswer. Runs on a sample of each released set and model (the full
replay: scripts/replay_s1.py)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS, DATA = ROOT / "results" / "system1", ROOT / "data"
pytestmark = pytest.mark.skipif(not (RESULTS / "answers").exists() or not (DATA / "requests").exists(),
                                reason="the released results and evaluation set are not next to the code")
SETS = {"external": "imaging_tools.json", "imaging_test": "imaging_tools.json", "biomedical_test": "biomedical_tools.json"}
MODELS = ["decider-4b", "pplx-decider-flash", "decider-35b-a3b", "openjev", "clef", "pplx-decider-27b-embed"]
SAMPLE = 15


@pytest.fixture(scope="module")
def replay():
    sys.path.insert(0, str(ROOT / "code" / "scripts"))
    import replay_s1
    return replay_s1


@pytest.mark.parametrize("model", MODELS)
def test_released_requests_replay_identically(model, replay):
    from s1agent import build_agent
    from s1agent.backends import store_backend
    from s1eval.records import NoCost, s1_record
    backend = store_backend(model, RESULTS / "answers" / f"{model}.jsonl")
    for s, cat in SETS.items():
        agent = build_agent(model, DATA / "catalogues" / cat, backends=(backend, backend))
        agent.select_min_conf = 0.0
        rows = [json.loads(l) for l in (DATA / "requests" / f"{s}.jsonl").read_text().splitlines() if l.strip()]
        run = {r["id"]: r for r in map(json.loads, (RESULTS / "records" / s / f"{model}.jsonl").read_text().splitlines())}
        for r in rows[:: max(1, len(rows) // SAMPLE)]:
            rec, _ = s1_record(agent, r, NoCost())
            assert replay.differences(rec, run[r["id"]]) == [], (model, s, r["id"])
