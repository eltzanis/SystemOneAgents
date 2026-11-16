"""What an LLM agent's answer becomes before scoring (llm_agent.score) and the settings key of its answer store."""
import json
from pathlib import Path

import pytest

from llm_agent.score import coerce, interpret, parse_arguments, record
from s1agent.catalog import ToolSpec

ROOT = Path(__file__).resolve().parents[2]
TOOL = ToolSpec("ts", "segment", "Segment.", {"type": "object", "required": ["input_path", "output_dir"], "properties": {
    "input_path": {"type": "string"}, "output_dir": {"type": "string"},
    "fast": {"type": "boolean", "default": False}, "folds": {"type": "array", "items": {"type": "integer"}},
    "threads": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}}})
BY_NAME = {"ts__segment": TOOL}


def entry(name, args, error=None):
    if error:
        return {"turns": [{"seconds": 1.0, "error": error}]}
    calls = [{"function": {"name": name, "arguments": json.dumps(args)}}] if name else None
    return {"turns": [{"seconds": 1.0, "finish_reason": "tool_calls", "message": {"content": "", "tool_calls": calls}}]}


def test_arguments_are_parsed_and_coerced_as_a_server_would():
    assert parse_arguments('"{\\"a\\": 1}"') == ({"a": 1}, None)          # encoded twice
    assert coerce("true", {"type": "boolean"}) is True
    assert coerce("5", {"type": "integer"}) == 5 and coerce(5.0, {"type": "integer"}) == 5
    assert coerce("[0, 1]", {"type": "array"}) == [0, 1]
    assert coerce("8", {"anyOf": [{"type": "integer"}, {"type": "null"}]}) == 8   # an optional integer


def test_record_of_a_right_call():
    row = {"id": "r1", "request": "x", "tool": "ts/segment", "args": {"input_path": "/d/ct.nii.gz", "threads": 8}}
    rec = record(row, entry("ts__segment", {"input_path": "/d/ct.nii.gz", "threads": "8", "fast": "false"}),
                 BY_NAME, {"output_dir"}, "/ws")
    assert rec["pred"] == "ts/segment" and rec["args_pred"]["output_dir"] == "/ws/segment_policy"
    assert rec["args_extra"] == [] and rec["args_missing"] == [] and rec["call_correct"]


def test_no_call_unknown_tool_and_errors():
    row = {"id": "r2", "request": "x", "tool": None}
    assert record(row, entry(None, {}), BY_NAME, set(), "/ws")["pred"] is None
    assert record(row, entry("ts__other", {}), BY_NAME, set(), "/ws")["pred"] == "?ts__other"
    assert interpret(entry(None, {}, error="HTTP 500"), BY_NAME)["kind"] == "error"


@pytest.mark.skipif(not (ROOT / "results" / "llm").exists(), reason="the released results are not next to the code")
def test_settings_keys_of_the_released_completions_are_reproduced():
    """The key hashes the model file, condition, template switches, sampling, seed, token limit and the functions built
    from the catalogue: reproducing it shows the configuration and the function definitions are those of the runs."""
    from llm_agent import tools as T
    from llm_agent.agent import LLMAgent, model_config
    for s, cat in (("external", "imaging_tools.json"), ("biomedical_test", "biomedical_tools.json")):
        fns, _, _ = T.build(ROOT / "data" / "catalogues" / cat)
        for p in sorted((ROOT / "results" / "llm" / s).glob("*.summary.json")):
            summary = json.loads(p.read_text())
            agent = LLMAgent(model_config(summary["model"]), summary["condition"], fns)
            assert agent.key == summary["settings"], p.name
