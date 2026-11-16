"""The prompt sent to llama-server must be decider's training layout, token for token (needs the tokenizer files of
decider-35b-a3b: the folder of configs/models.yaml, or $S1_DECIDER_DIR)."""
import os
from pathlib import Path

import pytest

from s1agent.backends.decider_llamacpp import DeciderLlamaCpp
from s1agent.config import models_config, resolve

cfg = models_config()
MODEL = Path(os.environ.get("S1_DECIDER_DIR") or resolve(cfg, cfg["models"]["decider-35b-a3b"]["model_dir"]))
pytestmark = pytest.mark.skipif(not (MODEL / "tokenizer.json").exists(), reason="tokenizer not downloaded")


@pytest.fixture(scope="module")
def b():
    return DeciderLlamaCpp("http://127.0.0.1:1", MODEL)


def test_narrow_layout(b):
    ids = b.build("My card was charged twice.", "Which department?", ["billing", "technical support", "sales"])
    assert b.tok.decode(ids) == ("Context:\nMy card was charged twice.\n\nQuestion: Which department?\nOptions:\n(A) billing"
                                 "\n(B) technical support\n(C) sales\nAnswer: (")


def test_wide_layout_uses_one_label_token_per_option(b):
    opts = [f"option {i}" for i in range(30)]
    ids = b.build("s", "q", opts)
    text = b.tok.decode(ids)
    assert "\n(Z) option 25" in text and "\n(AA) option 26" in text and text.endswith("\nAnswer: (")
    for j in range(30):
        assert b.label_ids[j] in ids


def test_temperature_from_config(b):
    assert b.temperature["choice"] == pytest.approx(1.08)
