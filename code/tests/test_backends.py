"""Backends: questions in rounds, the answer store and its key, the store-only backend."""
import pytest

from s1agent.backends.base import AnswerStore, Backend, Decision, MissingAnswer, RecordOnlyStore, StoreBackend


class Scores(Backend):
    """Answers with fixed scores per option text, normalised over the options it is shown; records each call's size."""
    name = "scores"

    def __init__(self, scores, option_limit=None):
        super().__init__()
        self.scores, self.option_limit, self.sizes = scores, option_limit, []

    def _choice(self, state, question, options, kind):
        self.sizes.append(len(options))
        w = [self.scores[o] for o in options]
        return Decision(question, options, [x / sum(w) for x in w], kind)


def test_small_questions_are_asked_once():
    b = Scores({f"o{i}": i + 1 for i in range(5)}, option_limit=5)
    d = b.choice("s", "q", [f"o{i}" for i in range(5)])
    assert b.sizes == [5] and d.choice == "o4"


def test_large_questions_run_in_even_groups_then_a_final():
    opts = [f"o{i}" for i in range(34)]
    b = Scores({o: (100 if o == "o20" else 1) for o in opts}, option_limit=24)
    d = b.choice("s", "q", opts)
    assert b.sizes == [17, 17, 2]                      # 34 -> 17 + 17, then the two group winners
    assert d.choice == "o20" and d.meta["rounds"] == 2
    assert abs(sum(d.probs) - 1) < 1e-9


def test_answer_store_key_is_stable():
    """The key is the SHA-256 of [model, kind, state, question, options] as JSON (the format of the released stores)."""
    assert AnswerStore.key("decider-4b", "User request: x", "Which?", ["a", "b"], "choice") == \
        "75d451ac50957c5cce231963b86311cfa9355fc32838329caf945b8ac68929fe"


def test_answers_are_recorded_and_replayed(tmp_path):
    path = tmp_path / "m.jsonl"
    b = Scores({"a": 1, "b": 3})
    b.name = "m"
    b.answers = RecordOnlyStore(path)
    d1 = b.choice("s", "q", ["a", "b"])
    b.choice("s", "q", ["a", "b"])                     # record-only: asked again
    assert b.sizes == [2, 2] and len(path.read_text().splitlines()) == 2
    r = StoreBackend("m", AnswerStore(path))
    d2 = r.choice("s", "q", ["a", "b"])
    assert d2.probs == d1.probs and d2.meta["replayed"]
    with pytest.raises(MissingAnswer):
        r.choice("s", "q", ["a", "b", "c"])           # one option more: another key
    with pytest.raises(MissingAnswer):
        StoreBackend("other-model", AnswerStore(path)).choice("s", "q", ["a", "b"])   # the model name is in the key
