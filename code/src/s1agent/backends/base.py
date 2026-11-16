"""What every decision model offers the agent: one typed question over a state, answered with a probability per option.

A backend never generates text. `choice` takes a state (the request and context), a question and a finite list of options
and returns a `Decision`: one probability per option, in the order given. A yes/no question is the same call with the
two options "no" and "yes" and kind "noul" (some models read yes/no questions with their own prompt form).

The answer store records every answer a backend gives, keyed by everything the model sees (`AnswerStore.key`), so that a
run can be replayed later, question by question, without the model.
"""
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Decision:
    question: str
    options: list
    probs: list                      # one probability per option, same order
    kind: str = "choice"             # "choice" | "noul" (yes/no)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    missing_labels: int = 0          # option labels absent from the server's returned list (their probability is a bound)
    meta: dict = field(default_factory=dict)

    @property
    def index(self):
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def choice(self):
        return self.options[self.index]

    @property
    def confidence(self):
        return self.probs[self.index]

    def as_dict(self):
        return dict(question=self.question, kind=self.kind, options=self.options, probs=[round(p, 4) for p in self.probs],
                    choice=self.choice, confidence=round(self.confidence, 4), latency_s=round(self.latency_s, 4),
                    prompt_tokens=self.prompt_tokens, missing_labels=self.missing_labels, **self.meta)


class AnswerStore:
    """The answers of one model, by question, one JSON line per answer ({"key", "probs", "prompt_tokens",
    "missing_labels"}), appended as they are given.

    A decision model reads a question in one forward pass, so the same (model, kind, state, question, options) gets the
    same probabilities. The key is the SHA-256 of exactly these five items; the model name is part of it, so answers of
    different models never mix. A run that records its answers can therefore be rebuilt without the model, and any
    change to a question's wording, an option's text or the state shows up as a missing key."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.saved = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.saved[r["key"]] = r
        self.hits = self.misses = 0

    @staticmethod
    def key(model, state, question, options, kind):
        return hashlib.sha256(json.dumps([model, kind, state, question, list(options)]).encode()).hexdigest()

    def get(self, key):
        with self.lock:
            r = self.saved.get(key)
            self.hits += r is not None
            self.misses += r is None
            return r

    def put(self, key, d):
        r = dict(key=key, probs=d.probs, prompt_tokens=d.prompt_tokens, missing_labels=d.missing_labels)
        with self.lock:
            self.saved[key] = r
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(r) + "\n")


class RecordOnlyStore(AnswerStore):
    """Saves every answer and never replays one: every question goes to the model, so times and energies are the
    model's own. This is how the evaluation runs record their answers for a later replay."""

    def get(self, key):
        with self.lock:
            self.misses += 1
        return None


class Backend:
    """Subclasses implement `_choice`; `choice`, `noul` and `choice_batch` add timing, the answer store and the trace."""
    name = "backend"
    max_options = 255
    option_limit = None              # most options the model takes in one question (None: no limit); see _choice_in_rounds
    concurrency = 1                  # > 1: choice_batch sends questions at once (the server must have as many slots)
    answers = None                   # AnswerStore, or None

    def __init__(self):
        self.trace = []              # every Decision, in order

    def _choice(self, state, question, options, kind):
        raise NotImplementedError

    def _ask(self, state, question, options, kind):
        """_choice, through the answer store when one is attached (a replayed answer has no latency)."""
        if self.answers is None:
            return self._choice(state, question, options, kind)
        key = AnswerStore.key(self.name, state, question, options, kind)
        r = self.answers.get(key)
        if r is not None:
            return Decision(question, list(options), list(r["probs"]), kind, prompt_tokens=r["prompt_tokens"],
                            missing_labels=r["missing_labels"], meta={"replayed": True})
        d = self._choice(state, question, options, kind)
        self.answers.put(key, d)
        return d

    def _choice_in_rounds(self, state, question, options, kind):
        """A question with more options than the model takes: the options are cut into consecutive groups of at most
        `option_limit`, as even as possible (34 -> 17 + 17); the model picks within each group, then among the group
        winners. P(option) = P(its group's winner in the final round) x P(option within its group), which sums to 1."""
        n = len(options)
        n_groups = math.ceil(n / self.option_limit)
        bounds = [round(i * n / n_groups) for i in range(n_groups + 1)]
        groups = [list(range(bounds[i], bounds[i + 1])) for i in range(n_groups)]
        within, winners, ntok, missing = [], [], 0, 0
        for g in groups:
            d = self._ask(state, question, [options[i] for i in g], kind)
            within.append(d.probs); winners.append(g[d.index]); ntok += d.prompt_tokens; missing += d.missing_labels
        final = self._ask(state, question, [options[i] for i in winners], kind)
        probs = [0.0] * n
        for gi, g in enumerate(groups):
            for j, i in enumerate(g):
                probs[i] = final.probs[gi] * within[gi][j]
        return Decision(question, options, probs, kind, prompt_tokens=ntok + final.prompt_tokens,
                        missing_labels=missing + final.missing_labels,
                        meta={**final.meta, "rounds": 2, "groups": len(groups)})

    def _timed(self, spec):
        options = list(spec["options"])
        if not 2 <= len(options) <= self.max_options:
            raise ValueError(f"{self.name}: a question needs 2..{self.max_options} options, got {len(options)}")
        t0 = time.perf_counter()
        kind = spec.get("kind", "choice")
        if self.option_limit and len(options) > self.option_limit:
            d = self._choice_in_rounds(spec["state"], spec["question"], options, kind)
        else:
            d = self._ask(spec["state"], spec["question"], options, kind)
        d.latency_s = time.perf_counter() - t0
        if spec.get("tag"):
            d.meta["tag"] = spec["tag"]
        return d

    def choice(self, state, question, options, kind="choice", tag=None):
        d = self._timed(dict(state=state, question=question, options=options, kind=kind, tag=tag))
        self.trace.append(d)
        return d

    def choice_batch(self, specs):
        """Independent questions, each a dict(state, question, options, kind="choice", tag=None); answers in order.
        With concurrency > 1 they are in flight together, so a server with several slots evaluates them in one batch
        (faster, but the probabilities may then depend on what shares the batch; the study used concurrency 1)."""
        if self.concurrency > 1 and len(specs) > 1:
            with ThreadPoolExecutor(min(self.concurrency, len(specs))) as ex:
                out = list(ex.map(self._timed, specs))
        else:
            out = [self._timed(s) for s in specs]
        self.trace.extend(out)
        return out

    def noul(self, state, question, tag=None):
        """P(yes) of a yes/no question, asked as the two options ("no", "yes")."""
        return self.choice(state, question, ["no", "yes"], kind="noul", tag=tag).probs[1]


class MissingAnswer(LookupError):
    """A question the answer store has no answer for (raised by StoreBackend)."""


class StoreBackend(Backend):
    """A backend that only replays an answer store and fails on any question it has no answer for. It stands in for a
    model to rebuild a recorded run exactly: same name (part of the key) and same option limit as the model."""

    def __init__(self, name, store, option_limit=None):
        super().__init__()
        self.name = name
        self.answers = store
        self.option_limit = option_limit

    def _choice(self, state, question, options, kind):
        raise MissingAnswer(f"{self.name}: no stored answer for a {kind} question with {len(options)} options: "
                            f"{question[:100]!r}")
