"""A baseline without a model: each option is scored by its word overlap with the state and question.

The score is a BM25-like sum over the option's (stemmed) words found in the state, weighted by how rare the word is
among the options, normalised by the option's length and turned into probabilities with a softmax. It needs no GPU and
no server, so it is the floor every decision model is compared with, and the backend the tests can run on.

`words` and `_stem` are also used by the compiler, to compare argument names with the words of a request.
"""
import math
import re
from collections import Counter

from s1agent.backends.base import Backend, Decision

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or with by from as at is are be this that it its into via using use "
            "please can you me my i we our file files data user request".split())


def words(text):
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _stem(w):
    for suf in ("ation", "ing", "ies", "es", "s", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


class LexicalBackend(Backend):
    name = "lexical"

    def __init__(self, sharpness=3.0):
        super().__init__()
        self.sharpness = sharpness

    def _choice(self, state, question, options, kind):
        query = Counter(_stem(w) for w in words(state))
        docs = [set(_stem(w) for w in words(o)) for o in options]
        df = Counter(w for d in docs for w in d)
        n = len(options)
        scores = []
        for d in docs:
            s = sum(math.log(1 + n / df[w]) * min(query[w], 2) for w in d if w in query)
            scores.append(s / math.sqrt(1 + len(d)))
        m = max(scores)
        e = [math.exp(self.sharpness * (v - m)) for v in scores]
        z = sum(e)
        return Decision(question, options, [v / z for v in e], kind, meta={"backend": self.name})
