"""Decider decision models (Mapika/decider-4b, Mapika/decider-35b-a3b) served by llama.cpp's llama-server.

The prompt is decider's plain, state-first layout, built as token ids with the checkpoint's own tokenizer, so that
llama.cpp sees exactly the sequence the model was trained on:

    Context:\\n<state>\\n\\nQuestion: <question>\\nOptions:\\n(A) <option>\\n(B) <option>\\nAnswer: (

The answer is read at the final " (" position: the log-probabilities of the option-label tokens, divided by the model's
fitted temperature and softmaxed over the valid labels only. llama-server returns them through `n_probs` (before
sampling, over the full vocabulary). A softmax over a subset of tokens does not depend on the full-vocabulary normaliser,
so this equals decider's letter-logit readout once every label is in the returned list. The list starts at 1,024 entries
and grows 16-fold (the prompt is cached, so a retry is cheap) until it holds every label or the whole vocabulary; a label
still absent then lies below every returned token, gets the lowest returned log-probability (an upper bound) and is
counted in Decision.missing_labels.

Up to 10 options use the letters A-J rendered as one string; more use decider's "wide" rendering, one label token per
option (A..Z, then two-letter labels that are single tokens), up to 255.
"""
import json
import math
import string
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from s1agent.backends.base import Backend, Decision

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)
MAX_OPTIONS = 255


class DeciderLlamaCpp(Backend):
    def __init__(self, base_url, model_dir, name=None, temperature_by_type=None, max_state_tokens=8192, timeout=600.0,
                 concurrency=1):
        super().__init__()
        self.concurrency = concurrency
        model_dir = Path(model_dir)
        self.name = name or model_dir.name
        self.base_url = base_url.rstrip("/")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = json.loads((model_dir / "decider_config.json").read_text())
        T = float(cfg.get("temperature", 1.0))
        self.temperature = {"choice": T, "noul": T, **(cfg.get("temperature_by_type") or {}), **(temperature_by_type or {})}
        self.max_state_tokens = max_state_tokens
        self.label_names, self.label_ids = self._label_table()
        self.open_ids = self._enc("\n(")
        self.http = httpx.Client(timeout=timeout)

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def _label_table(self):
        """decider's label table: A..Z, then the first two-letter upper-case strings that are single tokens."""
        U = string.ascii_uppercase
        names, ids = [], []
        for n in list(U) + [a + b for a in U for b in U]:
            t = self._enc(n)
            if len(t) == 1:
                names.append(n); ids.append(t[0])
            if len(names) == MAX_OPTIONS:
                break
        for j, L in enumerate(LETTERS):
            assert self._enc(L) == [ids[j]], f"label {L} is not one token for this tokenizer"
        return names, ids

    def build(self, state, question, options):
        """Token ids of the single-question prompt (plain layout, options in the order given)."""
        ids = self._enc("Context:\n" + state)[: self.max_state_tokens]
        head, tail = f"\n\nQuestion: {question}\nOptions:", "\nAnswer: ("
        if len(options) <= NARROW:
            ids += self._enc(head + "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)) + tail)
        else:
            ids += self._enc(head)
            for j, o in enumerate(options):
                ids += self.open_ids + [self.label_ids[j]] + self._enc(f") {o}")
            ids += self._enc(tail)
        return ids

    def _label_logprobs(self, ids, n):
        want = self.label_ids[:n]
        k = 1024
        while True:
            r = self.http.post(f"{self.base_url}/completion", json={
                "prompt": ids, "n_predict": 1, "n_probs": k, "temperature": 0.0,
                "cache_prompt": True, "post_sampling_probs": False})
            r.raise_for_status()
            body = r.json()
            top = body["completion_probabilities"][0]
            rows = top.get("top_logprobs")
            if rows is not None:
                got = {e["id"]: e["logprob"] for e in rows}
            else:                                                    # older servers: probabilities instead of log-probabilities
                got = {e["id"]: math.log(max(e["prob"], 1e-30)) for e in top["top_probs"]}
            missing = sum(i not in got for i in want)
            if not missing or len(got) < k:                          # every label found, or the whole vocabulary returned
                break
            k *= 16
        floor = min(got.values())
        return [got.get(i, floor) for i in want], missing, body.get("timings", {}), k

    def _choice(self, state, question, options, kind):
        ids = self.build(state, question, options)
        lps, missing, timings, k = self._label_logprobs(ids, len(options))
        T = self.temperature.get(kind, self.temperature["choice"])
        z = [lp / T for lp in lps]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        meta = {"backend": self.name, "n_probs": k}
        if timings:
            meta["prompt_eval_tokens"] = timings.get("prompt_n")
            meta["prompt_ms"] = round(timings.get("prompt_ms", 0.0), 1)
        return Decision(question, options, [v / s for v in e], kind, prompt_tokens=len(ids), missing_labels=missing, meta=meta)

    def health(self):
        r = self.http.get(f"{self.base_url}/health")
        return r.status_code == 200
