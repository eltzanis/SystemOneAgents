"""Backends by name: `lexical` (no model), or a model of configs/models.yaml served by llama-server on 127.0.0.1.

    kind                 class                                   how the model is read
    decider-llamacpp     decider_llamacpp.DeciderLlamaCpp        option-label log-probabilities (/completion)
    pplx-llamacpp        pplx_llamacpp.PplxLlamaCpp              hidden state at the readout marker (/embeddings)
    systemone-llamacpp   systemone_llamacpp.SystemOneLlamaCpp    the server's own decision endpoint (/v1/systemone)

The backend's name is the model's name in the file; it is part of every answer-store key.
"""
from s1agent.backends.base import AnswerStore, Backend, Decision, MissingAnswer, RecordOnlyStore, StoreBackend
from s1agent.backends.lexical import LexicalBackend
from s1agent.config import models_config, resolve


def make_backend(name, cfg=None):
    """The backend of model `name` (cfg: models_config(); default configs/models.yaml)."""
    if name == "lexical":
        return LexicalBackend()
    cfg = cfg or models_config()
    m = cfg["models"][name]
    url = f"http://127.0.0.1:{m['port']}"
    if m["kind"] == "decider-llamacpp":
        from s1agent.backends.decider_llamacpp import DeciderLlamaCpp
        b = DeciderLlamaCpp(url, resolve(cfg, m["model_dir"]), name=name, temperature_by_type=m.get("temperature_by_type"),
                            concurrency=m.get("concurrency", 1))
    elif m["kind"] == "pplx-llamacpp":
        from s1agent.backends.pplx_llamacpp import PplxLlamaCpp
        b = PplxLlamaCpp(url, resolve(cfg, m["hf_dir"]), name=name)
    elif m["kind"] == "systemone-llamacpp":
        from s1agent.backends.systemone_llamacpp import SystemOneLlamaCpp
        b = SystemOneLlamaCpp(url, name=name)
    else:
        raise ValueError(f"unknown backend kind {m['kind']!r} for {name}")
    if m.get("option_limit"):
        b.option_limit = int(m["option_limit"])
    return b


def store_backend(name, store_path, cfg=None):
    """A backend that replays model `name`'s answer store and fails on a missing answer (no model is run)."""
    m = {} if name == "lexical" else (cfg or models_config())["models"][name]
    return StoreBackend(name, AnswerStore(store_path), option_limit=int(m["option_limit"]) if m.get("option_limit") else None)


__all__ = ["AnswerStore", "Backend", "Decision", "LexicalBackend", "MissingAnswer", "RecordOnlyStore", "StoreBackend",
           "make_backend", "store_backend"]
