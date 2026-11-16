"""The LLM agent, its model settings and its answer store.

    request ─► one chat completion: the request as the only message (no system prompt: the model's chat template applies
               its own defaults), the catalogue's tools as functions, the server's default tool choice
            ─► the first function call of the answer is the agent's decision; an answer without a call means no tool

The LLMs are tested as distributed: no instructions of ours, no extra functions (to ask or to decline), no retry, one
completion per request. Only the vendor's reasoning switch (a chat-template keyword) and sampling values are set, with
a fixed seed (20261102) and at most 16,384 generated tokens (configs/llm_models.yaml). The tool is not executed.

Every completion is saved in an answer store together with a hash of everything that shapes it (model file, condition,
template switches, sampling, seed, token limit, functions), so an interrupted run continues where it stopped and a
record can be rescored without the model.
"""
import hashlib
import json
from pathlib import Path

import yaml

from llm_agent import LLM_MODELS_YAML
from llm_agent.client import complete


def all_models(path=LLM_MODELS_YAML):
    return yaml.safe_load(Path(path).read_text())


def model_config(name, path=LLM_MODELS_YAML):
    """A model's entry with the file's defaults filled in (its own `server_args` replace the default ones; its
    `server_args_extra` are added to them)."""
    cfg = all_models(path)
    own = cfg["models"][name]
    m = {**cfg.get("defaults", {}), **own, "name": name}
    m["server_args"] = list(own.get("server_args") or cfg.get("defaults", {}).get("server_args", [])) + \
        list(own.get("server_args_extra", []))
    return m


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LLMAgent:
    def __init__(self, model, condition, functions, url=None):
        self.m = model
        self.condition = condition
        self.cond = model["conditions"][condition]
        self.functions = functions
        self.url = url or f"http://127.0.0.1:{model['port']}"
        self.settings = dict(model=model["name"], file=model["file"], sha256=model.get("sha256"),
                             chat_template_file=model.get("chat_template_file"), condition=condition,
                             template=self.cond.get("template", {}), sampling=self.cond.get("sampling", {}),
                             seed=model["seed"], max_tokens=model["max_tokens"], functions=_sha(functions))
        self.key = _sha(self.settings)[:16]

    def body(self, messages):
        return {"model": self.m["name"], "messages": messages, "tools": self.functions, "max_tokens": self.m["max_tokens"],
                "seed": self.m["seed"], "cache_prompt": True, "chat_template_kwargs": self.cond.get("template", {}),
                **self.cond.get("sampling", {})}

    def run(self, request):
        """-> the stored entry: {"settings", "turns": [{"seconds", "error"?, "finish_reason", "message", "usage",
        "timings"}]} (one turn)."""
        out = complete(self.url, self.body([{"role": "user", "content": request}]), timeout=float(self.m.get("timeout", 3600)))
        turn = {"seconds": out["seconds"]}
        if "error" in out:
            turn["error"] = out["error"]
        else:
            choice = out["response"]["choices"][0]
            msg = choice.get("message") or {}
            turn.update(finish_reason=choice.get("finish_reason"),
                        message={k: msg.get(k) for k in ("content", "reasoning_content", "tool_calls")},
                        usage=out["response"].get("usage"), timings=out["response"].get("timings"))
        return {"settings": self.key, "turns": [turn]}


class AnswerStore:
    """One JSON line per answered request: {"id", "settings", "turns", "cost"}. Only entries with the agent's settings
    hash are reused; entries of other settings stay in the file untouched."""

    def __init__(self, path, key):
        self.path, self.key = Path(path), key
        self.entries = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("settings") == key:
                        self.entries[e["id"]] = e

    def get(self, rid):
        return self.entries.get(rid)

    def add(self, rid, entry):
        entry = {"id": rid, **entry}
        self.entries[rid] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry
