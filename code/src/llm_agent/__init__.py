"""The LLM agent baseline: a generating LLM as the agent's reasoning engine, used as its distributor ships it.

It sees the catalogue's MCP tools as native function definitions (descriptions and input schemas unchanged), answers
each request with one chat completion, and its first function call is scored by the same scorer as the System 1 agent
(s1eval.scoring). Modules: agent (the completion and its answer store), tools (catalogue -> functions), score (what an
answer becomes before scoring), client (the HTTP call), cli (`s1llm`).
"""
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
LLM_MODELS_YAML = CODE / "configs" / "llm_models.yaml"
