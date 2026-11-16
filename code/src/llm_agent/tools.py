"""The MCP tool catalogue as function definitions for a tool-calling LLM, exactly as the servers describe the tools.

Each tool of the catalogue (the same file the System 1 agent uses) becomes one function: its description is the tool's
MCP description and its parameters are the tool's MCP inputSchema, both unchanged. Only the name is qualified, as MCP
clients do: `<server>__<tool>` with '-' written '_' (training_job_status exists on three servers, and some chat
templates accept only [A-Za-z0-9_] in names). Nothing else is offered: no extra function to ask the user or to decline.
"""
import copy

from s1agent.catalog import load_catalogue
from s1agent.config import agent_settings


def fn_name(tool):
    return f"{tool.server}__{tool.name}".replace("-", "_")


def tool_function(tool):
    params = copy.deepcopy(tool.input_schema or {"type": "object", "properties": {}})
    return {"type": "function", "function": {"name": fn_name(tool), "description": tool.description, "parameters": params}}


def build(catalogue, settings=None):
    """-> (function definitions, {function name: ToolSpec}, the agent's policy parameter names (configs/agent.yaml))."""
    tools = load_catalogue(catalogue)
    fns = [tool_function(t) for t in tools]
    by_name = {f["function"]["name"]: t for f, t in zip(fns, tools)}
    if len(by_name) != len(tools):
        raise ValueError("two tools map to the same function name")
    cfg = settings if isinstance(settings, dict) else agent_settings(settings) if settings else agent_settings()
    return fns, by_name, set(cfg.get("policy_defaults") or {})
