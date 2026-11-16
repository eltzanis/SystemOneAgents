"""System 1 agents: MCP tool use driven by one-pass decision models instead of a generating LLM.

    from s1agent import build_agent
    agent = build_agent("decider-4b", "data/catalogues/imaging_tools.json")
    plan = agent.plan("Segment the liver in /data/ct/p01.nii.gz")      # needs the model's llama-server running

Modules: agent (the loop), compiler (schema -> typed questions -> arguments), candidates (values read from the request),
backends (decision models, answer store), catalog (tool catalogue), mcp_hub (MCP client), config, cli (`s1a`).
"""
from pathlib import Path

__version__ = "1.0.0"


def build_agent(model, catalogue, settings=None, models=None, args_model=None, backends=None):
    """The agent with `model` for every decision (args_model: another model for the argument questions only).
    catalogue: a catalogue JSON file. settings: configs/agent.yaml (or a dict). models: configs/models.yaml (or the
    loaded dict). backends: (select backend, args backend) to use instead of building them (e.g. StoreBackend)."""
    from s1agent.agent import S1Agent
    from s1agent.backends import make_backend
    from s1agent.catalog import load_catalogue
    from s1agent.config import agent_settings, models_config
    cfg = settings if isinstance(settings, dict) else agent_settings(settings) if settings else agent_settings()
    if backends is None:
        mcfg = None if model == "lexical" and args_model in (None, "lexical") else \
            (models if isinstance(models, dict) else models_config(models) if models else models_config())
        sel = make_backend(model, mcfg)
        argb = sel if (args_model or model) == model else make_backend(args_model, mcfg)
    else:
        sel, argb = backends
    return S1Agent(load_catalogue(catalogue), sel, argb, select_min_conf=cfg["select_min_conf"],
                   policy_defaults=cfg.get("policy_defaults"), workspace=str(Path(cfg["workspace"]).resolve()),
                   optional_min_conf=float(cfg["optional_min_conf"]), list_min_conf=float(cfg["list_min_conf"]))
