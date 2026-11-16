"""The two configuration files: the agent's settings (configs/agent.yaml) and the decision models (configs/models.yaml).

Relative paths in models.yaml start from its `root` key, itself relative to the file (default: the file's folder).
"""
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parents[2]
AGENT_YAML = CODE / "configs" / "agent.yaml"
MODELS_YAML = CODE / "configs" / "models.yaml"


def agent_settings(path=AGENT_YAML):
    """{select_min_conf, optional_min_conf, list_min_conf, policy_defaults, workspace}."""
    return yaml.safe_load(Path(path).read_text())


def models_config(path=MODELS_YAML):
    """The models file with `root` resolved to an absolute folder."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["root"] = (path.parent / cfg.get("root", ".")).resolve()
    return cfg


def resolve(cfg, p):
    """A path of the models file, absolute."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else cfg["root"] / p


def server_command(cfg, name, extra=()):
    """The llama-server command line and environment of one model -> (argv, env, gguf path, port)."""
    import os
    m = cfg["models"][name]
    exe = resolve(cfg, cfg["llama_server"])
    libs = [str(exe.parent)] + ([str(resolve(cfg, cfg["cuda_libs"]))] if cfg.get("cuda_libs") else [])
    env = {**os.environ, "LD_LIBRARY_PATH": ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]),
           **{k: str(v) for k, v in (m.get("env") or {}).items()}}
    gguf = resolve(cfg, m["model_dir"]) / m["gguf"]
    cmd = [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(m["port"]), *m.get("server_args", []), *extra]
    return cmd, env, gguf, m["port"]
