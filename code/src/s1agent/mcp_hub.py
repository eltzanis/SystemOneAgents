"""Connect to stdio MCP servers, list their tools and call them.

Servers are described in a YAML file (configs/servers.example.yaml):

    servers:
      <id>:
        command: /path/to/python          # the server's executable
        args: [-m, my_server]             # its arguments
        cwd: /path/to/server              # optional working folder
        env: {KEY: value}                 # optional extra environment
        description: one line             # optional; kept in the catalogue

Tool selection only needs the tool list, so `build_catalogue` starts every server once and saves its tools in a JSON
catalogue (s1agent.catalog); the agent then starts a server only when it executes one of its tools.
"""
import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from s1agent.catalog import ToolSpec


def load_servers(path):
    """-> {id: server entry} from a servers file."""
    return yaml.safe_load(Path(path).read_text())["servers"]


class MCPHub:
    """One ClientSession per started server; servers start lazily on first use. Use as `async with MCPHub(servers):`."""

    def __init__(self, servers, startup_timeout=180.0):
        self.servers = servers
        self.startup_timeout = startup_timeout
        self._stack = AsyncExitStack()
        self._sessions = {}

    async def __aenter__(self):
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._stack.__aexit__(*exc)

    def _params(self, spec):
        env = {**os.environ, **{k: os.path.expanduser(str(v)) for k, v in (spec.get("env") or {}).items()},
               "PYTHONUNBUFFERED": "1"}
        cwd = os.path.expanduser(spec["cwd"]) if spec.get("cwd") else None
        return StdioServerParameters(command=os.path.expanduser(spec["command"]), args=list(spec.get("args") or []),
                                     env=env, cwd=cwd)

    async def session(self, server_id):
        if server_id not in self._sessions:
            read, write = await self._stack.enter_async_context(stdio_client(self._params(self.servers[server_id])))
            s = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(s.initialize(), self.startup_timeout)
            self._sessions[server_id] = s
        return self._sessions[server_id]

    async def list_tools(self, server_id):
        s = await self.session(server_id)
        res = await s.list_tools()
        return [ToolSpec(server_id, t.name, t.description or "", _field(t, "input_schema", "inputSchema") or {})
                for t in res.tools]

    async def call(self, server_id, tool, arguments, timeout=None):
        s = await self.session(server_id)
        return await s.call_tool(tool, arguments, read_timeout_seconds=timeout)


def _field(obj, *names):
    """First present attribute: MCP SDK 2.x uses snake_case (input_schema), 1.x camelCase (inputSchema)."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _root_cause(e):
    """The first leaf exception of an ExceptionGroup (anyio wraps every stdio failure in one)."""
    while getattr(e, "exceptions", None):
        e = e.exceptions[0]
    return e


def result_text(result):
    """The text blocks of a CallToolResult joined, plus structured content when there is no text."""
    parts = [getattr(c, "text", "") for c in getattr(result, "content", []) or []]
    text = "\n".join(p for p in parts if p)
    structured = _field(result, "structured_content", "structuredContent")
    if not text and structured:
        text = json.dumps(structured, indent=2)
    return text


async def _build_catalogue(servers):
    out = {"servers": {}, "tools": [], "errors": {}}
    for sid, spec in servers.items():
        # A separate hub per server: one server failing to start must not take the others down.
        try:
            async with MCPHub({sid: spec}) as hub:
                tools = await hub.list_tools(sid)
            out["servers"][sid] = {"id": sid, "description": spec.get("description", "")}
            out["tools"] += [asdict(t) for t in tools]
        except BaseException as e:            # noqa: BLE001 - report every failure, keep going
            if isinstance(e, KeyboardInterrupt):
                raise
            e = _root_cause(e)
            out["errors"][sid] = f"{type(e).__name__}: {e}"
    return out


def build_catalogue(servers, path):
    """Start each server once, list its tools and write the catalogue to `path`."""
    cat = asyncio.run(_build_catalogue(servers))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cat, indent=2))
    return cat
