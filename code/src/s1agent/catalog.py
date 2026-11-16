"""The tool catalogue: the tools the MCP servers list, saved as JSON so that selection never has to start a server.

A catalogue file is {"servers": {id: {...}}, "tools": [{"server", "name", "description", "input_schema"}], "errors"}:
each tool exactly as its server returns it from tools/list (`s1a catalog` writes one; data/catalogues/ holds the two of
the study). The agent reads only the tools.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    @property
    def qualified(self):
        return f"{self.server}/{self.name}"


def load_catalogue(path):
    """-> list of ToolSpec, in the catalogue's order (the order of the selection question's options)."""
    cat = json.loads(Path(path).read_text())
    return [ToolSpec(**t) for t in cat["tools"]]
