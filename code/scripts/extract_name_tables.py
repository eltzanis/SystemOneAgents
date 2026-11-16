"""Extract the name tables and auto-detected column lists of the imaging MCP servers that the scorer uses.

    python scripts/extract_name_tables.py /path/to/mcp_servers [--out src/s1eval/data/reco_name_tables.json]

For each (source file, variable) of s1eval.equivalence.ALIASES and AUTODETECT, the literal first assigned to the variable
in the server's source is read with `ast` (never executed); in a dictionary, entries whose key or value is computed are
left out. The scorer reads the result instead of the servers' sources.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from a checkout without installing

from s1eval.equivalence import ALIASES, AUTODETECT, DATA  # noqa: E402


def literal(path, name):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Dict):
                out = {}
                for k, v in zip(node.value.keys, node.value.values):
                    try:
                        out[ast.literal_eval(k)] = ast.literal_eval(v)
                    except ValueError:
                        continue
                return out
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("servers", help="the folder holding the servers' *_tool.py files")
    ap.add_argument("--out", default=str(DATA / "reco_name_tables.json"))
    a = ap.parse_args()
    specs = sorted({(f, n) for f, n, _ in ALIASES.values()} | set(AUTODETECT.values()))
    tables = {f"{f}:{n}": literal(Path(a.servers) / f, n) for f, n in specs}
    doc = {"about": "Name tables of the imaging (ReCo) MCP servers that the scorer uses: for each, the server source file "
                    "and the variable that holds it (the literal assigned to it; dictionary entries whose key or value is "
                    "computed are left out). Extracted with scripts/extract_name_tables.py.",
           "tables": tables}
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(tables)} tables -> {a.out}")


if __name__ == "__main__":
    main()
