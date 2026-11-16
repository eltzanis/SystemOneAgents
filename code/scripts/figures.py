"""The figures of the manuscript, from the released data and results only.

    python scripts/figures.py [--out runs/figures]

Reads ../data (requests, catalogues), ../results/analysis/metrics.json (scripts/analyse.py), ../results/fabrication
(scripts/fabrication.py) and ../results/figure1_examples (the recorded decisions of the development request of
Figure 1b), and rescores every agent's records with scripts/analyse.py for the per-request
quantities (bootstrap intervals of argument filling, declines by out-of-scope kind). Writes PDF (vector, fonts
embedded), SVG and 600-dpi PNG per figure, and figure_data.json with every plotted number.
Style: Nature Methods figure specifications (width <= 180 mm, sans-serif 5-7 pt, panel labels 8-pt bold lowercase, no
coloured text, no background gridlines). Needs matplotlib >= 3.10.
"""
import argparse
import json
import math
import sys
import types
from pathlib import Path

import logging

import matplotlib as mpl
mpl.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial/Helvetica absent: quiet fallback
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE / "scripts"))
DATA = CODE.parent / "data"
RES = CODE.parent / "results"
EXAMPLES = RES / "figure1_examples"
MM = 1 / 25.4

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAM = {"s1": "#2a78d6", "think": "#eb6834", "nothink": "#1baf7a"}
MARK = {"s1": "o", "think": "^", "nothink": "s"}
FAMNAME = {"s1": "System 1 agent", "think": "LLM agent, reasoning on", "nothink": "LLM agent, reasoning off"}
S1 = ["decider-4b", "pplx-decider-flash", "decider-35b-a3b", "openjev", "clef", "pplx-decider-27b-embed"]
S1_NAME = {"decider-4b": "decider-4b", "pplx-decider-flash": "pplx-decider-flash", "decider-35b-a3b": "decider-35b-a3b",
           "openjev": "OpenJev", "clef": "Clef", "pplx-decider-27b-embed": "pplx-decider-27b"}
LLM = ["gemma-4-e4b", "nemotron-bolt-e4b", "qwen3.5-4b", "gemma-4-26b-a4b", "qwen3.5-27b", "qwen3.8-27b",
       "muse-glimmer-30b", "qwen3.5-35b-a3b", "nemotron-bolt-74b"]
LLM_NAME = {"gemma-4-e4b": "Gemma 4 E4B", "nemotron-bolt-e4b": "Nemotron 3.5 Bolt E4B", "qwen3.5-4b": "Qwen3.5-4B",
            "gemma-4-26b-a4b": "Gemma 4 26B-A4B", "qwen3.5-27b": "Qwen3.5-27B", "qwen3.8-27b": "Qwen3.8-27B",
            "muse-glimmer-30b": "Muse Glimmer 30B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
            "nemotron-bolt-74b": "Nemotron 3.5 Bolt 74B-A9B"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 0, "xtick.minor.size": 1.2, "xtick.minor.width": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS, "xtick.color": INK2,
    "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 4, "legend.frameon": False, "savefig.dpi": 600,
})


# ------------------------------------------------------------------------------------------------------------ data
def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def systems():
    """[(key, family, model, display name)] of the 24 agents compared (System 1 first, then the LLM settings)."""
    out = [(f"s1:{m}", "s1", m, S1_NAME[m]) for m in S1]
    for c in ("think", "nothink"):
        out += [(f"{c}:{m}", c, m, LLM_NAME[m]) for m in LLM]
    return out


def label(fam, name):
    return name if fam == "s1" else f"{name}, reasoning {'on' if fam == 'think' else 'off'}"


OUT = CODE / "runs" / "figures"


def rows():
    return {(f.stem, r["id"]): r for f in sorted((DATA / "requests").glob("*.jsonl")) for r in load(f)}


def agent_records(R):
    """{agent key: {(set, id): record}}, every agent rescored from the released results (scripts/analyse.py)."""
    import analyse as A
    ns = types.SimpleNamespace(data=str(DATA), results=str(RES))
    return {k: v["recs"] for k, v in A.systems(ns, R).items()}


def value_ok(rec, arg):
    if not rec.get("exact"):
        return False
    eq = rec.get("args_equivalent")
    return bool(eq and eq.get(arg)) or arg in (rec.get("args_conditional") or [])


def per_request(R, recs):
    """Per complete request: (values right, values); per out-of-scope request: (kind, declined)."""
    vals, oos = [], []
    for u, r in R.items():
        rec = recs[u]
        if r.get("tool") and not r.get("ask"):
            vals.append((sum(value_ok(rec, a) for a in r.get("args", {})), len(r.get("args", {}))))
        elif not r.get("tool"):
            kind = next(t for t in r["tags"] if t.startswith("oos-"))
            oos.append((kind, rec.get("pred") is None))
    return np.array(vals), oos


def cluster_ci(v, B=4000, seed=20261110):
    """95% percentile interval of pooled value accuracy, resampling requests (values of a request move together)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    r = v[idx, 0].sum(1) / v[idx, 1].sum(1)
    return 100 * np.percentile(r, 2.5), 100 * np.percentile(r, 97.5)


# --------------------------------------------------------------------------------------------------------- helpers
def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def panel(ax_or_fig, letter, x, y, fig_coords=False):
    kw = dict(fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
    if fig_coords:
        ax_or_fig.text(x, y, letter, transform=ax_or_fig.transFigure, **kw)
    else:
        ax_or_fig.text(x, y, letter, transform=ax_or_fig.transAxes, **kw)


def family_legend(ax, loc="lower center", anchor=(0.5, 1.0), ncol=3):
    h = [Line2D([], [], marker=MARK[f], ls="none", ms=3.6, mfc=FAM[f], mec="white", mew=0.4, label=FAMNAME[f])
         for f in FAM]
    return ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=ncol, handletextpad=0.3, columnspacing=1.2,
                     borderaxespad=0)


# ------------------------------------------------------------------------------------------------- figure 1: system
def box(ax, x, y, w, h, fc="#ffffff", ec=AXIS, lw=0.6, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def arrow(ax, p, q, both=False, color=INK2, lw=0.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=5, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=4))


def txt(ax, x, y, s, size=5.6, weight="normal", color=INK, ha="left", va="top", style="normal", family=None, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, style=style,
                   family=family or mpl.rcParams["font.sans-serif"], zorder=5, linespacing=1.25, **kw)


def fig1():
    """Figure 1: a, the architecture; b, one development request's recorded decisions (dev_v3 nn-05, pplx-decider-flash,
    final configuration, final machine)."""
    ex = json.loads((EXAMPLES / "nn-05.json").read_text())
    sel, cap = ex["steps"][0], ex["steps"][1]
    dec = {d["name"]: d for d in ex["arg_decisions"]}
    W, H = 183, 152
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    S1C, S1L = FAM["s1"], "#e9f1fb"
    MONO = ["Liberation Mono", "DejaVu Sans Mono"]

    # ---- a: architecture ----------------------------------------------------------------------------------------
    txt(ax, 1, H - 1, "a", size=8, weight="bold")
    yA = 78                                                   # bottom of panel a (the output strip)
    # inputs
    box(ax, 4, 128, 36, 19)
    txt(ax, 6, 145.5, "User request", weight="bold")
    txt(ax, 6, 141.6, "a researcher's instruction in\nplain language; may need one\ntool call, several, or none", color=INK2)
    box(ax, 4, 103, 36, 21)
    txt(ax, 6, 122.5, "MCP tool catalogue", weight="bold")
    txt(ax, 6, 118.6, "every tool's name, description\nand JSON input schema\n(42 imaging tools; 115\nbiomedical tools)", color=INK2)
    box(ax, 4, 86, 36, 13, fc="#ffffff", ec=GRID)
    txt(ax, 6, 97.5, "Rules of the call", weight="bold", size=5.4)
    txt(ax, 6, 94.2, "a written value fills one argument;\noptional values need P \u2265 0.7;\nno default or required value invented", size=5.0, color=INK2)
    arrow(ax, (40, 137), (48, 137)); arrow(ax, (40, 113), (48, 113))
    # agent
    box(ax, 48, 86, 58, 61, fc=S1L, ec=S1C, lw=0.8)
    txt(ax, 50.5, 145.5, "Agent", weight="bold", size=6.2)
    txt(ax, 60, 145.5, "deterministic control flow (code)", color=INK2)
    steps = [("1  Tool selection", "one question over every tool + \u201cnone of these\u201d", 8),
             ("2  Tool fit", "the two best tools within 0.3: which one has\nparameters for the values the request gives?", 10.5),
             ("3  Capability check", "\u201ccan this tool do what the request asks?\u201d\nP(yes) < 0.5 \u2192 decline", 10.5),
             ("4  Argument filling", "through the compiler", 8),
             ("5  Ask or call", "a required value not given \u2192 a fixed prompt\nwith its name and schema description; otherwise call the tool", 10.5)]
    top = 141.5
    centres = []
    for t, d, h in steps:
        box(ax, 51, top - h, 52, h, ec=AXIS)
        txt(ax, 53, top - 1.3, t, weight="bold", size=5.6)
        txt(ax, 53, top - 4.4, d, color=INK2, size=5.0)
        centres.append(top - h / 2)
        top -= h + 1.2
    # decision model (right top)
    box(ax, 122, 113, 57, 34, fc="#ffffff", ec=INK2, lw=0.8)
    txt(ax, 124.5, 145.5, "Decision model", weight="bold", size=6.2)
    txt(ax, 124.5, 141.8, "one forward pass; no text is generated", color=INK2)
    box(ax, 124.5, 126.5, 52, 11.5, fc="#f6f6f4", ec=GRID, r=0.8)
    txt(ax, 126.5, 136.7, "State: <request, tool, earlier steps>\nQuestion: <typed question>\nOptions: (A) <option>  (B) <option>  \u2026\nAnswer: \u2190 read-out position", size=5.0,
        family=MONO, color=INK)
    txt(ax, 124.5, 124.6, "read-out at the answer position \u2192 softmax over the\noption labels \u2192 a calibrated probability per option", size=5.0, color=INK2)
    # compiler (right bottom)
    box(ax, 122, 86, 57, 22, fc="#ffffff", ec=INK2, lw=0.8)
    txt(ax, 124.5, 106.5, "Compiler", weight="bold", size=6.2)
    txt(ax, 137.5, 106.5, "deterministic: schema \u2192 typed questions", color=INK2)
    txt(ax, 124.5, 102.6,
        "enum \u2192 its values  \u00b7  boolean \u2192 on / off / not mentioned\n"
        "number, path, text, column \u2192 candidates written in\n"
        "   the request (paths, numbers, quoted text, identifiers,\n"
        "   phrases, documented values, CSV headers) + \u201cnot specified\u201d\n"
        "list \u2192 items one at a time + \u201cno further item\u201d\n"
        "object \u2192 its keys, then a value per key", size=5.05, color=INK2)
    # connections
    yq = (centres[0] + centres[2]) / 2
    arrow(ax, (106, yq), (122, yq), both=True)
    txt(ax, 114, yq + 5.2, "selection\nquestions", size=5.0, color=INK2, ha="center")
    txt(ax, 114, yq - 1.4, "probabilities", size=5.0, color=INK2, ha="center")
    arrow(ax, (106, centres[3]), (122, centres[3]), both=True)
    txt(ax, 114, centres[3] + 5.2, "tool schema,\nrequest", size=5.0, color=INK2, ha="center")
    txt(ax, 114, centres[3] - 1.4, "arguments,\nmissing values", size=5.0, color=INK2, ha="center")
    arrow(ax, (150.5, 108), (150.5, 113), both=True)
    txt(ax, 152.5, 111.9, "argument questions \u21c4 probabilities", size=5.0, color=INK2)
    box(ax, 48, yA, 131, 5.4, fc="#ffffff", ec=INK2)
    arrow(ax, (77, 86), (77, yA + 5.4))
    txt(ax, 50.5, yA + 4.0, "Output", weight="bold", size=5.6)
    txt(ax, 61, yA + 4.0, "an MCP tool call {tool, arguments} with its confidence  \u00b7  a decline (no tool fits)  \u00b7  a prompt for a missing required value (template, no generated text)",
        size=5.0, color=INK2)

    # ---- b: worked example ----------------------------------------------------------------------------------------
    txt(ax, 1, 72, "b", size=8, weight="bold")
    box(ax, 4, 63, 175, 8, fc="#f6f6f4", ec=GRID)
    txt(ax, 6, 69.6, "Request", weight="bold", size=5.6)
    txt(ax, 20, 69.6, "“" + ex["request"] + "”", size=5.6)
    txt(ax, 20, 66.3, "development request; decisions recorded from pplx-decider-flash in the final configuration", size=5.0, color=MUTED)

    # 1: tool selection with bars
    x0, y0, w, h = 4, 4, 52, 55
    box(ax, x0, y0, w, h)
    txt(ax, x0 + 2, y0 + h - 1.5, "1  Tool selection", weight="bold")
    txt(ax, x0 + 2, y0 + h - 5.2, "“Which tool should be called next\nfor this request?”  "
        f"{len(sel['options'])} options", size=5.0, color=INK2)
    top = sorted(zip(sel["probs"], sel["options"]), reverse=True)
    names = [o.split(":")[0].replace(" / ", "/") for _, o in top[:3]]
    probs = [p for p, _ in top[:3]]
    rest = 1 - sum(probs)
    names.append(f"{len(top) - 3} other options"); probs.append(rest)
    bx, bw = x0 + 2, 26
    for i, (n, p) in enumerate(zip(names, probs)):
        yy = y0 + h - 17 - i * 7.5
        txt(ax, bx, yy + 3.6, n.replace("none of these", "“none of these”"), size=5.0)
        ax.add_patch(Rectangle((bx, yy - 1.2), bw, 2.2, fc=GRID, ec="none", zorder=2))
        ax.add_patch(Rectangle((bx, yy - 1.2), max(bw * p, 0.25), 2.2, fc=S1C if i == 0 else MUTED, ec="none", zorder=3))
        txt(ax, bx + bw + 1.5, yy + 1.2, f"{p:.3f}", size=5.0, color=INK if i == 0 else INK2)
    txt(ax, x0 + 2, y0 + 9.5, "2  Capability check", weight="bold")
    py = cap["probs"][cap["options"].index("yes")]
    txt(ax, x0 + 2, y0 + 6.2, f"“Can the tool nnunet_train do what the\nrequest asks?”  yes {py:.3f}", size=5.0, color=INK2)

    # 3: arguments
    x1, w1 = 60, 92
    box(ax, x1, y0, w1, h)
    txt(ax, x1 + 2, y0 + h - 1.5, "3  Argument filling: the compiler's questions and the model's answers", weight="bold")
    cols = [x1 + 2, x1 + 19, x1 + 63, x1 + 78]
    hy = y0 + h - 6.5
    for c, t in zip(cols, ["argument", "options offered", "answer", "outcome"]):
        txt(ax, c, hy, t, size=5.0, color=MUTED)
    ax.plot([x1 + 2, x1 + w1 - 2], [hy - 3.2, hy - 3.2], color=GRID, lw=0.5, zorder=3)
    lines = [("configuration", "enum", "2d · 3d_fullres · 3d_lowres ·\n3d_cascade_fullres · not specified", "set"),
             ("epochs", "number", "100 · 0 · not specified", "set"),
             ("model_name", "text", "liver_v1 · 100 · 0 · not specified", "set"),
             ("fold", "text", "liver_v1 · 100 · 0 · not specified", "is_default")]
    yy = hy - 5.0
    for name, kind, opts, _ in lines:
        d = dec[name]
        txt(ax, cols[0], yy, name, size=5.0, family=MONO)
        txt(ax, cols[0], yy - 2.9, kind, size=5.0, color=MUTED)
        txt(ax, cols[1], yy, opts, size=5.0, color=INK2)
        txt(ax, cols[2], yy, f"{d['value']}", size=5.0, family=MONO, weight="bold")
        txt(ax, cols[2], yy - 2.9, f"p = {d['p']:.3f}", size=5.0, color=INK2)
        outcome = "set" if d["status"] == "set" else "the default:\nnot passed"
        txt(ax, cols[3], yy, outcome, size=5.0, color=INK if d["status"] == "set" else INK2)
        yy -= 7.6 if "\n" in opts else 6.6
    others = [n for n, d in dec.items() if n not in {l[0] for l in lines}]
    txt(ax, cols[0], yy, f"{len(others)} other arguments", size=5.0)
    txt(ax, cols[1], yy, "not specified in the request, or no candidate\nwritten in it (the two path arguments)", size=5.0, color=INK2)
    txt(ax, cols[3], yy, "not passed:\nserver default", size=5.0, color=INK2)
    txt(ax, x1 + 2, y0 + 5.6, "Free values are offered as the request writes them (or in a form derived from it, such as a unit conversion),\nplus \u201cnot specified\u201d; an optional value is set when P \u2265 0.7; a value equal to the documented default is not passed.",
        size=5.0, color=MUTED)

    # 4: call
    x2, w2 = 156, 23
    box(ax, x2, y0, w2, h, fc=S1L, ec=S1C, lw=0.8)
    txt(ax, x2 + 2, y0 + h - 1.5, "4  Tool call", weight="bold")
    call = json.dumps(ex["args_pred"], indent=1).replace('"', "”").replace("”", '"')
    txt(ax, x2 + 2, y0 + h - 6, "nnunet/\nnnunet_train", size=5.0, family=MONO, weight="bold")
    txt(ax, x2 + 2, y0 + h - 13.5, "\n".join(f"{k}:\n  {json.dumps(v)}" for k, v in ex["args_pred"].items()),
        size=5.0, family=MONO)
    txt(ax, x2 + 2, y0 + 12, f"confidence\n{ex['confidence']:.3f}", size=5.0, color=INK2)
    txt(ax, x2 + 2, y0 + 5.6, f"{ex['model_calls']} model calls\n{ex['seconds']:.1f} s", size=5.0, color=INK2)
    arrow(ax, (56, 31), (60, 31)); arrow(ax, (152, 31), (156, 31))
    save(fig, "fig1_system")


# ----------------------------------------------------------------------------------------------- figure 2: accuracy
def fig2(M, R, RECS):
    S, n = M["systems"], M["n"]
    keys = sorted(systems(), key=lambda s: -S[s[0]]["full"])
    valci = {}
    for k, *_ in keys:
        v, _ = per_request(R, RECS[k])
        valci[k] = cluster_ci(v)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 108 * MM), sharey=True, gridspec_kw=dict(wspace=0.13))
    spec = [("tool", "Tool calling", n["main"], "tool_ci"), ("vals", "Argument filling", n["values"], None),
            ("full", "Fully correct", n["main"], "full_ci")]
    for ax, (m, title, N, ci) in zip(axes, spec):
        for y, (k, fam, model, name) in enumerate(keys):
            v = S[k]
            p = 100 * v[m] / N
            lo, hi = v[ci] if ci else valci[k]
            ax.plot([lo, hi], [y, y], color=FAM[fam], lw=0.9, solid_capstyle="butt", zorder=2, alpha=0.9)
            ax.plot(p, y, marker=MARK[fam], ms=3.6 if fam != "think" else 3.9, mfc=FAM[fam], mec="white", mew=0.4,
                    ls="none", zorder=3)
        ax.set_xlim(50, 100)
        ax.set_xticks(range(50, 101, 10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"{title} (% of {N:,} {'argument values' if m == 'vals' else 'requests'})")
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([label(f, nm) for _, f, _, nm in keys])
    for t, (_, f, _, _) in zip(axes[0].get_yticklabels(), keys):
        t.set_fontweight("bold" if f == "s1" else "normal")
    axes[0].set_ylim(len(keys) - 0.4, -0.6)
    family_legend(axes[1], anchor=(0.5, 1.07))
    for ax, l in zip(axes, "abc"):
        panel(ax, l, -0.02 if l != "a" else -0.62, 1.07)
    save(fig, "fig2_accuracy")
    return valci


# --------------------------------------------------------------------------------------------- figure 3: pairwise
def fig3(M):
    D, S = M["differences"], M["systems"]
    rows_ = [(c, m) for c in ("think", "nothink") for m in LLM]
    cmap = LinearSegmentedColormap.from_list("div", ["#b2302f", "#e66767", "#f4b9b8", "#f0efec", "#b7d3f6", "#5598e7",
                                                     "#184f95"])
    norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 92 * MM), sharey=True, gridspec_kw=dict(wspace=0.05))
    titles = {"full": "Fully correct", "tool": "Tool calling", "vals": "Argument filling"}
    for ax, meas, letter in zip(axes, ("full", "tool", "vals"), "abc"):
        Z = np.zeros((len(rows_), len(S1))); sig = np.zeros_like(Z, bool)
        for i, (c, m) in enumerate(rows_):
            for j, s in enumerate(S1):
                d = D[f"{meas}|{c}:{m}|s1:{s}"]
                Z[i, j], sig[i, j] = d["diff_pp"], d["p_holm"] < 0.05
        for i in range(len(rows_)):
            for j in range(len(S1)):
                col = cmap(norm(np.clip(Z[i, j], -15, 15)))
                ax.add_patch(Rectangle((j, i), 1, 1, fc=col if sig[i, j] else "#ffffff", ec="white", lw=0.8))
                if not sig[i, j]:
                    ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc="none", ec=AXIS, lw=0.4))
                dark = sig[i, j] and abs(Z[i, j]) > 9
                ax.text(j + 0.5, i + 0.52, f"{Z[i, j]:+.1f}", ha="center", va="center", fontsize=5.0,
                        color="#ffffff" if dark else (INK if sig[i, j] else MUTED))
        ax.set_xlim(0, len(S1)); ax.set_ylim(len(rows_), 0)
        ax.axhline(9, color=INK2, lw=0.6)
        ax.set_xticks(np.arange(len(S1)) + 0.5)
        ax.set_xticklabels([S1_NAME[s] for s in S1], rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        t = M["systems"][f"primary|{meas}"]
        ax.text(0, -1.25, titles[meas], fontsize=6.5, fontweight="bold", va="bottom")
        ax.text(0, -0.35, f"System 1 ahead {t['S1']}  \u00b7  LLM ahead {t['LLM']}  \u00b7  no difference {t['ns']}",
                fontsize=5.0, color=INK2, va="bottom")
        panel(ax, letter, -0.04 if letter != "a" else -0.66, 1.19)
    axes[0].set_yticks(np.arange(len(rows_)) + 0.5)
    axes[0].set_yticklabels([LLM_NAME[m] for _, m in rows_])
    axes[0].tick_params(axis="y", length=0, pad=2)
    for y0, t in ((4.5, "reasoning on"), (13.5, "reasoning off")):
        axes[0].text(-0.62, 1 - y0 / len(rows_), t, transform=axes[0].transAxes, rotation=90, ha="center", va="center",
                     fontsize=5.5, color=INK2)
    for x0, y0 in ((-0.585, 0), (-0.585, 9)):
        axes[0].annotate("", xy=(x0, (y0 + 0.15)), xytext=(x0, y0 + 8.85), xycoords=("axes fraction", "data"),
                         arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6))
    cax = fig.add_axes([0.55, -0.075, 0.30, 0.018])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-15, -10, -5, 0, 5, 10, 15]); cb.outline.set_visible(False); cb.ax.tick_params(length=1.5, labelsize=5)
    cb.set_label("System 1 minus LLM (percentage points); blue: System 1 ahead, red: LLM ahead", fontsize=5.3)
    fig.text(0.13, -0.065, "Shaded: significant after Holm correction over 108 pairs (P < 0.05).\nOutlined white: no significant difference.",
             fontsize=5.0, color=INK2, va="center")
    save(fig, "fig3_pairwise")


# ------------------------------------------------------------------------------------------------ figure 4: cost
def fig5(M):
    S = M["systems"]
    keys = systems()
    fig = plt.figure(figsize=(180 * MM, 100 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.62)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # every agent is labelled: (dx, dy) offset in points, horizontal alignment, leader line to the marker
    R, L = ((4, -0.3), "left", False), ((-4, -0.3), "right", False)
    place = {"s1:pplx-decider-27b-embed": R, "s1:openjev": L, "s1:clef": L, "s1:pplx-decider-flash": L,
             "s1:decider-35b-a3b": R, "s1:decider-4b": ((0, 5), "center", False),
             "think:qwen3.5-35b-a3b": ((3, -4.5), "left", False), "think:nemotron-bolt-74b": R,
             "think:qwen3.5-27b": R, "think:qwen3.8-27b": L, "think:muse-glimmer-30b": L,
             "think:gemma-4-26b-a4b": R, "think:qwen3.5-4b": L, "think:gemma-4-e4b": R, "think:nemotron-bolt-e4b": R,
             "nothink:qwen3.5-27b": R, "nothink:muse-glimmer-30b": R, "nothink:qwen3.5-35b-a3b": L,
             "nothink:qwen3.8-27b": ((3, 4.5), "left", False), "nothink:gemma-4-26b-a4b": R,
             "nothink:nemotron-bolt-74b": R, "nothink:nemotron-bolt-e4b": ((-8, -13), "right", True),
             "nothink:qwen3.5-4b": L, "nothink:gemma-4-e4b": R}
    for k, fam, model, name in keys:
        x, y = S[k]["cost"]["wh_per_full"], 100 * S[k]["full"] / M["n"]["main"]
        ax.plot(x, y, marker=MARK[fam], ms=4.2 if fam != "think" else 4.5, mfc=FAM[fam], mec="white", mew=0.45,
                ls="none", zorder=3)
        (dx, dy), ha, lead = place[k]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.0, ha=ha, va="center",
                    color=INK if fam == "s1" else INK2, fontweight="bold" if fam == "s1" else "normal", zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.35, color=MUTED, shrinkA=1.5, shrinkB=2.5) if lead else None)
    ax.set_xscale("log"); ax.set_xlim(0.02, 5); ax.set_ylim(58, 96)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("GPU energy per fully correct request (Wh, log scale)")
    ax.set_ylabel(f"Fully correct (% of {M['n']['main']:,} requests)")
    ax.tick_params(axis="y", length=2)
    family_legend(ax, loc="lower left", anchor=(0.0, 1.01))
    panel(ax, "a", -0.13, 1.08)

    order = sorted(keys, key=lambda t: S[t[0]]["cost"]["seconds_mean"])
    for y, (k, fam, model, name) in enumerate(order):
        c = S[k]["cost"]
        bx.plot([c["seconds_mean"]], [y], marker=MARK[fam], ms=3.6 if fam != "think" else 3.9, mfc=FAM[fam],
                mec="white", mew=0.4, ls="none", zorder=3)
        bx.plot([0.5, c["seconds_mean"]], [y, y], color=GRID, lw=0.5, zorder=1)
    bx.set_xscale("log"); bx.set_xlim(0.5, 40)
    bx.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([label(f, nm) for _, f, _, nm in order], fontsize=5.0)
    for t, (_, f, _, _) in zip(bx.get_yticklabels(), order):
        t.set_fontweight("bold" if f == "s1" else "normal")
    bx.set_ylim(len(order) - 0.4, -0.6)
    bx.spines["left"].set_visible(False); bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Time per request (s, mean, log scale)")
    panel(bx, "b", -0.75, 1.08)
    save(fig, "fig5_efficiency")


# ------------------------------------------------------------------------------- figure 5: decline and ask behaviour
ASKCAT = [("right tool, missing value left out (correct)", "right tool, missing value left for the user (correct)", "#008300"),
          ("no tool call", "no tool call (asks in text or declines)", "#6250d6"),
          ("right tool, missing value invented", "right tool, missing value filled in", "#898781"),
          ("wrong tool", "wrong tool", "#eda100"),
          ("right tool, other", "right tool, other error", "#c3c2b7")]


def fig4(M, R, RECS):
    """Reliability: invented content, underspecified requests, out-of-scope requests."""
    S = M["systems"]
    FB = RES / "fabrication"
    D = json.loads((FB / "fabrication.json").read_text())
    P = json.loads((FB / "precision.json").read_text())
    keys = systems()
    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 0.42], wspace=0.06, hspace=0.32)
    ax, bx, cx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # a: calls with invented content
    for y, (k, fam, model, name) in enumerate(keys):
        v = D[k]
        if fam == "s1":
            ax.plot(0, y, marker=MARK[fam], ms=3.6, mfc=FAM[fam], mec="white", mew=0.4, ls="none", zorder=3, clip_on=False)
            ax.text(0.8, y, f"0 of {v['calls']:,} calls", va="center", fontsize=5.0, color=INK)
            continue
        lo, hi = 100 * v["F4"] * P[fam]["strict"] / v["calls"], 100 * v["F4"] * P[fam]["high"] / v["calls"]
        ax.barh(y, lo, height=0.62, color=FAM[fam], edgecolor="white", linewidth=0.5, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=0.7, zorder=3)
        ax.plot([hi, hi], [y - 0.2, y + 0.2], color=INK2, lw=0.7, zorder=3)
    ax.set_xlim(0, 30)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([label(f, nm) for _, f, _, nm in keys], fontsize=5.0)
    for t, (_, f, _, _) in zip(ax.get_yticklabels(), keys):
        t.set_fontweight("bold" if f == "s1" else "normal")
    ax.set_ylim(len(keys) - 0.4, -0.6)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    for yb in (5.5, 14.5):
        ax.axhline(yb, color=INK2, lw=0.5)
    ax.set_xlabel("Calls with at least one invented argument value (%)")
    family_legend(ax, loc="lower left", anchor=(-0.02, 1.01))
    panel(ax, "a", -0.62, 1.10)

    # b: what each agent did with the underspecified requests
    n_ask = M["n"]["asks"]
    for y, (k, fam, model, name) in enumerate(keys):
        A = S[k]["ask"]
        left = 0
        for key, _, col in ASKCAT:
            v = 100 * A.get(key, 0) / n_ask
            if v > 0:
                bx.barh(y, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.5)
            left += v
    bx.set_ylim(len(keys) - 0.4, -0.6); bx.set_yticks([]); bx.set_xlim(0, 100)
    bx.spines["left"].set_visible(False)
    for yb in (5.5, 14.5):
        bx.axhline(yb, color=INK2, lw=0.5)
    bx.set_xlabel(f"Underspecified requests (% of {n_ask})")
    h = [Rectangle((0, 0), 1, 1, fc=c, ec="white") for _, _, c in ASKCAT]
    bx.legend(h, [l for _, l, _ in ASKCAT], loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=1, fontsize=5.0,
              handlelength=1.0, handleheight=0.8)
    panel(bx, "b", -0.04, 1.10)

    # c: out-of-scope requests declined, by kind
    kinds = [("oos-far", "far: unrelated to the tools"), ("oos-near", "near: a related task no tool offers"),
             ("oos-modality", "modality: a tool does the task, not on this input")]
    rates = {}
    for k, fam, *_ in keys:
        _, oos = per_request(R, RECS[k])
        rates[k] = {kd: 100 * np.mean([d for kk, d in oos if kk == kd]) for kd, _ in kinds}
    counts = {kd: sum(1 for r in R.values() if not r.get("tool") and kd in r["tags"]) for kd, _ in kinds}
    for i, (kd, nm) in enumerate(kinds):
        for j, fam in enumerate(FAM):
            xs = i + (j - 1) * 0.22
            vals = [rates[k][kd] for k, f, *_ in keys if f == fam]
            jit = np.linspace(-0.06, 0.06, len(vals))
            cx.plot(xs + jit, vals, marker=MARK[fam], ms=3.0, mfc=FAM[fam], mec="white", mew=0.35, ls="none", zorder=3)
            cx.plot([xs - 0.09, xs + 0.09], [np.mean(vals)] * 2, color=FAM[fam], lw=1.2, zorder=2)
    cx.set_xticks(range(3)); cx.set_xticklabels([f"{nm} (n = {counts[kd]})" for kd, nm in kinds])
    cx.set_xlim(-0.5, 2.5); cx.set_ylim(0, 102); cx.set_ylabel("Declined (%)")
    cx.tick_params(axis="x", length=0); cx.tick_params(axis="y", length=2)
    cx.set_title("Out-of-scope requests declined, by kind", loc="left", fontsize=6.0, fontweight="normal")
    panel(cx, "c", -0.07, 1.16)
    save(fig, "fig4_reliability")
    return rates


EXAMPLE_OPTIONS = {
    ("nn-05", "imaging_tools.json", "nnunet/nnunet_train"): {
        "configuration": ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"], "epochs": ["100", "0"],
        "model_name": ["liver_v1", "100", "0"], "fold": ["liver_v1", "100", "0"]},
}


def check_examples():
    """The option lists printed in Figure 1b are the release compiler's own questions."""
    from s1agent.backends.lexical import LexicalBackend
    from s1agent.catalog import load_catalogue
    from s1agent.compiler import NOT_SPECIFIED, ArgumentFiller
    from s1agent.config import agent_settings
    cfg = agent_settings()
    filler = ArgumentFiller(LexicalBackend(), cfg.get("policy_defaults"), cfg["workspace"])
    for (i, cat, tool), want in EXAMPLE_OPTIONS.items():
        ex = json.loads((EXAMPLES / f"{i}.json").read_text())
        t = {x.qualified: x for x in load_catalogue(DATA / "catalogues" / cat)}[tool]
        got = {n: [o for o in sp["options"] if o != NOT_SPECIFIED] for n, sp, _ in filler.questions(ex["request"], t) if sp}
        for arg, opts in want.items():
            if got.get(arg) != opts:
                sys.exit(f"STOP: {i} {arg}: the compiler offers {got.get(arg)}, the figure prints {opts}")
    print("worked examples: the printed options are the compiler's")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "runs" / "figures"))
    OUT = Path(ap.parse_args().out)
    M = json.loads((RES / "analysis" / "metrics.json").read_text())
    R = rows()
    assert sum(1 for r in R.values() if r.get("tool") and not r.get("ask")) == M["n"]["complete"]
    check_examples()
    RECS = agent_records(R)
    fig1()
    valci = fig2(M, R, RECS)
    fig3(M)
    rates = fig4(M, R, RECS)
    fig5(M)
    (OUT / "figure_data.json").write_text(json.dumps(dict(values_ci=valci, oos_declined=rates), indent=1, sort_keys=True))
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
