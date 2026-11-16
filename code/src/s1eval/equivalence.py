"""When two argument values make the server do the same thing: rules taken from the MCP servers' own code.

The strict verdict (s1eval.scoring.same) counts a value as right only if it equals the label after case and list
normalisation, or if it is omitted and the label equals the documented default. Some servers accept more than that, and
their code shows exactly what:

  equivalent     the server maps the value through an explicit name table before using it, so two values with the same
                 entry give the same execution ("random forest" and "rf" for feature_importance_analysis.method;
                 "kidneys" and "kidney_left, kidney_right" for totalsegmentator_segment.roi_subset). Lists are compared
                 as sets of mapped items.
  conditional    the argument is omitted and the server auto-detects it from the data by searching a fixed list of
                 column names that contains the labelled value. Whether the labelled column is the one found depends on
                 the other columns of the file, which the evaluation does not have, so this is reported separately.

Imaging servers: the name tables and column lists are the literals of the servers' source files, extracted unchanged
into data/reco_name_tables.json (file and variable of each). Only exact table entries are used; the servers' fuzzy
fallbacks (substring and similarity matching) are not, which keeps the rules conservative.

Biomedical servers (data/biomed, files copied unchanged from the servers' releases; licences there):
  server functions  ClinicalTrials.gov `statusFilter` and `phaseFilter` (normalizeStatusFilter, normalizePhaseFilter:
                    "Phase 3" and "PHASE3" are one phase) and find_eligible's `location` country and US state
                    (registryCountry, registryState: synonyms the server documents as matching exactly the same studies,
                    e.g. "UK" and "United Kingdom"). The server's own JavaScript runs through data/biomed/normalise.mjs
                    (Node.js; `node` on the PATH, or $S1EVAL_NODE), so nothing is re-implemented.
  name table        ChEMBL `assay_type` (ASSAY_TYPE_CODES: "binding" -> B), read from the server's source; keys trimmed
                    and upper-cased as the server does.
  search text       arguments the server sends to a database text search: two values are the same if they have the
                    same content words (case folded, apostrophes dropped, split on everything but letters and digits,
                    English stop words of scikit-learn 1.4.2 removed, any order); lists as sets of items. SEARCH_TEXT_RULE
                    = False switches this rule off (the strict sensitivity analysis). The stop words include `and`,
                    `or`, `not`, which the registry also reads as Boolean operators, so `A NOT B` and `A B` count as
                    the same search.
"""
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
BIOMED = DATA / "biomed"
SEARCH_TEXT_RULE = True

_CT_SEARCH = ("clinicaltrials_search_studies", "clinicaltrials_get_study_count")
# (tool, argument) -> normaliser of the ClinicalTrials.gov server (data/biomed/normalise.mjs)
SERVER_FUNCTIONS = {**{(t, "statusFilter"): "status" for t in _CT_SEARCH},
                    **{(t, "phaseFilter"): "phase" for t in _CT_SEARCH},
                    ("clinicaltrials_find_eligible", "location"): "location"}
# (tool, argument) -> (JavaScript file, `new Map([...])` literal holding the name table)
JS_TABLES = {("chembl_get_bioactivities", "assay_type"): ("chembl/input-normalization.js", "ASSAY_TYPE_CODES")}
# (tool, argument) the server sends to a database text search
SEARCH_TEXT = {*((t, a) for t in _CT_SEARCH for a in ("query", "conditionQuery", "interventionQuery", "locationQuery",
                                                      "sponsorQuery", "titleQuery", "outcomeQuery")),
               ("clinicaltrials_find_eligible", "conditions"), ("clinicaltrials_get_field_definitions", "query"),
               ("chembl_search_molecules", "query")}

# (tool, argument) -> (server source file, variable holding the name table, key normalisation the server applies)
ALIASES = {
    ("feature_importance_analysis", "method"): ("feature_importance_analysis_tool.py", "method_mapping", "squash"),
    ("pycaret_classification", "include_models"): ("pycaret_classification_tool.py", "MODEL_NAME_MAPPING", "lower"),
    ("pycaret_classification", "exclude_models"): ("pycaret_classification_tool.py", "MODEL_NAME_MAPPING", "lower"),
    ("totalsegmentator_segment", "roi_subset"): ("totalsegmentator_tool.py", "ORGAN_NAME_MAPPING", "lower"),
    ("totalsegmentator_segment", "roi_subset_robust"): ("totalsegmentator_tool.py", "ORGAN_NAME_MAPPING", "lower"),
    ("pyradiomics_extract_features", "image_types"):
        ("pyradiomics_feature_extraction_tool.py", "FILTER_NAME_MAPPING", "squash"),
}
LISTS = {("pycaret_classification", "include_models"), ("pycaret_classification", "exclude_models"),
         ("totalsegmentator_segment", "roi_subset"), ("totalsegmentator_segment", "roi_subset_robust"),
         ("pyradiomics_extract_features", "image_types")}
# (tool, argument) -> (server source file, variable holding the column names searched when the argument is omitted)
AUTODETECT = {
    ("feature_importance_analysis", "target_column"): ("feature_importance_analysis_tool.py", "target_patterns"),
    ("pycaret_classification", "target_column"): ("pycaret_classification_tool.py", "common_target_names"),
    ("pycaret_regression", "target_column"): ("pycaret_regression_tool.py", "common_target_names"),
    ("pycaret_classification_inference", "ground_truth_column"):
        ("pycaret_classification_inference_tool.py", "common_ground_truth_names"),
}


@lru_cache(maxsize=None)
def _tables():
    return json.loads((DATA / "reco_name_tables.json").read_text())["tables"]


def _literal(filename, name):
    """The literal assigned to `name` in a server source file (data/reco_name_tables.json)."""
    return _tables()[f"{filename}:{name}"]


def _key(s, how):
    s = str(s).strip().lower()
    return re.sub(r"[-_\s]+", "", s) if how == "squash" else s


def _items(v):
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [x for x in (p.strip() for p in str(v).split(",")) if x]


def canonical(tool, arg, value):
    """The value as the server will use it, according to its name table (None when no table applies)."""
    spec = ALIASES.get((tool, arg))
    if spec is None or value is None:
        return None
    filename, name, how = spec
    table = {_key(k, how): v for k, v in _literal(filename, name).items()}

    def one(x):
        m = table.get(_key(x, how), table.get(_key(x, "lower")))
        if m is None:
            return frozenset([_key(x, "lower")])
        return frozenset(m) if isinstance(m, (list, tuple)) else frozenset([str(m)])

    if (tool, arg) in LISTS:
        return frozenset().union(*[one(x) for x in _items(value)]) if _items(value) else frozenset()
    return one(value)


class _Node:
    """One Node.js process running data/biomed/normalise.mjs, started on first use."""
    proc = None

    @classmethod
    def ask(cls, **q):
        if cls.proc is None or cls.proc.poll() is not None:
            node = os.environ.get("S1EVAL_NODE") or shutil.which("node")
            if not node:
                raise RuntimeError("Node.js is needed to score the ClinicalTrials.gov arguments (put `node` on the PATH "
                                   "or set S1EVAL_NODE)")
            cls.proc = subprocess.Popen([node, str(BIOMED / "normalise.mjs")], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        cls.proc.stdin.write(json.dumps(q) + "\n")
        cls.proc.stdin.flush()
        return json.loads(cls.proc.stdout.readline())["out"]


@lru_cache(maxsize=None)
def _node(fn, value_json, country_json="null"):
    return _Node.ask(fn=fn, value=json.loads(value_json), country=json.loads(country_json))


@lru_cache(maxsize=None)
def _js_table(filename, name):
    m = re.search(rf"const {name} = new Map\(\[(.*?)\]\);", (BIOMED / filename).read_text(), re.S)
    if m is None:
        raise KeyError(f"{name} not found in {filename}")
    return dict(re.findall(r"\[\s*'([^']*)'\s*,\s*'([^']*)'\s*\]", m.group(1)))


@lru_cache(maxsize=None)
def _stop_words():
    return frozenset((BIOMED / "stopwords_english.txt").read_text().split())


def content_words(text):
    s = re.sub(r"['’]", "", str(text).casefold())
    return frozenset(w for w in re.split(r"[\W_]+", s) if w and w not in _stop_words())


def _bio_canonical(tool, arg, value):
    """The value as a biomedical server will use it (None: no rule applies or the server cannot read it)."""
    key = (tool, arg)
    if value is None:
        return None
    if key in SERVER_FUNCTIONS:
        fn = SERVER_FUNCTIONS[key]
        if fn == "location":
            if not isinstance(value, dict):
                return None
            country = json.dumps(str(value.get("country") or ""))
            out = {}
            for k, v in value.items():
                if v is None:
                    continue
                if k == "country":
                    v = _node("country", json.dumps(str(v)))
                elif k == "state":
                    v = _node("state", json.dumps(str(v)), country)
                out[str(k).strip().lower()] = str(v).strip().lower()
            return frozenset(out.items())
        out = _node(fn, json.dumps(value if isinstance(value, (str, list)) else str(value)))
        # an entry may hold several values joined by ',' or '|', which the server splits on
        return None if out is None else frozenset(p.strip() for v in out for p in re.split(r"[,|]", str(v)) if p.strip())
    if key in JS_TABLES:
        k = str(value).strip().upper()
        return _js_table(*JS_TABLES[key]).get(k, k)
    if key in SEARCH_TEXT and SEARCH_TEXT_RULE:
        if isinstance(value, (list, tuple)):
            return frozenset(content_words(x) for x in value)
        return content_words(value)
    return None


def equivalent(tool, arg, pred, gold):
    """True if the server maps both values to the same thing through its name table (or, for the biomedical servers,
    its normalising functions or a database text search)."""
    if (tool, arg) in ALIASES:
        c = canonical(tool, arg, pred)
        return c is not None and c == canonical(tool, arg, gold)
    c = _bio_canonical(tool, arg, pred)
    return c is not None and c == _bio_canonical(tool, arg, gold)


def autodetected(tool, arg, gold):
    """True if omitting the argument lets the server search for a column with the labelled name."""
    spec = AUTODETECT.get((tool, arg))
    if spec is None or not isinstance(gold, str):
        return False
    return gold.strip().lower() in {str(n).strip().lower() for n in _literal(*spec)}


# ---- scoring sensitivity: PyCaret's own model names for the regression server ------------------------------------------
_PYCARET_ARGS = {("pycaret_regression", "include_models"), ("pycaret_regression", "exclude_models")}


def _squash(s):
    return re.sub(r"[-_\s]+", "", str(s).strip().lower())


@lru_cache(maxsize=None)
def _pycaret_table():
    """{normalised ID or name: ID} of PyCaret 3.3.2's regression models: an item matches when it equals the ID or the
    name, without case, spaces, '-' and '_', with or without the name's trailing 'Regressor' or 'Regression'."""
    out = {}
    for i, name in json.loads((DATA / "pycaret_regression_models.json").read_text())["models"]:
        short = re.sub(r"\s+(?:regressor|regression)$", "", name, flags=re.I)
        for key in (i, name, short):
            out.setdefault(_squash(key), i)
    return out


def _pycaret_canonical(value):
    if value is None:
        return None
    items = value if isinstance(value, list) else [x for x in (p.strip() for p in str(value).split(",")) if x]
    return frozenset(_pycaret_table().get(_squash(x), _squash(x)) for x in items)


def use_pycaret_lists():
    """Sensitivity analysis: the PyCaret regression server has no name table of its own and passes model names to
    PyCaret, which takes its own IDs and names; with this switched on, `include_models` and `exclude_models` of
    pycaret_regression are compared through PyCaret's table (lists as sets)."""
    global equivalent
    base = equivalent

    def with_pycaret(tool, arg, pred, gold):
        if (tool, arg) in _PYCARET_ARGS:
            c = _pycaret_canonical(pred)
            return c is not None and c == _pycaret_canonical(gold)
        return base(tool, arg, pred, gold)
    equivalent = with_pycaret
