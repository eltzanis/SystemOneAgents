"""The compiler: fill a tool's arguments from its JSON schema with typed questions, never with generated text.

Each property of the tool's inputSchema becomes one or more multiple-choice questions over finite options, built from
the schema and the request:

    enum               a choice over the enum values (plus "not specified" when the argument is optional)
    boolean            on / off / not mentioned
    integer, number    a choice over the numbers the request states, in digits, in words or in another form the argument
                       takes ('half a minute' for an argument in seconds, '2 kb' for a length in bp; candidates.py)
    string, path       a choice over the paths the request writes, and paths it names without writing them in full (the
                       folder of a written file, a file name placed in a written folder)
    string, column     a choice over the columns of the CSV file another argument points to, and the request's words
    string, text       a choice over quoted strings, identifiers, words and phrases, and the values the description
                       documents that the request names; a closed set of documented values is offered alone
    string, list       ("comma-separated" in the description) and array: items picked one at a time. The first item
    array              question ("which item ...?", with "none" as an option) is asked with the scalar questions; further
                       items are asked as "which item comes next?" over every remaining candidate plus "no further item",
                       so candidates compete with each other instead of each being judged alone. An array of arrays or
                       of objects is offered the lists the request writes, whole, as one value question
    object             the dictionaries the request writes, as one value question; otherwise a yes/no gate, the object's
                       keys picked like list items, then one value question per key

Every free-valued property, and every optional one, has a "not specified" option: for an optional argument it keeps the
server's default; for a required one it means the request does not give the value. A required argument left without a
value is reported as missing, so the agent asks the user for it instead of inventing one.

Rules that keep the call faithful to the request:
  - a value fills as many arguments (or list items) as the request writes it: once, unless the request writes it again
    or says it applies to several ('both', 'each way', 'in the same folder');
  - the arguments that want the same value settle strongest first: the argument whose best answer is the most probable
    takes it, the others go on without it. When two first answers take the same value and each would be accepted, the
    model is asked which argument the value belongs to, and that answer decides only when it reaches the optional bar;
  - an optional argument is set only when the model is confident the request gives it at all: the probability mass on
    the values still allowed (1 - P(not specified) - P(values already taken)), or P(on/off) for a boolean, must reach
    optional_min_conf; a list's first item must reach list_min_conf;
  - a value equal to the argument's documented default is not passed (the server would use it anyway);
  - a value already taken stays visible as an option (removing options changes the model's answer) but cannot be
    taken again.

All question texts and option texts are part of the answer-store key (backends.base.AnswerStore): changing any of them
changes the questions the model is asked.
"""
import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import jsonschema

from s1agent import candidates as C
from s1agent.backends.lexical import _stem

NOT_SPECIFIED = "not specified in the request (keep the default)"
ON, OFF = "yes: turn it on", "no: turn it off"
UNMENTIONED = "not mentioned in the request (keep the default)"
NO_ITEM = "none: the request names no item for it"
NO_MORE = "no further item: the list is complete"

_PATHLIKE = re.compile(r"(^|_)(path|dir|directory|folder|file|root)$")
_COLUMNLIKE = re.compile(r"column|feature|target|label|outcome")
_EXAMPLE_KEY = re.compile(r"""["']([A-Za-z][\w]*)["']\s*:""")
_CAMEL = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]*|ID|UID)+)\b")


@dataclass
class ArgDecision:
    name: str
    kind: str
    status: str                   # set | default | policy | missing | low_conf | is_default | unsupported
    value: object = None
    confidence: float = None
    options: list = field(default_factory=list)


@dataclass
class Filled:
    args: dict
    missing: list
    uncertain: list
    decisions: list
    errors: list
    model_calls: int = 0


def _type(prop):
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def flat_schema(prop):
    """A union schema (anyOf/oneOf, e.g. pydantic's Optional[X]) read as one type: null branches are dropped and a single
    remaining branch is merged into the argument's schema (the argument's own description, default and title kept); of
    a string and an array the array is taken, of a number and a string the number. Other unions stay as they are (and
    are not supported). List items are read the same way."""
    if not isinstance(prop, dict):
        return prop
    out = dict(prop)
    if isinstance(out.get("items"), dict):
        out["items"] = flat_schema(out["items"])
    key = next((k for k in ("anyOf", "oneOf") if isinstance(prop.get(k), list)), None)
    if key is None:
        return out
    branches = [b for b in prop[key] if isinstance(b, dict) and b.get("type") != "null"]
    types = {_type(b) for b in branches}
    if len(branches) == 1:
        pick = branches[0]
    elif len(branches) == 2 and types == {"string", "array"}:
        pick = next(b for b in branches if _type(b) == "array")
    elif len(branches) == 2 and "string" in types and len(types & {"integer", "number"}) == 1:
        pick = next(b for b in branches if _type(b) in ("integer", "number"))
    else:
        return out
    merged = {k: v for k, v in out.items() if k != key}
    for k, v in flat_schema(pick).items():
        merged.setdefault(k, v)
    return merged


def param_kind(name, prop):
    """How an argument is asked: enum, bool, number, array, object, path, list, column, text or unsupported."""
    t = _type(prop)
    desc = (prop.get("description") or "").lower()
    if "enum" in prop:
        return "enum"
    if t == "boolean":
        return "bool"
    if t in ("integer", "number"):
        return "number"
    if t == "array":
        return "array"
    if t == "object":
        return "object"
    if t == "string" or (t is None and not any(k in prop for k in ("anyOf", "oneOf", "properties"))):
        if _PATHLIKE.search(name) or desc.startswith(("path to", "directory", "folder", "full path")):
            return "path"
        if "comma-separated" in desc or "comma separated" in desc:
            return "list"
        if _COLUMNLIKE.search(name):
            return "column"
        return "text"
    return "unsupported"


_DEFAULT_RE = re.compile(r"default(?:s)?(?: is| to|:)?\s*[\'\"`]?([\w.-]+)", re.I)
_LIST_DEFAULT_RE = re.compile(r"default(?:s)?(?: is| to|:)?\s*(\[[^\]]*\])", re.I)


def documented_default(prop):
    """The schema's "default", else a "Default: X" stated in the description (a list too: 'Default: [1, 1, 1]');
    None when neither exists."""
    if "default" in prop:
        return prop["default"]
    m = _LIST_DEFAULT_RE.search(prop.get("description") or "")
    if m:
        try:
            return list(ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            pass
    m = _DEFAULT_RE.search(prop.get("description") or "")
    if not m:
        return None
    v = m.group(1).rstrip(".")
    return {"true": True, "false": False}.get(v.lower(), v)


def is_default(value, prop):
    """Is `value` the argument's documented default (numbers by value, text without case, lists item by item)?"""
    d = documented_default(prop)
    if d is None or isinstance(d, bool) != isinstance(value, bool):
        return False
    if isinstance(d, list) or isinstance(value, list):
        try:
            return isinstance(d, list) and isinstance(value, list) and [float(x) for x in d] == [float(x) for x in value]
        except (TypeError, ValueError):
            return [str(x).lower() for x in d] == [str(x).lower() for x in value]
    try:
        return float(d) == float(value)
    except (TypeError, ValueError):
        return str(d).lower() == str(value).lower()


_QUOTED_VALUE = re.compile(r"""['"`]([A-Za-z][\w-]*(?: [A-Za-z][\w-]*)*)['"`]""")
_LISTED = re.compile(r":\s*([A-Za-z][\w-]*(?:,\s*(?:or\s+)?[A-Za-z][\w-]*)+)")
_FILE_NAME = re.compile(r"(^|_)(csv|tsv|filename)$")
_FILE_DESC = re.compile(r"\bfilename\b|^csv\b", re.I)
_LIKE = re.compile(r"^\s*like\s+`?(\w+)`?", re.I)
_FLEXIBLE = re.compile(r"natural[- ]language|flexible names|full names", re.I)
_SEARCH_DESC = re.compile(r"\bsearch(?:es|ing)?\b|\bquer(?:y|ies)\b|\bmatch(?:es|ed|ing)?\b", re.I)
_PREFIX_NAME = re.compile(r"(^|_)prefix$", re.I)
_SEARCH_WORDS = re.compile(r"\bsearch(?:es|ing)?\b|\bquer(?:y|ies)\b|\bmatched as\b", re.I)
_COLUMN_NAME = re.compile(r"(^|_)(column|col|field|fields)($|_)", re.I)
_TUPLE_DESC = re.compile(r"\[[^\[\]]+,[^\[\]]+\]")                    # '[chr, start, end, name] tuples'


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_EXAMPLE_LIST = re.compile(r"\be\.g\.,?\s*([A-Za-z][\w-]*(?:,\s*(?:or\s+)?[A-Za-z][\w-]*)+)")
_OPEN_LIST = re.compile(r"\be\.g\.|\bfor example\b|\bsuch as\b|\betc\b|\bi\.e\.", re.I)


def documented_values(prop, examples=False):
    """The values a free-text argument's description lists: quoted words ('heavy', 'Random Forest') and the list after a
    colon ('model type: resnet18, resnet34, or resnet50'); with examples, also an unquoted list of examples ('(e.g.,
    firstorder, shape, glcm)'). Patterns, extensions and numbers are not values here."""
    desc = prop.get("description") or ""
    vals = _QUOTED_VALUE.findall(desc)
    for m in list(_LISTED.finditer(desc)) + (list(_EXAMPLE_LIST.finditer(desc)) if examples else []):
        vals += [x.strip() for x in re.split(r",\s*(?:or\s+)?", m.group(1)) if x.strip()]
    return C._dedup(vals)


def closed_values(prop):
    """The values of a free-text argument whose description lists a closed set of them ("Strategy: 'full', 'partial',
    'head_only'", 'ResNet model type: resnet18, ..., or resnet152'): at least two documented values, and the description
    neither gives them as examples (e.g., such as, etc.) nor accepts other names (natural language). [] otherwise."""
    desc = prop.get("description") or ""
    vals = documented_values(prop)
    return vals if len(vals) >= 2 and not _OPEN_LIST.search(desc) and not _FLEXIBLE.search(desc) else []


_DEFAULT_CLAUSE = re.compile(r"\(?\bdefaults?\b[^).]*\)?", re.I)


def closed_set(prop):
    """The description lists a closed set of values (closed_values), and not only as its default ("Key for images
    (default: 'images' or 'test_images')" names defaults; any key is allowed)."""
    rest = {**prop, "description": _DEFAULT_CLAUSE.sub(" ", prop.get("description") or "")}
    return bool(closed_values(prop)) and bool(closed_values(rest))


def named_values(prop, request, examples=False, plural=False):
    """The documented values the request names (paths left out), compared as squashed words: 'head only' names
    head_only, 'ResNet-50' names resnet50, 'random forest' names 'Random Forest'. examples: see documented_values.
    plural: a plural in the request names a singular value ('protein complexes' -> 'PROTEIN COMPLEX')."""
    text = request
    for p in C.paths(request):
        text = text.replace(p, " ")
    toks = re.findall(r"[A-Za-z0-9]+", text)
    spans = {_squash("".join(toks[i:i + n])) for n in range(1, 5) for i in range(len(toks) - n + 1)}
    if plural:
        spans |= {s[:-k] for s in spans for k in (1, 2) if len(s) > 3 and s.endswith("es"[2 - k:])}
    return [v for v in documented_values(prop, examples) if _squash(v) in spans]


_COUNT_NAME = re.compile(r"num_?classes|n_?classes|num_?categories|num_?labels")
_COUNT_DESC = re.compile(r"\bnumber of (?:\w+ ){0,2}(?:classes|categories|labels)\b", re.I)
_ADDED_DESC = re.compile(r"\bafter adding\b|\btotal number of classes\b", re.I)


def is_count_arg(name, prop):
    """An integer argument for a number of classes: its name says num_classes (new_num_classes, n_classes) or its
    description 'number of (output) classes'."""
    return _type(prop) == "integer" and bool(_COUNT_NAME.search(name) or _COUNT_DESC.search(prop.get("description") or ""))


def acronym_values(prop, request):
    """Documented values (examples included) written as the phrase their letters start: 'gray level co-occurrence
    matrix' -> glcm, 'Laplacian of Gaussian' -> LoG. Letters only, two to six; a hyphenated word gives its first letter
    or the first letters of its parts ('grey-level run-length matrix' -> glrlm)."""
    text = request
    for p in C.paths(request):
        text = text.replace(p, " ")
    toks = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    forms = [{w[0].lower(), "".join(x[0].lower() for x in w.split("-") if x)} for w in toks]
    out = []
    for v in documented_values(prop, examples=True):
        k = v.lower()
        if not re.fullmatch(r"[a-z]{2,6}", k):
            continue
        for i in range(len(toks)):                      # each word gives its first letter, a hyphenated one also its parts'
            reach = {""}
            for f in forms[i:i + len(k)]:
                reach = {r + x for r in reach for x in f if k.startswith(r + x)}
                if k in reach or not reach:
                    break
            if k in reach:
                out.append(v); break
    return out


def is_file_arg(name, prop):
    """A string argument that holds a file: its name ends in csv/tsv/filename or its description says filename."""
    return bool(_FILE_NAME.search(name) or _FILE_DESC.search((prop.get("description") or "").strip()))


_PATTERN_ARG = re.compile(r"pattern|regex|glob|wildcard", re.I)
_DIR_NAME = re.compile(r"(^|_)(dir|directory|folder|root)($|_)")
_DIR_DESC = re.compile(r"\b(directory|folder)\b", re.I)
_FILE_WORD = re.compile(r"\bfiles?\b|\.[a-z]{2,5}\b", re.I)


def takes_files(prop):
    """The description of a path argument also says it takes a file ('NIfTI file or DICOM directory')."""
    return bool(_FILE_WORD.search(prop.get("description") or ""))


def is_dir_arg(name, prop):
    """A path argument that can hold a folder: its name or its description says directory, folder or root."""
    return bool(_DIR_NAME.search(name) or _DIR_DESC.search(prop.get("description") or ""))


_NAME_ARG = re.compile(r"file_?name", re.I)
_NAME_DESC = re.compile(r"\bfile ?name$", re.I)


def is_name_arg(name, prop):
    """An argument that takes a file name, not a path: its name says filename, or the first sentence of its description
    ends in 'filename' ('Output prediction filename. Default: ...'); the server joins it with the output folder."""
    first = re.split(r"(?<=[.;])\s|\(", prop.get("description") or "", maxsplit=1)[0].strip().rstrip(".")
    return bool(_NAME_ARG.search(name) or _NAME_DESC.search(first))


def short(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def _words_of(v):
    """'lung_vessels' -> 'lung vessels' (compares enum values with request phrases)."""
    return re.sub(r"[_\-]+", " ", str(v)).strip().lower()


def object_keys(prop, request):
    """Keys an object argument can take: its declared properties, else the keys of the example object in its description,
    plus CamelCase field names written in the request (e.g. PatientID)."""
    if prop.get("properties"):
        return list(prop["properties"])
    keys = _EXAMPLE_KEY.findall(prop.get("description") or "")
    keys += [k for k in _CAMEL.findall(request) if k not in keys]
    return C._dedup(keys)


_SAME_PLACE = re.compile(r"\b(?:the\s+)?same\s+(?:folder|directory|dir|place|location|sub-?folder)\b", re.I)


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_folder(v):
    """A written absolute folder (no file extension)."""
    return isinstance(v, str) and v.startswith(("/", "~")) and not re.search(r"\.[A-Za-z][A-Za-z0-9]{0,5}$", v)


class _Uses:
    """The values the arguments have taken, used like a set: a value is "in" once it has been taken as many times as
    `count(value)` allows (the times the request writes it, plus one when the request says it is shared). Numbers count
    by value: '2' and '2.0' are one number."""

    def __init__(self, count):
        self.n, self.count = {}, count

    @staticmethod
    def _key(v):
        try:
            return ("number", float(v))
        except (TypeError, ValueError):
            return ("text", v)

    def __contains__(self, v):
        return self.n.get(self._key(v), 0) >= self.count(v)

    def add(self, v):
        k = self._key(v)
        self.n[k] = self.n.get(k, 0) + 1

    def update(self, vs):
        for v in vs:
            self.add(v)


class _Barred:
    """The values one argument may not take: those taken, and those the model gave to another argument."""

    def __init__(self, used, barred):
        self.used, self.barred = used, barred

    def __contains__(self, v):
        return v in self.barred or v in self.used


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


class ArgumentFiller:
    """Fills one tool's arguments for one request by asking the backend (a decision model) typed questions.

    policy_defaults: values for arguments the request leaves open, e.g. {"output_dir": "{workspace}/{tool}_{timestamp}"}
    (filled in after the questions; never shown to the model). optional_min_conf: the bar for optional values and
    objects. list_min_conf: the bar for a list's first item (default: optional_min_conf). required_min_conf: a required
    value below it is reported as uncertain (the interactive agent asks the user to confirm it)."""

    def __init__(self, backend, policy_defaults=None, workspace="runs/workspace", required_min_conf=0.5,
                 optional_min_conf=0.7, max_items=30, list_min_conf=None):
        self.b = backend
        self.policy = policy_defaults or {}
        self.workspace = workspace
        self.required_min_conf = required_min_conf
        self.optional_min_conf = optional_min_conf
        self.list_min_conf = optional_min_conf if list_min_conf is None else list_min_conf
        self.max_items = max_items
        self.NS = NOT_SPECIFIED
        self.UNM = UNMENTIONED

    # ---- helpers -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _props(tool):
        """The tool's argument schemas as the questions read them (unions read as one type)."""
        props = (tool.input_schema or {}).get("properties", {}) or {}
        return {n: flat_schema(p) for n, p in props.items()}

    @staticmethod
    def _state(request, tool, context):
        s = f"User request: {request}\nTool: {tool.name}: {short(tool.description, 400)}"
        return s + (f"\nEarlier steps:\n{context}" if context else "")

    def _policy_value(self, name, tool):
        v = self.policy.get(name)
        if v is None:
            return None
        return v.format(workspace=self.workspace, tool=tool.name, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))

    def _resolve(self, d, used=()):
        """A value question's answer under the one-value-one-argument rule: the best option not used yet.
        -> (value or the not-specified option, its probability, probability mass on the allowed values)."""
        allowed = [i for i, o in enumerate(d.options) if o not in used]
        best = max(allowed, key=lambda i: d.probs[i])
        mass = sum(d.probs[i] for i in allowed if d.options[i] != self.NS)
        return d.options[best], d.probs[best], mass

    @staticmethod
    def _dicts(source):
        """The dictionaries the request writes: literals and key-number runs, prose, 'A x, y; B z' groups and groups
        written one per line."""
        ds = C.dicts(source)
        keys = {repr(sorted(d.items(), key=str)) for d in ds}
        ds += [d for d in C.prose_dicts(source) if repr(sorted(d.items(), key=str)) not in keys]
        ds += [d for d in C.grouped_lists(source) if d not in ds]
        ds += [d for d in C.grouped_lines(source) if d not in ds]
        return ds

    @staticmethod
    def _written_arrays(prop, source):
        """For an array whose items are arrays (or untyped items, when the description shows a tuple): the lists of
        lists the request writes, bracketed lists in sequence and lines of number pairs. For an array whose items are
        objects: the JSON arrays of objects the request writes."""
        items = prop.get("items") if isinstance(prop.get("items"), dict) else {}
        untyped = "items" in prop and _type(items) is None and not items.get("properties") \
            and bool(_TUPLE_DESC.search(prop.get("description") or ""))
        out = []
        if _type(items) == "array" or untyped:
            out += C.written_arrays(source)
            out += [x for x in C.bracket_lists(source) + C.number_rows(source) if x not in out]
        elif _type(items) == "object":
            out += C.object_arrays(source)
        return out

    @staticmethod
    def _value_question(name, desc):
        return f"Which value should the argument `{name}` take? `{name}`: {desc}"

    @staticmethod
    def _gate_question(name, desc):
        return f"Does the request say which items the argument `{name}` should contain? `{name}`: {desc}"

    @staticmethod
    def _enum_options(prop):
        """Option text per enum value (the value as written) -> {option: value}."""
        return {str(v): v for v in prop["enum"]}

    @staticmethod
    def _item_question(name, prop, chosen=(), what="item"):
        desc = short(prop.get("description"), 240)
        return (f"Which {what} does the request want in `{name}`"
                f"{' next' if chosen else ''}? `{name}`: {desc}"
                + (f" Already chosen: {', '.join(chosen)}." if chosen else ""))

    @staticmethod
    def _without(remaining, pick):
        """A chosen phrase also takes the words it is made of, and a chosen word the phrases containing it."""
        parts = set(pick.split())
        return [c for c in remaining if c != pick and c not in parts and not (set(c.split()) & parts)]

    def _after_pick(self, remaining, pick, chosen, text, consumed):
        """The candidates still offered after `pick` was chosen (chosen includes it). An item written in the request is
        removed only when every place it is written overlaps the place of a chosen item ('no tumor' leaves the 'tumor'
        written after it, 'right kidney' leaves 'left kidney'); items not written literally follow _without. A chosen
        item stays while the request writes it more often than it was chosen (a one-character mark such as '+' once
        more when the request says 'both' or 'each'). text None (an object's keys): _without."""
        if text is None:
            return self._without(remaining, pick)
        places = [sp for sp in C.spans(text, pick) if not any(_overlap(sp, t) for t in consumed)]
        longer = [sp for c in remaining if len(c) > len(pick) for sp in C.spans(text, c)]
        alone = [sp for sp in places if not any(t[0] <= sp[0] and sp[1] <= t[1] for t in longer)]
        if places:
            consumed.append((alone or places)[0])
        old = set(self._without(remaining, pick))

        def keep(c):
            where = C.spans(text, c)
            if not where:
                return c in old
            return any(not any(_overlap(sp, t) for t in consumed) for sp in where)
        out = {c for c in remaining if c != pick and keep(c)}
        extra = 1 if len(pick) == 1 and C.SHARED.search(text) else 0
        if chosen.count(pick) < C.occurrences(text, pick) + extra:
            out.add(pick)
        return [c for c in remaining if c in out]

    def _items(self, state, name, prop, cands, tag, what="item", chosen=(), text=None):
        """Items picked one at a time until "no further item" (or "none") is the more likely answer: -> list (may be
        empty). chosen: items already decided (a list's first item), the questions go on from there. text: the request,
        for _after_pick."""
        chosen, remaining, consumed = list(chosen), list(cands), []
        for i, pick in enumerate(chosen):
            remaining = self._after_pick(remaining, pick, chosen[:i + 1], text, consumed)
        while remaining and len(chosen) < self.max_items:
            stop = NO_MORE if chosen else NO_ITEM
            opts = remaining[:254] + [stop]
            d = self.b.choice(state, self._item_question(name, prop, chosen, what), opts, tag=tag + ":item")
            if d.probs[-1] >= 0.5:
                break
            best = max(range(len(d.probs) - 1), key=d.probs.__getitem__)
            pick = remaining[best]
            chosen.append(pick)
            remaining = self._after_pick(remaining, pick, chosen, text, consumed)
        return chosen

    @staticmethod
    def _list_candidates(name, prop, kind, request, source, cols, stems):
        """Items a list or array argument can take, before the values other arguments use are taken out: the enum of
        its items; numbers (and the numbers of written ranges) for numeric items; otherwise the documented values the
        request names (also by acronym) and a closed set of documented values, quoted strings, written list items,
        identifiers, dotted names, compact tokens, strand marks, search chunks for search arguments, the CSV's columns
        the request writes, content phrases and numbers."""
        items = prop.get("items", {}) if kind == "array" else {}
        if "enum" in items:
            cands = [str(v) for v in items["enum"]]
        elif _type(items) in ("integer", "number"):
            cands = C.numbers(source)
            cands += C.range_items(source)
            cands += C.number_groups(source)
            cands += C.number_scales(source)
        else:
            desc = prop.get("description") or ""
            cands = C.quoted(source) + C.written_items(source) + C.identifiers(request)
            cands += C.dotted_names(source)
            cands = named_values(prop, source, examples=True) + cands
            cands = acronym_values(prop, source) + cands
            cands += C.compact_ids(source)
            cands = closed_values(prop) + cands
            if _SEARCH_DESC.search(desc):
                cands += C.possessive_phrases(source)
            cands += C.strand_marks(source)
            if _SEARCH_WORDS.search(desc):
                cands += C.search_chunks(source)
            cands += [c for c in cols if c.lower() in source.lower()]
            cands += [ph for ph in C.phrases(request) if not all(_stem(w.lower()) in stems for w in ph.split())]
            cands += C.numbers(source)
            cands += C.range_items(source)
        cands = C._dedup(cands, strip=False)
        seen = set()                                    # 'Square Root' and 'square root' are one item
        return [c for c in cands if not (c.lower() in seen or seen.add(c.lower()))]

    @staticmethod
    def _coerce(value, prop, enum_map=None):
        t = _type(prop) or ("string" if "enum" in prop else None)
        if "enum" in prop:
            if enum_map and value in enum_map:
                return enum_map[value]
            return next((e for e in prop["enum"] if str(e) == value), value)
        if t == "integer":
            f = float(value)
            return int(f) if f.is_integer() else None
        if t == "number":
            return float(value)
        return value

    @staticmethod
    def _number_candidates(source, name, prop):
        """The numbers a number argument is offered: written numbers and those the request states in another form the
        argument takes (candidates.py: derived numbers, class counts, durations and other units, spans and lengths in
        bp for integers, ages, charges, levels, fractions, complements)."""
        desc = prop.get("description") or ""
        integer = _type(prop) == "integer"
        cands = C.numbers(source)
        cands += C.derived_numbers(source, name, prop)
        if is_count_arg(name, prop):
            cands += C.class_counts(source, total=bool(_ADDED_DESC.search(desc)))
        cands += C.durations(source, name, prop)
        cands += C.number_groups(source, name, prop)
        if integer:
            cands += C.span_ends(source)
            cands += C.length_units(source)
            cands += C.age_notation(source)
        cands += C.number_scales(source, name, prop)
        cands += C.milli_units(source)[0]
        if integer:
            cands += C.charge_forms(source)
        if C._FRACTION_ARG.search(desc):
            cands += C.split_fractions(source)
        cands += C.complement_levels(source, prop)
        if integer:
            cands += C.level_digits(source, name, prop)
        return cands

    @staticmethod
    def _path_candidates(source, name, prop):
        """The paths a path argument is offered: written paths (quoted paths with spaces whole), the folders the request
        names without writing them (folder arguments only), paths joined from written parts (a folder argument that
        takes no file is offered no joined file), dotted names when the description shows them, and for arguments that
        can take a file: bare data-file names and outputs named after a written file."""
        desc = prop.get("description") or ""
        folder_only = is_dir_arg(name, prop) and not takes_files(prop)
        cands = C.paths(source)
        spaced = C.quoted_paths(source)
        cands = spaced + [c for c in cands if not any(c in sp for sp in spaced)]
        spaced2 = [c for c in C.spaced_paths(source) if c not in cands]
        cands = spaced2 + [c for c in cands if not any(c in sp for sp in spaced2)]
        if is_dir_arg(name, prop):
            cands += C.stated_folders(source, list(cands))
        joined = C.joined_paths(source)
        if folder_only:
            joined = [j for j in joined if not C._HAS_EXTENSION.search(os.path.basename(j))]
        cands += joined
        if C.dotted_names(desc):
            cands += C.dotted_names(source)
        if not folder_only:
            cands += C.bare_files(source)
            cands += C.renamed_outputs(source, bool(_SAME_PLACE.search(source)))
            cands += C.renamed_outputs_ext(source, C.spaced_paths(source))
        return cands

    def _free_candidates(self, kind, source, csv_cols, prop, stems=frozenset(), name="", own=frozenset()):
        """The options of a number, path, text or column argument (before "not specified" is added), at most 254."""
        desc = prop.get("description") or ""
        if kind == "number":
            cands = self._number_candidates(source, name, prop)
        elif kind == "text" and closed_set(prop):
            return closed_values(prop)[:254]                    # a closed set of documented values is offered alone
        else:
            cands = self._path_candidates(source, name, prop) if kind == "path" else []
            if kind == "column":
                cands += csv_cols
                if not _COLUMN_NAME.search(name):               # routed to columns by a name word (target, label, ...)
                    undefaulted = {**prop, "description": _DEFAULT_CLAUSE.sub(" ", desc)}
                    cands = named_values(undefaulted, source, plural=True) + cands
            if kind != "path":
                cands += C.quoted(source) + C.identifiers(source)
                cands += C.dotted_names(source)
                cands += C.compact_ids(source)
                if _PREFIX_NAME.search(name):                   # an output prefix is often a path
                    cands = C.paths(source) + cands
            if kind in ("text", "column"):
                words1 = C.phrases(source, max_n=1, limit=200)  # single words, not words of the argument or tool names
                cands += [w for w in words1 if _stem(w.lower()) not in stems and _squash(w) not in own]
                cands += C.extensions(source)
                cands += C.named_words(source)
            search = bool(_SEARCH_DESC.search(desc))
            if kind in ("text", "column") and search:
                cands = C.boolean_queries(source) + cands
            if kind == "column" and search:
                cands += C.phrases(source, max_n=3, limit=200)
            if kind in ("text", "column") and _SEARCH_WORDS.search(desc):
                cands = C.bracket_queries(source) + cands + C.search_chunks(source)
            if kind == "text":
                cands += C.alleles(source)
                cands += C.colon_blocks(source, desc)
                if _SEARCH_DESC.search(desc):
                    cands += C.possessive_phrases(source)
                cands += C.compound_parts(source)
                cands += C.chromosome_names(source)
                cands += C.variant_alleles(source)
                cands += C.strand_marks(source)
                cands += C.joined_tokens(source)
                cands += C.milli_units(source)[1]
                cands += C.unit_words(source)
                cands += C.numbers(source)                      # string-typed numbers, e.g. an nnU-Net fold "0"
                cands = closed_values(prop) + cands
                cands = named_values(prop, source, plural=True) + cands
                if _FLEXIBLE.search(desc) or _SEARCH_DESC.search(desc):
                    cands += C.phrases(source, max_n=3, limit=200)
                mine = {_stem(w) for w in name.lower().split("_") if w}     # text written after a label naming it
                cands = [v for label, v in C.labelled_text(source)
                         if (ws := {_stem(w) for w in re.split(r"[\s_-]+", label.lower()) if w})
                         and (ws <= mine or mine <= ws)] + cands
                if _PATTERN_ARG.search(name) or _PATTERN_ARG.search(desc):
                    which = "glob" if re.search(r"glob|wildcard", name + " " + desc, re.I) else \
                        "regex" if re.search(r"regex|regular expression", name + " " + desc, re.I) else None
                    cands = C.patterns(source, which) + cands
        cands = C._dedup(cands, strip=False)
        if is_name_arg(name, prop):                             # a file name has no folder
            cands = [c for c in cands if "/" not in c and "\\" not in c]
        seen = set()                                            # 'Random Forest' and 'random forest' are one option
        cands = [c for c in cands if not (c.lower() in seen or seen.add(c.lower()))]
        return cands[:254]

    # ---- main ----------------------------------------------------------------------------------------------------------
    def kinds(self, tool):
        """{argument: kind}: param_kind, then string arguments named or described as a file are paths, pattern arguments
        are text, and an argument described as "Like <other argument>" is read as that argument's kind."""
        props = self._props(tool)
        kinds = {n: param_kind(n, props[n]) for n in props}
        kinds.update({n: "path" for n, k in kinds.items() if k in ("text", "column") and is_file_arg(n, props[n])})
        kinds.update({n: "text" for n, k in kinds.items() if k in ("path", "column") and _PATTERN_ARG.search(n)})
        for n in props:
            m = _LIKE.match(props[n].get("description") or "")
            if m and m.group(1) in props and m.group(1) != n and kinds[n] == "text":
                kinds[n] = kinds[m.group(1)]
        return kinds

    def questions(self, request, tool, context=""):
        """Every question that does not depend on another answer: one per scalar argument, the first item question of
        each list, one question or gate per object. -> list of (argument name, question spec or None, kind) in settling
        order: required arguments first, paths first, then lists, arrays and objects."""
        props = self._props(tool)
        required = set((tool.input_schema or {}).get("required", []) or [])
        state = self._state(request, tool, context)
        source = request + ("\n" + context if context else "")
        kinds = self.kinds(tool)
        csv_cols = [c for pth in C.paths(source) for c in C.csv_columns(pth)]
        stems = {_stem(w) for n in props for w in n.split("_")}
        own = {_squash(w) for w in re.split(r"[-_\s]+", f"{tool.server} {tool.name}") if w}   # words naming the tool
        later = ("list", "array", "object")
        order = sorted((n for n in props if kinds[n] not in later), key=lambda n: (n not in required, kinds[n] != "path"))
        out = []
        for name in order + [n for n in props if kinds[n] in later]:
            prop, optional, kind = props[name], name not in required, kinds[name]
            tag, desc = f"arg:{tool.name}.{name}", short(prop.get("description"), 240)
            spec = None
            if kind == "enum" and not (not optional and len(prop["enum"]) == 1):
                spec = dict(question=self._value_question(name, desc),
                            options=list(self._enum_options(prop)) + ([self.NS] if optional else []))
            elif kind == "bool":
                spec = dict(question=f"Does the request turn the option `{name}` on or off? `{name}`: {desc}",
                            options=[ON, OFF] + ([self.UNM] if optional else []))
            elif kind in ("number", "path", "text", "column"):
                cands = self._free_candidates(kind, source, csv_cols, prop, stems, name, own)
                if cands:
                    spec = dict(question=self._value_question(name, desc), options=cands + [self.NS])
            elif kind == "array" and (arrs := self._written_arrays(prop, source)):
                arrays = {json.dumps(x): x for x in arrs[:254]}
                spec = dict(question=self._value_question(name, desc), options=list(arrays) + [self.NS], arrays=arrays)
            elif kind in ("list", "array"):
                cands = self._list_candidates(name, prop, kind, request, source, csv_cols, stems)
                if cands:
                    spec = dict(question=self._item_question(name, prop), options=cands[:254] + [NO_ITEM],
                                tag=tag + ":item")
            elif kind == "object" and (ds := self._dicts(source)):
                dicts = {json.dumps(x): x for x in ds[:254]}
                spec = dict(question=self._value_question(name, desc), options=list(dicts) + [self.NS], dicts=dicts)
            elif kind == "object":
                spec = dict(question=self._gate_question(name, desc), options=["no", "yes"], kind="noul",
                            tag=tag + ":gate")
            if spec:
                spec = {"state": state, "kind": "choice", "tag": tag, **spec}
            out.append((name, spec, kind))
        return out

    def fill(self, request, tool, context=""):
        """Ask the independent questions together (one choice_batch), settle them, then pick list items and object
        fields. -> Filled."""
        schema = tool.input_schema or {}
        props = self._props(tool)
        required = set(schema.get("required", []) or [])
        state = self._state(request, tool, context)
        source = request + ("\n" + context if context else "")
        calls0 = len(self.b.trace)
        args, missing, uncertain, decisions = {}, [], [], []
        same = bool(_SAME_PLACE.search(source))            # 'in the same folder': a written folder may fill one more
        each = bool(C.EACH_WAY.search(source))             # 'each way', 'for both': a number may fill one more
        shared = bool(C.SHARED.search(source))             # 'both', 'for each': any value may fill one more
        chroms = C.chromosome_counts(source)
        used = _Uses(lambda v: (max(C.occurrences(source, v), chroms.get(v, 0))
                                + (1 if same and _is_folder(v) else 0)
                                + (1 if (each and _is_number(v)) or shared else 0)))
        barred = {}                                 # argument -> values the model gave to another argument
        done = set()                                # object and array arguments already settled as one value
        chosen_words = set()                        # the enum values already chosen, as words

        def settle(ad, optional, value, conf, specified):
            """Apply the optional-argument rules to a value the model picked; `specified` is P(the argument is given)."""
            ad.value, ad.confidence = value, conf
            if optional and specified < self.optional_min_conf:
                ad.status = "low_conf"
            elif optional and is_default(value, props[ad.name]):
                ad.status = "is_default"
            else:
                ad.status = "set"

        plan = self.questions(request, tool, context)
        free = ("number", "path", "text", "column")
        later = ("list", "array", "object")
        spec = {name: sp for name, sp, _ in plan}
        kind_of = {name: k for name, _, k in plan}
        todo = [name for name, _, _ in plan]
        enum_maps = {n: self._enum_options(props[n]) for n in props if kind_of.get(n) == "enum"}

        def used_by(n):
            """The values n may not take: those taken, and those the model gave to another argument."""
            return _Barred(used, barred[n]) if barred.get(n) else used

        def settle_one(name):
            kind, prop, optional = kind_of[name], props[name], name not in required
            ad = ArgDecision(name, kind, "default")
            d = answer.get(name)
            if kind == "object" and d is not None and "dicts" in spec[name]:     # a dictionary the request writes
                ad.options = d.options
                choice, conf, mass = self._resolve(d)
                if choice != self.NS and (not optional or mass >= self.optional_min_conf):
                    ad.status, ad.value, ad.confidence = "set", spec[name]["dicts"][choice], mass
                self._record(ad, optional, tool, args, missing, uncertain, decisions)
                done.add(name)
                return
            if kind == "array" and d is not None and "arrays" in (spec[name] or {}):   # a list of lists the request writes
                ad.options = d.options
                choice, conf, mass = self._resolve(d, used_by(name))
                if choice != self.NS and (not optional or mass >= self.list_min_conf):
                    ad.status, ad.value, ad.confidence = "set", spec[name]["arrays"][choice], mass
                    used.add(choice)
                self._record(ad, optional, tool, args, missing, uncertain, decisions)
                done.add(name)
                return
            if kind in later:                               # recorded after the items are picked
                if kind == "object" and d is not None and d.probs[1] >= self.optional_min_conf:
                    gated.append(name)
                return
            if kind == "enum" and not optional and len(prop["enum"]) == 1:
                ad.status, ad.value = "set", prop["enum"][0]
            elif d is not None and kind == "enum":
                ad.options = d.options
                choice, conf, mass = self._resolve(d)
                if choice != self.NS:
                    settle(ad, optional, self._coerce(choice, prop, enum_maps.get(name)), conf, mass)
                    if ad.status == "set":
                        chosen_words.add(_words_of(ad.value))
            elif d is not None and kind == "bool":
                ad.options = d.options
                if d.choice != self.UNM:
                    settle(ad, optional, d.choice == ON, d.confidence, d.confidence)
            elif d is not None and any(o not in used_by(name) for o in d.options[:-1]):
                ad.options = d.options
                choice, conf, mass = self._resolve(d, used_by(name))
                value = self._coerce(choice, prop) if choice != self.NS else None
                if value is not None:
                    settle(ad, optional, value, conf, mass)
                    if ad.status == "set":
                        used.add(choice)
            elif kind == "unsupported":
                ad.status = "unsupported"
            self._record(ad, optional, tool, args, missing, uncertain, decisions)

        def strongest(names):
            """The free-valued argument whose best still-unused value is the most probable."""
            def best(n):
                d = answer.get(n)
                if d is None:
                    return -1.0
                allowed = [p for o, p in zip(d.options, d.probs) if o not in used_by(n) and o != self.NS]
                return max(allowed, default=-1.0)
            return max(names, key=best)

        # Every independent question goes out at once. Enum, boolean and gate questions settle first, then the free
        # values strongest first, so a stated value goes to the argument that wants it most, not to the one listed first.
        answer, gated = {}, []
        batch = [n for n in todo if spec[n] is not None]
        answer.update(zip(batch, self.b.choice_batch([spec[n] for n in batch])))
        barred.update(self._owners(state, tool, props, required, answer, kind_of, used))
        for n in [n for n in todo if kind_of[n] not in free]:
            settle_one(n)
        rest = [n for n in todo if kind_of[n] in free]
        while rest:
            n = strongest(rest)
            rest.remove(n)
            settle_one(n)

        # Lists, arrays and objects. A list's first item question was answered with the scalar questions; the list is
        # filled when the mass on the items still allowed reaches list_min_conf (any mass for a required list), and lists
        # settle strongest first, so an item two lists want goes to the list whose first answer wants it most. An item
        # another argument has taken is not offered again.
        stems = {_stem(w) for n in props for w in n.split("_")}
        cols = [c for pth in C.paths(source) for c in C.csv_columns(pth)]

        def allowed(c, n=None):
            """An item no other argument has taken, nor (as words) an enum value already chosen; strand marks ('+', '.')
            are not words. With n: nor a value the model gave to another argument."""
            mark = not re.search(r"[A-Za-z0-9]", c)
            return not (c in (used_by(n) if n else used) or (_words_of(c) in chosen_words and not mark))

        def first_answer(n):
            """(best allowed option, its probability, mass on the allowed items) of n's first item question."""
            d = answer.get(n)
            if kind_of[n] == "object" or d is None:
                return None
            ok = [i for i, o in enumerate(d.options) if o == NO_ITEM or allowed(o, n)]
            best = max(ok, key=d.probs.__getitem__)
            return d.options[best], d.probs[best], sum(d.probs[i] for i in ok if d.options[i] != NO_ITEM)

        def strength(n):
            fa = first_answer(n)
            return -1.0 if fa is None or fa[0] == NO_ITEM else fa[1]

        rest = [name for name, _, kind in plan if kind in later]
        while rest:
            name = max(rest, key=strength)
            rest.remove(name)
            if name in done:
                continue
            kind, prop, tag = kind_of[name], props[name], f"arg:{tool.name}.{name}"
            optional = name not in required
            items = prop.get("items", {}) if kind == "array" else {}
            ad = ArgDecision(name, kind, "default")
            chosen = []
            if name in gated:                                        # an object whose gate said yes
                value = self._object(state, name, prop, request, source, used, tag)
                if value:
                    ad.status, ad.value = "set", value
            elif (fa := first_answer(name)) is not None:
                pick, _, mass = fa
                ad.confidence = mass
                if pick != NO_ITEM and (not optional or mass >= self.list_min_conf):
                    cands = [c for c in self._list_candidates(name, prop, kind, request, source, cols, stems)
                             if allowed(c, name)]
                    chosen = self._items(state, name, prop, cands, tag, chosen=[pick], text=source)
            elif answer.get(name) is not None:
                ad.confidence = answer[name].probs[1]                # the gate said no: its P(yes), for failure analysis
            if chosen:
                used.update(chosen)
                ad.status = "set"
                ad.value = [self._coerce(c, items) for c in chosen] if kind == "array" else ", ".join(chosen)
            self._record(ad, optional, tool, args, missing, uncertain, decisions)

        errors = []
        if not missing:
            v = jsonschema.Draft202012Validator(schema)
            errors = [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in v.iter_errors(args)]
        return Filled(args, missing, uncertain, decisions, errors, model_calls=len(self.b.trace) - calls0)

    def _owners(self, state, tool, props, required, answer, kind_of, used):
        """Values the first answers of more arguments take than the request allows (each answer enough to set its
        argument: a required one, or the mass on its allowed values reaches the bar). The model is asked which argument
        the value belongs to; when that answer reaches optional_min_conf, the other arguments may not take the value.
        -> {argument: {values}}."""
        claims = {}
        for n, d in answer.items():
            if d is None:
                continue
            kind, optional = kind_of[n], n not in required
            if kind in ("number", "path", "text", "column"):
                choice, _, mass = self._resolve(d, used)
                if choice != self.NS and (not optional or mass >= self.optional_min_conf):
                    claims.setdefault(choice, []).append(n)
            elif kind in ("list", "array") and d.options[-1] == NO_ITEM:      # a first item question
                ok = [i for i, o in enumerate(d.options) if o == NO_ITEM or o not in used]
                best = max(ok, key=d.probs.__getitem__)
                mass = sum(d.probs[i] for i in ok if d.options[i] != NO_ITEM)
                if d.options[best] != NO_ITEM and (not optional or mass >= self.list_min_conf):
                    claims.setdefault(d.options[best], []).append(n)
        times = used.count
        contested = [(v, ns) for v, ns in claims.items() if len(ns) > times(v)]
        if not contested:
            return {}
        specs = [dict(state=state, question=f"The request gives the value '{v}'. Which argument does it belong to?",
                      options=[f"`{n}`: {short(props[n].get('description'), 160)}" for n in ns], kind="choice",
                      tag=f"arg:{tool.name}:owner") for v, ns in contested]
        out = {}
        for (v, ns), d in zip(contested, self.b.choice_batch(specs)):
            ranked = sorted(range(len(ns)), key=lambda i: -d.probs[i])
            if sum(d.probs[i] for i in ranked[:times(v)]) < self.optional_min_conf:
                continue                                        # an unsure answer does not decide
            for i in ranked[times(v):]:
                out.setdefault(ns[i], set()).add(v)
        return out

    def _object(self, state, name, prop, request, source, used, tag):
        """An object argument whose gate said yes. Keys are picked like list items, then each key gets a value question
        over the request's candidates (quoted text, identifiers, numbers, words, proper names, documented formats).
        -> {key: value} with the keys whose value reached optional_min_conf, or {}."""
        keys = self._items(state, name, prop, object_keys(prop, request), tag + ":keys", what="field")
        desc = short(prop.get("description"), 240)
        cands = (C.quoted(source) + C.identifiers(source) + C.plain_numbers(source)
                 + C.formatted(source, prop.get("description", "")) + C.phrases(request, max_n=1))
        cands += C.proper_names(source)
        cands = [c for c in C._dedup(cands, strip=False) if c not in used and c not in keys][:254]
        out = {}
        for key in keys:
            if not cands:
                break
            d = self.b.choice(state, f"Which value does the request give for the field `{key}` of `{name}`? `{name}`: {desc}",
                              cands + [self.NS], tag=f"{tag}:value:{key}")
            choice, conf, mass = self._resolve(d, used)
            if choice != self.NS and mass >= self.optional_min_conf:
                sub = (prop.get("properties") or {}).get(key, {})
                out[key] = self._coerce(choice, sub) if sub else choice
                used.add(choice)
        return out

    def _record(self, ad, optional, tool, args, missing, uncertain, decisions):
        if ad.status != "set" and (pv := self._policy_value(ad.name, tool)) is not None:
            ad.status, ad.value = "policy", pv
        if ad.status in ("set", "policy"):
            args[ad.name] = ad.value
            if not optional and ad.status == "set" and ad.confidence is not None and ad.confidence < self.required_min_conf:
                uncertain.append(ad.name)
        elif not optional:
            ad.status = "missing"; missing.append(ad.name)
        decisions.append(ad)
