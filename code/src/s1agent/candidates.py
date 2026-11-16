"""Candidate values for free-valued arguments, read from the request.

A decision model cannot write a value, only pick one, so every argument whose value is not a closed set (an enum or a
boolean) needs a finite list of candidates. This module finds them in the request text (and the text of earlier tool
results): paths, numbers, quoted strings, identifiers, content phrases, list items, dictionaries, and values the request
states in another form than the argument takes (a number in words, a duration in minutes for an argument in seconds, a
chromosome written without 'chr', a folder named but not written out). It also reads the header of a CSV file another
argument points to, so that column arguments can be offered the file's columns.

Every function here only proposes; the compiler decides which candidates an argument is offered (s1agent.compiler), and
the model picks among them. Nothing is invented: each candidate is built from words, numbers or paths the request
writes. The functions return candidates in the order they appear (deduplicated), which keeps the option lists stable.
"""
import ast
import csv
import json
import os
import re

from s1agent.backends.lexical import _STOP

_PATH = re.compile(r"""(?<![\w])((?:~|\.{1,2})?/[^\s'"`,;()]+|[\w.-]+/[^\s'"`,;()]+|[\w-]+\.(?:csv|tsv|xlsx?|json|ya?ml|txt|nii(?:\.gz)?|nrrd|mha|dcm|pkl|pth|pt|png|jpe?g))""")
_NUMBER = re.compile(r"(?<![\w./])(-?\d+(?:\.\d+)?)(?=$|[\s,;:)%x×-]|\.(?:\s|$))")
# A quote does not open right after a letter or digit, nor close right before one, so the apostrophes of "I'm" and
# "model's" are not quotes.
_QUOTED = re.compile(r"""(?<![A-Za-z0-9])['"`“‘]([^'"`”’]{1,200})['"`”’](?![A-Za-z0-9])""")
_PATH_START = re.compile(r"(?:~|\.{1,2})?/|[\w.-]+/")
_IDENT = re.compile(r"(?<![\w/.])([A-Za-z][\w.-]*[\d_][\w.-]*|[\w-]*\d[\w-]*[A-Za-z][\w-]*)(?![\w/])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+-]*")


def _dedup(xs, strip=True):
    """Unique values in order. strip: also drop trailing punctuation (sentence ends after paths and words); quoted text
    is passed with strip=False, because inside quotes the punctuation is part of the value."""
    seen, out = set(), []
    for x in xs:
        x = x.strip().rstrip(".,;:?!") if strip else x.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


def paths(text):
    return _dedup(m.group(1) for m in _PATH.finditer(text))


# ---- numbers ----------------------------------------------------------------------------------------------------------
# A number stands alone (not inside a path or an identifier). It may end before a quote, a bracket, '?', '!' or a full
# stop followed by a quote or bracket ("num_classes=`8`", 'a batch size of 6?'), and may use scientific notation (2e-4).
_NUMBER_ENDS = re.compile(r"(?<![\w./])(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)(?=$|[\s,;:)%x×'\"`\]}?!-]|\.(?:[\s'\"`)\]}]|$))")
_UNITS = {w: i for i, w in enumerate("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen
    fifteen sixteen seventeen eighteen nineteen""".split())}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_SCALES = {"hundred": 100, "thousand": 1000}


def numbers(text):
    """Numbers that stand alone ('5-fold' and '300 dpi' count), in digits (see _NUMBER_ENDS) or in words
    (number_words)."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ")
    return _dedup([m.group(1) for m in _NUMBER_ENDS.finditer(masked)] + number_words(masked))


def plain_numbers(text):
    """Numbers written in digits only, ending at a space or a separator: the simpler reading used for the values of an
    object's fields."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ")
    return _dedup(m.group(1) for m in _NUMBER.finditer(masked))


def number_words(text):
    """Numbers written in words, as digit strings, read as English numbers: 'nine' -> '9', 'one hundred' -> '100',
    'two hundred and fifty' -> '250', 'fifty thousand' -> '50000', 'a hundred' -> '100', 'six-class' -> '6'.
    Punctuation ends a number, 'and' continues one only after hundred or thousand, and a word that cannot follow the one
    before starts a new number ('ten, twenty-five and fifty' -> 10, 25, 50; 'forty-two and ten' -> 42, 10)."""
    out, state = [], {"total": 0, "cur": 0, "last": None}

    def flush():
        if state["last"] is not None:
            out.append(str(state["total"] + state["cur"]))
        state.update(total=0, cur=0, last=None)
    prev = ""
    for m in re.finditer(r"[A-Za-z]+|[,;:.!?()]", text.lower()):
        tok, last = m.group(0), state["last"]
        if tok in _UNITS:
            v = _UNITS[tok]
            if not (last in (None, "scale") or (last == "tens" and 1 <= v <= 9)):
                flush()
            state["cur"] += v; state["last"] = "unit"
        elif tok in _TENS:
            if last not in (None, "scale"):
                flush()
            state["cur"] += _TENS[tok]; state["last"] = "tens"
        elif tok in _SCALES and (last is not None or prev == "a"):
            state["cur"] = max(state["cur"], 1) * _SCALES[tok]
            if tok == "thousand":
                state["total"], state["cur"] = state["total"] + state["cur"], 0
            state["last"] = "scale"
        elif not (tok == "and" and last == "scale"):
            flush()
        prev = tok
    flush()
    return out


_EXTENSION = re.compile(r"(?<![\w/])(\.[A-Za-z][A-Za-z0-9]{1,5}(?:\.gz)?)(?![\w/])")


def extensions(text):
    """File extensions written on their own ('.tif', '.nii.gz'), not as the end of a path or file name."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ")
    return _dedup((m.group(1) for m in _EXTENSION.finditer(masked)), strip=False)


def quoted(text, verbatim=True):
    """Quoted spans (apostrophes inside words are not quotes). verbatim: keep the span exactly as written, final
    punctuation included; otherwise trailing punctuation is dropped."""
    return _dedup((m.group(1) for m in _QUOTED.finditer(text)), strip=not verbatim)


def quoted_paths(text):
    """Quoted spans that are paths containing spaces ('/home/jlee/Shared Data/scan.nii.gz'), whole."""
    return [q for q in quoted(text) if " " in q.strip() and _PATH_START.match(q)]


_SAME_FOLDER = re.compile(r"\b(?:same|parent|containing)\s+(?:folder|directory|dir|location)\b|\bnext to\b|\balongside\b"
                          r"|\bbeside\b", re.I)
_SPLIT_FOLDERS = {"train", "training", "val", "valid", "validation", "test", "testing"}
_HAS_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,5}$")


def stated_folders(text, written):
    """Folders the request names without writing them, from the paths it writes (`written`): the folder of a written
    file when the request refers to it ('in the same folder as the CSV', 'next to the input'), the folder that holds
    written split folders ('/data/histo/train and /data/histo/val' -> '/data/histo') and the folder that holds written
    split files ('/data/idh/train.npz' -> '/data/idh')."""
    out = []
    for p in written:
        p = p.rstrip("/")
        folder = os.path.dirname(p)
        if folder in ("", "/", "~", ".", ".."):
            continue
        name = os.path.basename(p)
        stem = re.sub(r"(\.[A-Za-z][A-Za-z0-9]{0,5})+$", "", name)
        if name.lower() in _SPLIT_FOLDERS or (_HAS_EXTENSION.search(name) and _SAME_FOLDER.search(text)) \
                or (stem != name and stem.lower() in _SPLIT_FOLDERS):
            out.append(folder)
    return _dedup(out)


_DATE_RANGE = re.compile(r"\b\d{8}-\d{8}\b")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_PREFIX = re.compile(r"\b(?:starts?|starting|begins?|beginning)\s+with\s+['\"`]?([\w.-]+)", re.I)


def formatted(text, description):
    """Values in a format the argument's description shows, built from what the request says: a year ->
    'YYYY0101-YYYY1231' when the description shows a YYYYMMDD-YYYYMMDD range; 'starts with X' -> 'X*' when it mentions
    wildcards. Each value is the request's own year or word in the documented format."""
    out = []
    if _DATE_RANGE.search(description or ""):
        out += [f"{y}0101-{y}1231" for y in _YEAR.findall(text)]
    if "*" in (description or "") or "wildcard" in (description or "").lower():
        out += [f"{w}*" for w in _PREFIX.findall(text)]
    return _dedup(out)


def identifiers(text):
    """Tokens like job_20260927_ab12, Dataset501_Liver, run-3, resnet50: letters mixed with digits or underscores."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ")
    return _dedup(m.group(1) for m in _IDENT.finditer(masked) if len(m.group(1)) >= 3)


# Command words describe the request rather than name an item of a list argument.
COMMAND_WORDS = set("""run train segment get use make save compute extract apply predict include exclude only all just also
    do give build create perform keep set skip write show check find rank compare classify load import continue add remove
    drop ignore want need like would should could let new existing my your their its some any every each but which what
    most how why where who there here not no more less very""".split())


# Words that never name a value ('... and trachea are enough').
FILLER_WORDS = set("""enough please thanks thank quick quickly instead again already really rather still yet too ok okay
    pls plz now later maybe perhaps etc""".split())


def phrases(text, max_n=2, limit=40):
    """Content-word n-grams (n <= max_n) in order of appearance ('kidneys, liver and spleen'). Paths, quoted strings and
    identifiers are candidates of their own, so they are cut out first; '+', '&', a spaced '/', commas, brackets and
    function words separate phrases ('liver + spleen' is two items, not the phrase 'liver spleen'). Command words, filler
    words ('enough', 'please') and number words never start, end or form a phrase (a number word is a number candidate of
    its own)."""
    masked = text
    for p in paths(text) + quoted(text, verbatim=False) + identifiers(text):
        masked = masked.replace(p, " , ")
    drop = _STOP | COMMAND_WORDS | FILLER_WORDS | set(_UNITS) | set(_TENS) | set(_SCALES) | {"dozen"}
    masked = re.sub(r"[+&]|\s/\s", " , ", masked)
    chunks = re.split(r"[,;:()]|\band\b|\bor\b|\bthen\b|\bwith\b|\bfrom\b|\bin\b|\bon\b|\bof\b|\binto\b|\bto\b|\bfor\b", masked, flags=re.I)
    out = []
    for c in chunks:
        ws = [w for w in _WORD.findall(c) if w.lower() not in drop and len(w) > 1]
        for n in range(1, max_n + 1):
            out += [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    return _dedup(out)[:limit]


def csv_columns(path, limit=200):
    """Header of a CSV/TSV file, or [] when the path is not a readable delimited file."""
    p = os.path.expanduser(str(path))
    if not os.path.isfile(p) or not p.lower().endswith((".csv", ".tsv", ".txt")):
        return []
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            head = f.readline()
        delim = "\t" if p.lower().endswith(".tsv") or head.count("\t") > head.count(",") else ","
        return _dedup(next(csv.reader([head], delimiter=delim)))[:limit]
    except (OSError, StopIteration, csv.Error):
        return []


# ---- paths put together from parts the request writes -----------------------------------------------------------------
_FOLDER_WORD = r"(?:sub-?folder|folder|sub-?directory|directory|dir)"
_FOLDER_NAMES = [re.compile(rf"\b{_FOLDER_WORD}\s+(?:called|named)\s+['\"`]?([\w.-]+)", re.I),     # a folder called X
                 re.compile(rf"['\"`]([\w.-]+)['\"`]\s+{_FOLDER_WORD}\b", re.I),                  # a 'X' folder
                 re.compile(rf"\b(?:an?|the)\s+([\w.-]+)\s+{_FOLDER_WORD}\b", re.I),               # the X folder
                 re.compile(r"\bsub-?(?:folder|directory)\s+['\"`]?([\w.-]+)", re.I),              # a subfolder X
                 re.compile(rf"\b(?:an?|the|new)\s+(?:new\s+)?{_FOLDER_WORD}\s+['\"`]?([\w-]*[_\d][\w.-]*)", re.I),
                 re.compile(rf"\b(?:an?|the|new)\s+(?:new\s+)?{_FOLDER_WORD}\s+['\"`]?([\w.-]+)['\"`]?\s+"
                            r"(?:next to|beside|alongside|inside|in|under|within)\b", re.I)]
_NOT_FOLDER_NAMES = set("""same parent containing new that this its their my our your separate other own output input
    working current respective corresponding whole entire single specified given of in inside next beside for with called
    named a an the to into under within at by from on""".split())
_RELATIVE_IN = re.compile(r"(?<![\w/~.])([\w.-]+/[\w./-]+)\s+(?:inside|in|under|within)\s+((?:~|\.{1,2})?/\S+)")


def folder_names(text):
    """Names the request gives to folders without writing their path: 'a folder called merlin_embeddings', "a
    'merlin_pheno' folder", 'the masks folder', 'a subfolder risk_merlin', 'a folder brainage_pycaret' (a name with a
    digit or underscore right after the folder word) and 'a folder tseg next to ...' (a name followed by where the folder
    goes). 'the model directory /data/x' names no folder (the path follows)."""
    out = [m.group(1) for p in _FOLDER_NAMES for m in p.finditer(text)
           if not _PATH_START.match(text[m.end():].lstrip())]
    return _dedup(n for n in out if n.lower() not in _NOT_FOLDER_NAMES and not n.startswith((".", "-")))


_ABS = r"((?:~|\.{1,2})?/[^\s'\"`,;()]+)"
_SUBFOLDER_OF = [re.compile(rf"(?<![\w/~.])([\w.-]+)/?\s+{_FOLDER_WORD}\s+(?:of|in|inside|under|within)\s+{_ABS}", re.I),
                 re.compile(rf"\bsub-?(?:folder|directory)\s+['\"`]?([\w.-]+)['\"`]?\s+(?:of|in|inside|under|within)\s+{_ABS}",
                            re.I)]
_ITS_SUBFOLDER = re.compile(rf"\b(?:its|their|the same)\s+['\"`]?([\w.-]+)['\"`]?\s+{_FOLDER_WORD}", re.I)
_SUBFOLDERS = re.compile(r"((?:[\w.-]+\s*,\s*)*[\w.-]+\s*,?\s+(?:and|or)\s+[\w.-]+)\s+(?:sub-?)?(?:folders|directories|dirs)\b",
                         re.I)
_SAVED_AS = re.compile(r"\b(?:saved|stored|exported|written|kept)\s+(?:as|under)\s+['\"`]?([A-Za-z][\w.-]*)"
                       r"|\b(?:model|pipeline|file|checkpoint|weights|network|artifact)\b(?:\s+\w+){0,2}\s+(?:called|named)"
                       r"\s+['\"`]?([A-Za-z][\w.-]*)", re.I)
_BARE_FILE = re.compile(r"(?<![\w/.~-])([\w-]+\.(?:npz|npy|h5|hdf5|tiff?|mhd|bmp|joblib|onnx|ckpt|safetensors))(?![\w/])",
                        re.I)
_NAME_IN = re.compile(rf"(?<![\w/.~-])([A-Za-z][\w.-]*[_\d][\w.-]*)\s+\(?(?:[A-Za-z]+\s+){{0,3}}"
                      rf"(?:in|from|inside|under|within)\s+(?:the\s+)?{_ABS}", re.I)


def joined_paths(text, limit=30):
    """Paths put together from parts the request writes. Every join is offered; the model picks the one the request
    means. In this order:
      - folders named relative to a written folder ('the test/ subfolder of /data/ct', 'its gt subfolder', 'imagesTr and
        labelsTr subfolders');
      - names placed in a written folder ('final_v2 (it lives in /models/x)');
      - names given to an output ('saved as tuned_lgbm', 'a pipeline called final_ridge') in each written folder and
        each folder above;
      - file names in those named subfolders ('train.csv in the splits subfolder of /data/x');
      - a written folder, or the folder of a written path, joined with a file name written without its folder
        ('best.pt', 'external.csv', 'test.npz') or with a folder the request names (folder_names);
      - a relative path placed inside a folder ('jobs/7731 inside /data/x').
    Paths the request writes itself are not repeated."""
    written = paths(text)
    absolute = [p for p in written if p.startswith(("/", "~"))]
    bases = []
    for p in absolute:
        q = p.rstrip("/")
        if not _HAS_EXTENSION.search(os.path.basename(q)):
            bases.append(q)
        d = os.path.dirname(q)
        if d not in ("", "/", "~", ".", ".."):
            bases.append(d)
    bases = _dedup(bases, strip=False)
    names = [p for p in written if "/" not in p] + folder_names(text)
    names += [m.group(1) for m in _BARE_FILE.finditer(text) if m.group(1) not in names]
    first = []
    folders = [b for b in bases]
    sub = []
    for pat in _SUBFOLDER_OF:
        sub += [os.path.join(m.group(2).rstrip(".,;:").rstrip("/"), m.group(1).rstrip("/")) for m in pat.finditer(text)
                if m.group(1).lower() not in _NOT_FOLDER_NAMES]
    rel = [m.group(1) for m in _ITS_SUBFOLDER.finditer(text) if m.group(1).lower() not in _NOT_FOLDER_NAMES]
    for m in _SUBFOLDERS.finditer(text):
        rel += [x for x in re.split(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", m.group(1).strip())
                if x and x.lower() not in _NOT_FOLDER_NAMES and x.lower() not in _STOP]
    sub += [os.path.join(b, n) for n in _dedup(rel) for b in folders if n != os.path.basename(b)]
    sub = _dedup(sub, strip=False)
    given = [(m.group(1) or m.group(2)).rstrip(".,;:") for m in _SAVED_AS.finditer(text)]
    given = [n for n in given if n.lower() not in _NOT_FOLDER_NAMES and "/" not in n]
    first += sub
    first += [os.path.join(m.group(2).rstrip(".,;:").rstrip("/"), m.group(1).rstrip(".,;:"))
              for m in _NAME_IN.finditer(text)]
    first += [os.path.join(b, n) for n in _dedup(given) for b in bases + sub if n != os.path.basename(b)]
    first += [os.path.join(s, n) for n in _dedup(names) for s in sub if n != os.path.basename(s)]
    out = [os.path.join(b, n) for n in _dedup(names) for b in bases if n != os.path.basename(b)]
    out += [os.path.join(m.group(2).rstrip(".,;:").rstrip("/"), m.group(1)) for m in _RELATIVE_IN.finditer(text)]
    return [p for p in _dedup(first + out) if p not in written][:limit]


# ---- numbers stated in another form ------------------------------------------------------------------------------------
_FRACTION_ARG = re.compile(r"\b(fraction|proportion|ratio|share)\b|between 0 and 1", re.I)
_SECONDS_ARG = re.compile(r"(^|_)(sec|secs|seconds)$|\bseconds?\b", re.I)
_ID_ARG = re.compile(r"(^|_)id$")
_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)", re.I)
_DENOMINATORS = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fifth": 5, "fifths": 5,
                 "sixth": 6, "sixths": 6, "seventh": 7, "sevenths": 7, "eighth": 8, "eighths": 8, "ninth": 9, "ninths": 9,
                 "tenth": 10, "tenths": 10}
_FRACTION = re.compile(r"\b(a|an|one|two|three|four|five|six|seven|eight|nine)\s+(" + "|".join(_DENOMINATORS) + r")\b", re.I)
_DURATION = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|[A-Za-z]+)\s*(minutes?|mins?|hours?|hrs?)\b", re.I)
_DOZEN = re.compile(r"\b(a|one|two|three|four|five|half a)\s+dozen\b", re.I)
_ORDINALS = {w: i for i, w in enumerate("""first second third fourth fifth sixth seventh eighth ninth tenth eleventh
    twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth""".split(), start=1)}
_ORDINAL_CLASS = re.compile(r"\b(" + "|".join(_ORDINALS) + r"|\d+(?:st|nd|rd|th))\s+(?:new\s+)?"
                            r"(?:class|classes|category|categories|label|labels|grade|grades)\b", re.I)
_EMBEDDED_ID = re.compile(r"\b[A-Za-z]+0*(\d+)_[A-Za-z]\w*")


def _num(v):
    v = round(v, 4)
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _word_number(w):
    w = w.lower()
    return {"a": 1, "an": 1}.get(w, _UNITS.get(w))


def derived_numbers(text, name, prop):
    """Numbers the request states in another form, for a number argument. Each rule needs the argument to take that kind
    of number: 'binary' -> 2 (integers); '20 percent', 'one fifth' -> 0.2 (fractions: the description says fraction,
    proportion or ratio); '5 minutes' -> 300 (seconds); 'a dozen' -> 12; 'a fourth class' -> 4 (integers);
    'Dataset112_LiverMets' -> 112 (an argument named ..._id)."""
    desc = prop.get("description") or ""
    integer = prop.get("type") == "integer"
    out = []
    if integer and re.search(r"\bbinary\b", text, re.I):
        out.append("2")
    if _FRACTION_ARG.search(desc):
        out += [_num(float(m.group(1)) / 100) for m in _PERCENT.finditer(text)]
        out += [_num(_word_number(m.group(1)) / _DENOMINATORS[m.group(2).lower()]) for m in _FRACTION.finditer(text)
                if _word_number(m.group(1))]
    if _SECONDS_ARG.search(name) or _SECONDS_ARG.search(desc):
        for m in _DURATION.finditer(text):
            n = _word_number(m.group(1)) if m.group(1).isalpha() else float(m.group(1))
            if n:
                out.append(_num(n * (3600 if m.group(2).lower().startswith("h") else 60)))
    out += [_num(12 * (0.5 if m.group(1).lower() == "half a" else _word_number(m.group(1)))) for m in _DOZEN.finditer(text)]
    if integer:
        for m in _ORDINAL_CLASS.finditer(text):
            w = m.group(1).lower()
            out.append(str(_ORDINALS[w]) if w in _ORDINALS else re.match(r"\d+", w).group())
        if _ID_ARG.search(name):
            out += [str(int(m.group(1))) for m in _EMBEDDED_ID.finditer(text)]
    return _dedup(out)


def occurrences(text, value):
    """How many times the request writes a value: numbers by value ('2' and '2.0' are one number, written in digits or
    words), other values as written (case ignored); at least 1, since some values are not written literally
    (documented values, joined paths). A value written k times may fill k arguments or list items."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ")
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = None
    if v is not None:
        nums = [m.group(1) for m in _NUMBER_ENDS.finditer(masked)] + number_words(masked)
        return max(1, sum(1 for n in nums if float(n) == v))
    return max(1, len(spans(text, str(value))))


def spans(text, value):
    """Where the request writes a value, as (start, end) pairs, case ignored."""
    if not value.strip():
        return []
    return [(m.start(), m.end()) for m in re.finditer(r"(?<![\w])" + re.escape(value) + r"(?![\w])", text, re.I)]


def _number(s):
    return int(s) if re.fullmatch(r"-?\d+", s) else float(s)


# ---- dictionaries and list items ---------------------------------------------------------------------------------------
_PAIR = r"(?:[A-Za-z_][\w-]*|\d+)\s*[:=]\s*-?\d+(?:\.\d+)?"
_PAIRS = re.compile(rf"(?<![\w.:=]){_PAIR}(?:\s*,\s*(?:and\s+)?{_PAIR})+")
_ONE_PAIR = re.compile(r"([A-Za-z_][\w-]*|\d+)\s*[:=]\s*(-?\d+(?:\.\d+)?)")


def dicts(text):
    """Dictionaries the request writes: JSON or Python literals ('{"rotation": 20, "horizontal_flip": true}',
    '{0: 1.0, 1: 1.2}') and runs of at least two key-number pairs ('0:1.0, 1:9.0', 'RV = 1, MYO = 2, LV = 3'). Keys
    become strings, as in JSON."""
    out, i = [], 0
    while (i := text.find("{", i)) != -1:
        depth, j = 0, i
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
        span = text[i:j + 1]
        for parse in (json.loads, lambda s: ast.literal_eval(re.sub(r"\b(true|false|null)\b",
                      lambda m: {"true": "True", "false": "False", "null": "None"}[m.group(1)], s))):
            try:
                d = parse(span)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(d, dict) and d:
                out.append({str(k): v for k, v in d.items()})
                break
        i = j + 1 if depth == 0 else i + 1
    masked = re.sub(r"\{[^{}]*\}", " ", text)
    for m in _PAIRS.finditer(masked):
        out.append({k: _number(v) for k, v in _ONE_PAIR.findall(m.group(0))})
    seen, unique = set(), []
    for d in out:
        key = repr(sorted(d.items(), key=str))
        if key not in seen:
            seen.add(key); unique.append(d)
    return unique


_ITEM = r"[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3}"
_AFTER_COLON = re.compile(r":\s*(.+?)(?=\.\s|\.$|;|\n|$)")
_COMMA_LIST = re.compile(rf"({_ITEM}(?:\s*,\s*{_ITEM}){{2,}}(?:\s*,?\s+(?:and|or)\s+{_ITEM})?)")
_SLASH_LIST = re.compile(r"(?<![\w/~.])([A-Za-z][A-Za-z-]*(?:/[A-Za-z][A-Za-z-]*)+)(?![\w/])")
_ITEM_SPLIT = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s*/\s*|\s+(?:and|or)\s+")


def written_items(text):
    """List items as the request writes them: the parts of an enumeration after a colon ('Class names: no DR, mild,
    moderate'), of a comma list of three or more ('cyst, normal, stone and tumor') and of words joined by slashes
    ('benign/malignant/normal'), multi-word items whole. From the first part of a comma list only its last word or two
    are kept, since it starts in the sentence before the list; an item ends before a word that cannot be part of it
    ('trachea are')."""
    masked = text
    for p in paths(text):
        if p.startswith(("/", "~", ".")) or re.search(r"[\d_.]", p):
            masked = masked.replace(p, " ; ")
    out = []
    for m in _AFTER_COLON.finditer(masked):
        parts = [x.strip(" '\"`[]") for x in _ITEM_SPLIT.split(m.group(1))]
        parts = [x for x in parts if re.fullmatch(_ITEM, x)]
        if len(parts) >= 2:
            out += parts
    for m in _COMMA_LIST.finditer(masked):
        parts = [x.strip() for x in _ITEM_SPLIT.split(m.group(1)) if x.strip()]
        first = parts[0].split()
        out += [" ".join(first[-n:]) for n in (1, 2) if len(first) >= n] + parts[1:]
    for m in _SLASH_LIST.finditer(masked):
        out += m.group(1).split("/")
    drop = _STOP | COMMAND_WORDS | FILLER_WORDS
    cut = []
    for x in out:
        ws = x.split()
        n = next((i for i, w in enumerate(ws) if i and w.lower() in drop), len(ws))
        if ws[0].lower() not in drop or (n > 1 and ws[0].lower() == "no"):              # 'no DR', 'no tumor'
            cut.append(" ".join(ws[:n]))
    return _dedup(cut)


_LABEL_LINE = re.compile(r"(?m)^[ \t]*([A-Za-z][A-Za-z _-]{0,40}?)[ \t]*:[ \t]*(.*)$")


def labelled_text(text):
    """Values written after a label on their own line ('Report text:' and the lines that follow, up to a blank line)
    -> [(label, value)]."""
    out = []
    for m in _LABEL_LINE.finditer(text):
        value = m.group(2).strip()
        if not value:
            after = text[m.end():].lstrip("\n").split("\n\n")[0]
            value = " ".join(x.strip() for x in after.splitlines() if x.strip())
        if value:
            out.append((m.group(1).strip(), value))
    return out


_GLOB = re.compile(r"[*?]")
_REGEX = re.compile(r"\\[dDwWsSbB.]|\(.*\)|\[.*\]|^\^|\$$")


def patterns(text, which=None):
    """Glob and regex patterns the request writes ('*_lesionmask.nii.gz', 'sub-(\\d+)'), without the sentence's
    punctuation around them. which: 'glob' or 'regex' keeps only that kind (a glob has * or ?, a regex escapes, groups
    or anchors); None keeps both."""
    out = []
    for tok in text.split():
        t = tok.strip("'\"`").rstrip(".,;:")
        while t.endswith(")") and t.count(")") > t.count("("):
            t = t[:-1].rstrip(".,;:")
        t = t.lstrip("(") if t.count("(") > t.count(")") else t
        glob, regex = bool(_GLOB.search(t)) and not _REGEX.search(t), bool(_REGEX.search(t))
        if re.search(r"[A-Za-z0-9]", t) and ((glob and which != "regex") or (regex and which != "glob")):
            out.append(t)
    return _dedup(out, strip=False)


# ---- ranges and counts -------------------------------------------------------------------------------------------------
_SMALL = {**{w: i for w, i in _UNITS.items()}, "twenty": 20}
_RANGE_NUM = r"(\d{1,2}|" + "|".join(_SMALL) + r")"
_TO = r"\s*(?:-|–|\.\.|\bto\b|\bthrough\b|\bthru\b)\s*"
_RANGE = re.compile(rf"(?<![\w.]){_RANGE_NUM}{_TO}{_RANGE_NUM}(?!\w|\.\d)", re.I)
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
_ROMAN_RANGE = re.compile(rf"(?<![\w.])(XII|XI|IX|X|VIII|VII|VI|IV|V|III|II|I){_TO}(XII|XI|IX|X|VIII|VII|VI|IV|V|III|II|I)(?!\w)")
_PREFIX_RANGE = re.compile(rf"(?<![\w.])([A-Z]{{1,3}})(\d{{1,2}}){_TO}\1?(\d{{1,2}})(?!\w|\.\d)")
_CLASS_WORD = re.compile(r"\b(?:grades?|grading|scores?|stages?|levels?|types?|class(?:es)?|categor(?:y|ies)|labels?|"
                         r"ratings?|severity)\b", re.I)


def _small(s):
    return int(s) if s.isdigit() else _SMALL[s.lower()]


def ranges(text, classes=False):
    """Inclusive ranges of small whole numbers the request writes ('0-4', '0 to 4', 'one through three'), outside paths,
    as (first, last) with first < last <= first + 20. classes: only ranges next to a word for grades, scores, stages,
    types, classes or labels (within four words before, or right after), and then also Roman numerals ('type I to VI')
    and ranges with a letter prefix ('S0 to S3', 'F0-F4')."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ")
    found = [(m, _small(m.group(1)), _small(m.group(2))) for m in _RANGE.finditer(masked)]
    if classes:
        found += [(m, _ROMAN[m.group(1)], _ROMAN[m.group(2)]) for m in _ROMAN_RANGE.finditer(masked)]
        found += [(m, int(m.group(2)), int(m.group(3))) for m in _PREFIX_RANGE.finditer(masked)]
    out = []
    for m, a, b in found:
        if not a < b <= a + 20:
            continue
        before = " ".join(re.findall(r"[\w-]+", masked[:m.start()])[-4:])
        after = masked[m.end():m.end() + 20]
        if classes and not (_CLASS_WORD.search(before) or _CLASS_WORD.match(after.lstrip(" )"))):
            continue
        out.append((a, b))
    return list(dict.fromkeys(out))


def range_items(text):
    """The numbers of each written range, as list items ('folds 0 through 4' -> 0, 1, 2, 3, 4)."""
    return _dedup(str(i) for a, b in ranges(text) for i in range(a, b + 1))


_ITEM2 = r"[A-Za-z][\w'+-]*(?:\s+[A-Za-z0-9][\w'+-]*){0,3}"
_COMMA_LIST2 = re.compile(rf"({_ITEM2}(?:\s*,\s*{_ITEM2})+\s*,?\s+(?:and|or)\s+{_ITEM2}|{_ITEM2}(?:\s*,\s*{_ITEM2}){{2,}})")
_PAIR_LIST = re.compile(rf"({_ITEM2}\s+(?:and|or|vs\.?|versus|/)\s+{_ITEM2})")
_FROM_PAIR = re.compile(rf"\b(?:separate|separating|distinguish|distinguishing|differentiate|differentiating|discriminate|"
                        rf"tell)\s+({_ITEM2})\s+from\s+({_ITEM2})", re.I)
_QUOTED_LIST = re.compile(r"""((?<![A-Za-z0-9])['"`][^'"`]{1,60}['"`](?:\s*(?:,\s*(?:and\s+|or\s+)?|\s(?:and|or|vs\.?|versus)\s)\s*"""
                          r"""['"`][^'"`]{1,60}['"`](?![A-Za-z0-9]))+)""")
_SLASH_ITEMS = re.compile(r"(?<![\w/~.])([A-Za-z][\w+-]{0,9}(?:/[A-Za-z][\w+-]{0,9})+)(?![\w/.])")
_SPLIT2 = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s+/\s+|\s+(?:and|or|vs\.?|versus)\s+")
_HEAD_NOUNS = set("""model models algorithm algorithms method methods learners estimators regressors classifiers features
    columns variables classes categories labels organs structures filters lesions cases""".split())


def _item_forms(part, where="whole"):
    """An item as written, without leading articles and command words, and also without a trailing noun such as
    'models'. where: 'first' (the first part of a list starts in the sentence before it, so its last one to four words
    are each an item), 'last' (cut before a later word that cannot be part of it: 'trees are enough'), 'middle' (between
    two separators: kept whole, 'ductal carcinoma in situ'), 'whole' (as 'last')."""
    drop = _STOP | COMMAND_WORDS | FILLER_WORDS
    ws = part.split()
    out = []
    for w in ([ws[-n:] for n in (1, 2, 3, 4) if len(ws) >= n] if where == "first" else [ws]):
        while w and w[0].lower() in drop and not (w[0].lower() == "no" and len(w) > 1):
            w = w[1:]
        if where != "middle":
            n = next((i for i, t in enumerate(w) if i and t.lower() in drop), len(w))
            w = w[:n]
        if w:
            out.append(" ".join(w))
            if len(w) > 1 and w[-1].lower() in _HEAD_NOUNS:
                out.append(" ".join(w[:-1]))
    return out


def _masked_items(text):
    masked = text
    for p in paths(text):
        if p.startswith(("/", "~", ".")) or re.search(r"[\d_.]", p):
            masked = masked.replace(p, " ; ")
    return masked


def _list_parts(text):
    """(parts, whether the list starts in the sentence before) for every enumeration the request writes."""
    masked = _masked_items(text)
    out = []
    for m in _AFTER_COLON.finditer(masked):
        parts = [x.strip(" '\"`[]") for x in _SPLIT2.split(m.group(1))]
        parts = [x for x in parts if re.fullmatch(_ITEM2, x)]
        if len(parts) >= 2:
            out.append((parts, False))
    for pat in (_COMMA_LIST2, _PAIR_LIST):
        for m in pat.finditer(masked):
            out.append(([x.strip() for x in _SPLIT2.split(m.group(1)) if x.strip()], True))
    return out


def _forms_of(parts, open_start):
    n = len(parts)
    return [_item_forms(x, "first" if open_start and i == 0 else "last" if i == n - 1 else "middle")
            for i, x in enumerate(parts)]


def item_groups(text):
    """The enumerations the request writes, each as its list of items: lists after a colon, comma lists of three or more
    ('A, B and C' included), pairs joined by 'and', 'or', 'vs' or a spaced slash, words joined by slashes
    ('KL0-1/KL2/KL3-4'), 'separate A from B', and quoted values joined the same way."""
    groups = []
    for parts, open_start in _list_parts(text):
        forms = [f[-1] if open_start and i == 0 else f[0] for i, f in enumerate(_forms_of(parts, open_start)) if f]
        if len(forms) >= 2:
            groups.append(forms)
    for m in _SLASH_ITEMS.finditer(text):
        items = m.group(1).split("/")
        if not any(_HAS_EXTENSION.search(x) for x in items):
            groups.append(items)
    for m in _FROM_PAIR.finditer(_masked_items(text)):           # 'separate adenoma from non-adenoma'
        a, b = _item_forms(m.group(1), "first"), _item_forms(m.group(2), "last")
        if a and b:
            groups.append([a[-1], b[0]])
    for m in _QUOTED_LIST.finditer(text):
        groups.append(re.findall(r"""['"`]([^'"`]{1,60})['"`]""", m.group(1)))
    return groups


_N_CLASS = re.compile(r"(?<![\w.])(\d{1,2})[-\s]class(?:es)?\b|(?<![\w.])(\d{1,2})\s+(?:existing\s+|original\s+|old\s+)"
                      r"?(?:classes|categories)\b|\boriginal\s+(\d{1,2})\b", re.I)
_NEW_CLASSES = re.compile(r"(?<![\w.])(\d{1,2}|one|two|three|four|five|a|an)\s+(?:new|additional|extra|more)\s+"
                          r"(?:class(?:es)?|categor(?:y|ies)|labels?)\b", re.I)
_ADDS = re.compile(r"\badd(?:s|ing|ed)?\b", re.I)


def class_counts(text, total=False):
    """Class counts the request gives without writing them: the number of items of each enumeration it writes
    (item_groups: 'glioma, meningioma, pituitary and no tumour' -> 4, 'benign vs malignant' -> 2) and of each range next
    to a word for grades, scores or classes ('grade 0 to 4' -> 5). total: for an argument that holds the number of
    classes after adding new ones, the counts the request gives (also '3-class', 'the original 3') plus the classes it
    adds ('2 new classes', 'a new class', 'adds tuberculosis' -> 1)."""
    counts = _dedup([str(len(g)) for g in item_groups(text) if len(g) >= 2]
                    + [str(b - a + 1) for a, b in ranges(text, classes=True)])
    if not total:
        return counts
    base = [int(c) for c in counts] + [int(next(g for g in m.groups() if g)) for m in _N_CLASS.finditer(text)]
    added = [int(m.group(1)) if m.group(1).isdigit() else (_word_number(m.group(1)) or 1) for m in _NEW_CLASSES.finditer(text)]
    if not added and _ADDS.search(text):
        added = [1]
    return _dedup(counts + [str(b + a) for b in base for a in added])


_FRACTIONS = {"a half": 0.5, "half": 0.5, "a quarter": 0.25, "one quarter": 0.25, "quarter": 0.25, "three quarters": 0.75}
_FRACTION_OF = re.compile(r"\b(half|a\s+quarter|one\s+quarter|quarter|three\s+quarters)\s+(?:of\s+)?(?:an?\s+)?"
                          r"(minute|hour)s?\b", re.I)
_AND_FRACTION = re.compile(r"(?<![\w.])(\d+|[A-Za-z]+)\s+(minutes?|hours?)\s+and\s+(a\s+half|a\s+quarter|three\s+quarters)\b"
                           r"|(?<![\w.])(\d+|[A-Za-z]+)\s+and\s+(a\s+half|a\s+quarter|three\s+quarters)\s+(minutes?|hours?)\b",
                           re.I)
_UNIT = r"(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
_COMPOUND = re.compile(rf"(?<![\w.])(\d+\s*|[A-Za-z]+(?:-[A-Za-z]+)?\s+){_UNIT}\s*,?\s*(?:and\s+)?"
                       rf"(\d+\s*|[A-Za-z]+(?:-[A-Za-z]+)?\s+){_UNIT}", re.I)


def _seconds(unit):
    u = unit.lower()
    return 3600 if u.startswith("h") else 60 if u.startswith("m") else 1


def _amount(w):
    if w.isdigit():
        return int(w)
    n = number_words(w)
    return int(n[0]) if len(n) == 1 else {"a": 1, "an": 1}.get(w.lower())


def durations(text, name, prop):
    """Durations written with fractions or in two units, for an argument in seconds: 'half a minute' -> 30, 'a quarter
    of an hour' -> 900, 'a minute and three quarters' -> 105, 'one and a half minutes' -> 90, '4 min 30 s' -> 270, 'two
    minutes and twenty seconds' -> 140."""
    if not (_SECONDS_ARG.search(name) or _SECONDS_ARG.search(prop.get("description") or "")):
        return []
    out = []
    for m in _FRACTION_OF.finditer(text):
        out.append(_num(_FRACTIONS[re.sub(r"\s+", " ", m.group(1).lower())] * _seconds(m.group(2))))
    for m in _AND_FRACTION.finditer(text):
        n, u, f = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(4), m.group(6), m.group(5))
        v = _amount(n)
        if v:
            out.append(_num((v + _FRACTIONS[re.sub(r"\s+", " ", f.lower())]) * _seconds(u)))
    for m in _COMPOUND.finditer(text):
        a, ua, b, ub = m.groups()
        x, y = _amount(a.strip()), _amount(b.strip())
        if x is not None and y is not None and _seconds(ua) > _seconds(ub):
            out.append(_num(x * _seconds(ua) + y * _seconds(ub)))
    return _dedup(out)


# ---- dictionaries written as prose -------------------------------------------------------------------------------------
_VALUE = r"-?\d+(?:\.\d+)?"
_KEY = r"(?:[A-Za-z_][\w-]*|\d+)"
_SEP = r"\s*(?:,\s*(?:and\s+)?|;\s*|\s+and\s+)"
_WSEP = r"\s*(?:,\s*(?:and\s+)?|\s+and\s+)"
_ARROW_PAIR = rf"(?:(?:class|label|category)\s+)?{_KEY}\s*(?:->|→|=>|:|=)\s*{_VALUE}"
_ARROW_PAIRS = re.compile(rf"(?<![\w.:=]){_ARROW_PAIR}(?:{_SEP}{_ARROW_PAIR})+")
_ONE_ARROW = re.compile(rf"({_KEY})\s*(?:->|→|=>|:|=)\s*({_VALUE})")
_FOR_PAIR = rf"{_VALUE}\s+for\s+(?:the\s+)?(?:class(?:es)?\s+|labels?\s+|categor(?:y|ies)\s+)?{_KEY}"
_FOR_PAIRS = re.compile(rf"(?<![\w.]){_FOR_PAIR}(?:{_SEP}{_FOR_PAIR})+")
_ONE_FOR = re.compile(rf"({_VALUE})\s+for\s+(?:the\s+)?(?:class(?:es)?\s+|labels?\s+|categor(?:y|ies)\s+)?({_KEY})")
_WORD_VALUE = r"[A-Za-z_][\w-]*\s+(?:-?\d+(?:\.\d+)?|true|false)(?!\w|\.\d)"
_WORD_PAIRS = re.compile(rf"(?<![\w.]){_WORD_VALUE}(?:{_WSEP}{_WORD_VALUE})+", re.I)
_ONE_WORD_VALUE = re.compile(r"([A-Za-z_][\w-]*)\s+(-?\d+(?:\.\d+)?|true|false)(?!\w|\.\d)", re.I)
_LABEL_PAIR = re.compile(r"\blabels?\s+(?:is|are|=|:)?\s*(?:the\s+)?([A-Za-z_][\w-]*)\s+(\d+)(?!\w|\.\d)", re.I)
_CUE_KEYS = {"class", "classes", "weight", "weights", "weighted", "counts", "count", "label", "labels", "gets", "at"}
_NUM = r"(?:-?\d+(?:\.\d+)?|" + "|".join(_SMALL) + r")"
_VALUE_LIST = re.compile(rf"(?<![\w.])({_NUM}(?:\s*(?:,|/|,?\s+and)\s*{_NUM})+)(?!\w|\.\d)", re.I)
_KEYSPEC = r"(\d{1,2}\s*(?:to|-|–|through|thru)\s*\d{1,2}|\d{1,2}(?:\s*(?:,\s*and|,|and)\s*\d{1,2})*)"
_CLASS_MENTION = re.compile(rf"\bclass(?:es)?\s+{_KEYSPEC}(?!\w|\.\d)", re.I)
_AFTER_MENTION = re.compile(rf"\s*\)?\s*(?:\([^()]{{0,60}}\)\s*)?(?:(?!class(?:es)?\s+\d)[A-Za-z'-]+\s+){{0,5}}?({_NUM})(?!\w|\.\d)",
                            re.I)


def _keys(spec):
    """'0 to 3' -> 0, 1, 2, 3; '0, 1 and 2' -> 0, 1, 2."""
    m = re.fullmatch(r"(\d{1,2})\s*(?:to|-|–|through|thru)\s*(\d{1,2})", spec.strip(), re.I)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return [str(i) for i in range(a, b + 1)] if a < b <= a + 20 else []
    keys = []
    for k in re.findall(r"\d+", spec):                  # 'class 0, 50 epochs': the list ends before 50
        if int(k) > 20 or k in keys:
            break
        keys.append(k)
    return keys


def _num_value(s):
    return _number(s) if re.fullmatch(_VALUE, s) else _SMALL[s.lower()]


def _values(s):
    return [_num_value(x) for x in re.findall(_NUM, s, re.I)]


def class_values(text):
    """Per-class values written around class numbers: 'class 1 weighted 1.5 and class 0 weighted 1', 'weight classes 4
    and 5 at 2 and classes 0 to 3 at 1', 'class 2 (gangrenous) gets weight 3', 'weights 1.0/1.5/2.0 over classes 0-2',
    'classes 0 to 2 (normal, TB, other) get weights 1, 3 and 1.5'. -> dictionaries keyed by class."""
    masked = text
    for p in paths(text):
        if re.search(r"[A-Za-z]", p):
            masked = masked.replace(p, " ; ")
    out, merged = [], {}
    mentions = list(_CLASS_MENTION.finditer(masked))
    for m in mentions:
        keys = _keys(m.group(1))
        if not keys:
            continue
        rest = masked[m.end():m.end() + 120]
        nxt = next((x.start() - m.end() for x in mentions if x.start() > m.start()), len(rest))
        rest = rest[:max(0, nxt)]
        lst = re.match(rf"\s*\)?\s*(?:\([^()]{{0,60}}\)\s*)?(?:[A-Za-z'-]+\s+){{0,5}}?{_VALUE_LIST.pattern}", rest, re.I)
        if lst and len(keys) > 1 and len(_values(lst.group(1))) == len(keys):
            out.append(dict(zip(keys, _values(lst.group(1))))); continue
        one = _AFTER_MENTION.match(rest)
        if one:
            merged.update({k: _num_value(one.group(1)) for k in keys})
    for v in _VALUE_LIST.finditer(masked):                     # values first, then the classes they are for
        vals = _values(v.group(1))
        for m in mentions:
            gap = masked[v.end():m.start()]
            keys = _keys(m.group(1))
            if 0 <= m.start() - v.end() <= 40 and not re.search(r"\d", gap) and len(keys) == len(vals) > 1:
                out.append(dict(zip(keys, vals)))
    if len(merged) >= 2:
        out.insert(0, merged)
    return out


def prose_dicts(text):
    """Dictionaries written as prose: key-value pairs with arrows ('0 -> 1.0 and 1 -> 2.2', 'class 0: 1, class 1: 3'),
    values for named keys ('1.0 for class 0 and 1.6 for class 1'), values around class numbers (class_values), a list of
    values for a range of keys ('1.0, 0.5, 1.5 and 1.5 for classes 0 to 3'), runs of name-value pairs ('kidney 1 and
    tumor 2', 'rotation 90, horizontal_flip true') and a single label ('the only label is head 1'). Keys become strings,
    as in JSON."""
    masked = text
    for p in paths(text):
        masked = masked.replace(p, " ; ")
    drop = _STOP | COMMAND_WORDS | FILLER_WORDS | _CUE_KEYS
    out = class_values(text)
    for m in _ARROW_PAIRS.finditer(masked):
        out.append({k: _number(v) for k, v in _ONE_ARROW.findall(m.group(0))})
    for m in _FOR_PAIRS.finditer(masked):
        out.append({k: _number(v) for v, k in _ONE_FOR.findall(m.group(0))})
    for m in _WORD_PAIRS.finditer(masked):
        pairs = [(k, v) for k, v in _ONE_WORD_VALUE.findall(m.group(0)) if k.lower() not in drop]
        if len(pairs) >= 2:
            out.append({k: _value(v) for k, v in pairs})
    for m in _LABEL_PAIR.finditer(masked):                     # a single label, when no run of pairs has it
        if m.group(1).lower() not in drop and not any(m.group(1) in d for d in out):
            out.append({m.group(1): int(m.group(2))})
    seen, unique = set(), []
    for d in out:
        key = repr(sorted(d.items(), key=str))
        if d and key not in seen:
            seen.add(key); unique.append(d)
    return unique


def _value(v):
    return {"true": True, "false": False}.get(v.lower()) if v.lower() in ("true", "false") else _number(v)


_DOTTED = re.compile(r"(?<![\w/.~-])([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+)(?![\w/-])")
_FILE_EXT = re.compile(r"\.(?:csv|tsv|xlsx?|json|ya?ml|txt|nii|gz|nrrd|mha|dcm|pkl|pth|pt|png|jpe?g|npz|npy|h5|hdf5|py|"
                       r"tif|tiff|bmp|zip|tar|log|md|html|pdf)$", re.I)


def dotted_names(text):
    """Names with a dot inside that are not file names ('deadstatus.event', 'event.flag'); 'e.g' and 'i.e' are not
    names."""
    masked = text
    for p in paths(text):
        if "/" in p:
            masked = masked.replace(p, " ")
    return _dedup(m.group(1) for m in _DOTTED.finditer(masked)
                  if not _FILE_EXT.search(m.group(1)) and m.group(1).lower() not in ("e.g", "i.e", "etc")
                  and all(len(x) >= 2 for x in m.group(1).split(".")))


# ---- compact tokens, sequences, genomic and spectral notation ---------------------------------------------------------
_TOKEN = re.compile(r"\S+")
_INNER = re.compile(r"[:\[\]=()+]")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]+=[\w.+-]*$")           # num_classes=5: an argument written out
_WORD_JOIN = re.compile(r"^[A-Za-z]+(?:\+[A-Za-z]+)+$")                   # plan+preprocess: words joined by '+'
_PAIRS_OF = {")": "(", "]": "[", "}": "{"}


def _balanced(s):
    depth = []
    for ch in s:
        if ch in "([{":
            depth.append(ch)
        elif ch in _PAIRS_OF:
            if not depth or depth.pop() != _PAIRS_OF[ch]:
                return False
    return not depth


def _strip_token(t):
    """Sentence punctuation, quotes, unbalanced outer brackets and a pair of brackets around the whole token."""
    while True:
        u = t.strip("'\"`“”‘’").rstrip(".,;:?!")
        if u[-1:] in _PAIRS_OF and not _balanced(u):
            u = u[:-1]
        if u[:1] in "([{" and not _balanced(u):
            u = u[1:]
        if len(u) > 2 and u[0] in "([{" and u[-1] == {"(": ")", "[": "]", "{": "}"}[u[0]] and _balanced(u[1:-1]):
            u = u[1:-1]
        if u == t:
            return t
        t = u


def compact_ids(text):
    """Tokens with an inner ':', '[', ']', '=', '(', ')' or '+', offered whole: ontology CURIEs (UBERON:0002048),
    modified peptides (ALPM[UNIMOD:35]HIR, EM[+15.9949]VNHPGK), SMILES, USIs, URLs, references (ptr:spectrum:7f3a).
    The token keeps balanced brackets, has a letter and 4 or more characters, and is not a path itself, an argument
    written out (num_classes=5) or words joined by '+' (plan+preprocess)."""
    written = set(paths(text))
    out = []
    for m in _TOKEN.finditer(text):
        t = _strip_token(m.group())
        if (len(t) >= 4 and _INNER.search(t) and re.search(r"[A-Za-z]", t) and _balanced(t) and t not in written
                and not t.startswith(("/", "~", "./", "../")) and not _ASSIGNMENT.match(t) and not _WORD_JOIN.match(t)):
            out.append(t)
    return _dedup(out, strip=False)


_ALLELE = re.compile(r"(?<![\w>])([ACGTN]+)\s*>\s*([ACGTN]+)(?![\w>])")
_POSITION_ALLELES = re.compile(r"\bchr\w+[:\s]\s*\d[\d,]*\s+([ACGTN]+)\s+([ACGTN]+)(?![\w>])")


def alleles(text):
    """The two sides of a substitution written X>Y ('chr7:140753336 A>T' -> A, T), or written after a position in the
    VCF order ('chr10 112998590 C T' -> C, T)."""
    found = list(_ALLELE.finditer(text)) + list(_POSITION_ALLELES.finditer(text))
    return _dedup((x for m in found for x in m.groups()), strip=False)


_GROUPED = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+)(?!\d|,\d|\.\d)")
_KILO = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[kK](?!\w)")
_BARE_DECIMAL = re.compile(r"(?<![\w.])\.(\d+)(?!\w|\.\d)")
_HYPHEN_DURATION = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)-(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", re.I)


def number_groups(text, name="", prop=None):
    """Digits grouped by thousands commas ('43,044,295' -> 43044295), k multiples ('50k' -> 50000), decimals without the
    leading zero ('.1' -> 0.1), and for an argument in seconds (name and prop given) a hyphenated duration ('45-minute'
    -> 2700)."""
    out = [m.group(1).replace(",", "") for m in _GROUPED.finditer(text)]
    out += [_num(float(m.group(1)) * 1000) for m in _KILO.finditer(text)]
    out += ["0." + m.group(1) for m in _BARE_DECIMAL.finditer(text)]
    if prop is not None and (_SECONDS_ARG.search(name) or _SECONDS_ARG.search(prop.get("description") or "")):
        out += [_num(float(m.group(1)) * _seconds(m.group(2))) for m in _HYPHEN_DURATION.finditer(text)]
    return _dedup(out, strip=False)


_SPAN_UNITS = {"bp": 1, "kb": 1000, "mb": 1000000}
_LEN = r"(\d+(?:\.\d+)?)\s*(bp|kb|mb)\b"
_POS = r"(?:chr\w+[:\s]\s*)?(\d{1,3}(?:,\d{3})+|\d+)(?![\d,.]\d)"
_SPAN_WORD = r"(?:[\w-]+\s+){0,2}?(?:window|region|interval|span|segment)"
_SPAN_START = re.compile(rf"{_LEN}\s+{_SPAN_WORD}\s+(?:(?:starting|beginning|that\s+starts|which\s+starts)\s+(?:at|from)"
                         rf"|from|at)\s+{_POS}", re.I)
_SPAN_CENTRE = re.compile(rf"{_LEN}\s+{_SPAN_WORD}\s+(?:centred|centered)\s+(?:on|at|around)\s+{_POS}", re.I)
_SPAN_SIDES = re.compile(rf"{_LEN}\s+(?:on\s+)?(?:either|each|both)\s+sides?\s+of\s+{_POS}", re.I)


def span_ends(text):
    """The ends of a span written as a length and a start or centre: 'a 1 kb window starting at chr5:1295000' ->
    1296000; 'a 2 kb interval centred on chr17:43,044,295' -> 43043295, 43045295; '500 bp on either side of
    chr10:112999000' -> 112998500, 112999500."""
    out = []
    for m in _SPAN_START.finditer(text):
        n, p = float(m.group(1)) * _SPAN_UNITS[m.group(2).lower()], int(m.group(3).replace(",", ""))
        out.append(_num(p + n))
    for m in _SPAN_CENTRE.finditer(text):
        n, p = float(m.group(1)) * _SPAN_UNITS[m.group(2).lower()], int(m.group(3).replace(",", ""))
        out += [_num(p - n / 2), _num(p + n / 2)]
    for m in _SPAN_SIDES.finditer(text):
        n, p = float(m.group(1)) * _SPAN_UNITS[m.group(2).lower()], int(m.group(3).replace(",", ""))
        out += [_num(p - n), _num(p + n)]
    return _dedup(out, strip=False)


_BARE_NAME = re.compile(r"(?<![\w/.~-])([\w-]+(?:\.[\w-]+)*\.([A-Za-z][A-Za-z0-9]{1,7}))(?![\w/-]|\.\w)")
DATA_EXTENSIONS = set("""csv tsv txt json jsonl yaml yml xml html htm pdf md log parquet feather arrow h5 hdf5 hdf h5ad h5mu
    loom zarr mat npy npz rds rdata pkl pickle joblib xls xlsx ods sav dta nii gz nrrd mha mhd dcm png jpg jpeg tif tiff
    bmp gif svg svs ndpi czi fasta fa fna faa fastq fq bam sam cram vcf bcf bed bigwig bw bedgraph gff gff3 gtf mzml mzxml
    mgf mztab mzid raw wiff pepxml idxml msp pdb cif mmcif sdf mol mol2 smi pt pth ckpt onnx safetensors keras pb bin zip
    tar tgz bz2 xz""".split())


def bare_files(text):
    """File names without a folder whose extension is a common data-file extension (DATA_EXTENSIONS: tables, arrays,
    single-cell, sequence, spectra, structures, images, reports, models, archives): 'heart_norm.h5ad', 'sample_A.raw'.
    paths() knows only a short list of extensions; a dotted column name ('os.status') is not a file."""
    written = paths(text)
    return _dedup((m.group(1) for m in _BARE_NAME.finditer(text) if m.group(1) not in written
                   and m.group(2).lower() in DATA_EXTENSIONS), strip=False)


_ANY_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
_SAME_STEM = re.compile(r"\bsame\s+(?:file\s*)?(?:stem|base\s*name|basename|file\s*name|filename|name)\b", re.I)
_FORMAT_WORD = re.compile(r"\b(?:to|into|as)\s+([A-Za-z][A-Za-z0-9]{1,7})\b", re.I)


def renamed_outputs(text, same_place):
    """Output files named from written ones. same_place (the request says the same folder or directory): each bare file
    name in each written folder and in the folder of each written file. 'same stem / base name / file name': each
    written file's stem with the format word after 'to', 'into' or 'as' ('Y_KO_14.raw ... to mzML' -> Y_KO_14.mzML), in
    its own folder and in each written folder."""
    written = paths(text)
    files = [p for p in written if _ANY_EXTENSION.search(os.path.basename(p))]
    folders = [p.rstrip("/") for p in written if p not in files and p.rstrip("/")]
    out = []
    if same_place:
        for n in bare_files(text):
            out += [f"{d}/{n}" for d in folders + [os.path.dirname(f) for f in files] if d]
    if _SAME_STEM.search(text):
        fmts = [w for w in _FORMAT_WORD.findall(text) if w.lower() not in _STOP and w.lower() not in COMMAND_WORDS]
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            out += [f"{d}/{stem}.{e}" for e in fmts for d in [os.path.dirname(f)] + folders if d]
    return _dedup(out, strip=False)


def written_arrays(text):
    """The bracketed lists of lists the request writes ('[[94.065, 35], [146.06, 100]]'), parsed."""
    out = []
    for m in re.finditer(r"\[\s*\[", text):
        depth = 0
        for i in range(m.start(), len(text)):
            depth += {"[": 1, "]": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    v = ast.literal_eval(text[m.start():i + 1])
                except (ValueError, SyntaxError):
                    v = None
                if isinstance(v, list) and v and all(isinstance(x, list) for x in v) and v not in out:
                    out.append(v)
                break
    return out


_BOOL_OP = re.compile(r"\s(?:AND|OR|NOT)\s")
_LEAD = {"search", "query", "find", "for", "about", "on"}


def boolean_queries(text):
    """A search expression joined by upper-case AND, OR or NOT, whole ('long COVID AND fatigue'): the clause it is
    written in, without leading or trailing function words."""
    out = []
    for chunk in re.split(r"[,;:()\n]|\.(?:\s|$)", text):
        if not _BOOL_OP.search(f" {chunk.strip()} "):
            continue
        ws = chunk.split()
        drop = _STOP | COMMAND_WORDS | _LEAD
        while ws and ws[0].lower() in drop:
            ws.pop(0)
        while ws and ws[-1].lower() in drop:
            ws.pop()
        if 3 <= len(ws) <= 12:
            out.append(" ".join(ws))
    return _dedup(out, strip=False)


_COLON = re.compile(r":\s+")


def colon_blocks(text, description=""):
    """The text after a colon that runs to the end of the request as one statement (no sentence end inside, 3 or more
    words), whole, when the argument's description names its first word: 'run: SELECT a, COUNT(*) FROM t' for an
    argument described as 'A read-only SELECT ...'."""
    named = {w.lower() for w in re.findall(r"[A-Za-z]+", description)}
    out = []
    for m in _COLON.finditer(text):
        block = text[m.end():].strip()
        block = block[:-1] if block.endswith(".") else block
        if (len(block.split()) >= 3 and not re.search(r"[.!?]\s", block) and "\n" not in block
                and re.match(r"[A-Za-z]+", block) and re.match(r"[A-Za-z]+", block).group().lower() in named):
            out.append(block)
    return _dedup(out, strip=False)


_POSSESSIVE = re.compile(r"\b([A-Za-z][\w-]*['’]s)\s+([A-Za-z][\w-]*)")


def possessive_phrases(text):
    """Two-word phrases that keep a possessive ("Crohn's disease"); phrases() drops the apostrophe."""
    return _dedup((f"{a} {b}" for a, b in _POSSESSIVE.findall(text) if b.lower() not in _STOP), strip=False)


_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")


def proper_names(text):
    """Runs of two or three capitalised words ('South Korea', 'New York')."""
    return _dedup((m.group(1) for m in _PROPER.finditer(text)), strip=False)


_MODIFIED = re.compile(r"\b([A-Za-z][A-Za-z0-9]+)-(?:weighted|based|wise|level|specific)\b|\b[Pp]er-([A-Za-z][A-Za-z0-9]+)\b")


def compound_parts(text):
    """The word of a hyphenated word made of a word and a modifier: 'Intensity-weighted' -> Intensity, 'Per-peptidoform'
    -> peptidoform ('-weighted', '-based', '-wise', '-level', '-specific', 'per-')."""
    return _dedup((a or b for a, b in _MODIFIED.findall(text)), strip=False)


EACH_WAY = re.compile(r"\b(?:each\s+way|both\s+ways|in\s+(?:both|each)\s+directions?|for\s+both)\b", re.I)


def grouped_lists(text):
    """Groups written 'A x, y; B z, w' (two or more, separated by semicolons) as one dictionary of lists {A: [x, y], B:
    [z, w]}: in each group the last word before the first comma is the first item, the words before it the key (at most
    three words); every item is one token and not a path."""
    out = []
    for seg in re.split(r"(?<=[.!?])\s+|:\s+", text):
        groups = [g.strip().rstrip(".") for g in seg.split(";")]
        if len(groups) < 2:
            continue
        d = {}
        for g in groups:
            parts = [x.strip() for x in re.split(r",|\s+and\s+", g) if x.strip()]
            head = parts[0].split() if parts else []
            items = [head[-1]] + parts[1:] if len(head) >= 2 else []
            if not 2 <= len(head) <= 4 or any(len(x.split()) > 1 or "/" in x for x in items):
                d = None
                break
            d[" ".join(head[:-1])] = items
        if d and len(d) >= 2 and d not in out:
            out.append(d)
    return out


_CHROM_WORD = re.compile(r"\bchrom(?:osome)?\s+(\d{1,2}|[XY]|MT|M)\b", re.I)
_CHROM_POS = re.compile(r"(?<![\w:.,])(\d{1,2}|[XY]|MT|M):(\d{1,3}(?:,\d{3})+|\d{5,})(?!\d)")


def chromosome_names(text):
    """A chromosome written without 'chr', with it: 'chromosome 7' -> chr7, '19:44908822:C:T' -> chr19,
    '1:55,039,548-55,064,852' -> chr1 (a position N:P with five or more digits or thousands commas)."""
    out = [(m.start(), "chr" + m.group(1).upper()) for m in _CHROM_WORD.finditer(text)]
    out += [(m.start(), "chr" + m.group(1)) for m in _CHROM_POS.finditer(text)]
    return _dedup((v for _, v in sorted(out)), strip=False)


def chromosome_counts(text):
    """How many times each chromosome is written without 'chr' (a value written twice may fill two arguments)."""
    counts = {}
    for v in ["chr" + m.group(1).upper() for m in _CHROM_WORD.finditer(text)] + \
             ["chr" + m.group(1) for m in _CHROM_POS.finditer(text)]:
        counts[v] = counts.get(v, 0) + 1
    return counts


_COLON_VARIANT = re.compile(r"(?<![\w:])(?:chr)?(?:\d{1,2}|[XY]|MT|M):\d[\d,]*:([ACGTN]+):([ACGTN]+)(?![\w:])")
_SLASH_VARIANT = re.compile(r"(?<![\w.])\d[\d,]*\s+([ACGTN]+)/([ACGTN]+)(?![\w/])")
_NAMED_ALLELE = re.compile(r"\b(?i:ref|reference|alt|alternate|alternative)(?:\s+(?i:allele|base|bases|nucleotide))?"
                           r"\s*[:=]?\s+([ACGTN]+)(?![\w])")


def variant_alleles(text):
    """A variant's bases written colon-joined ('19:44908822:C:T' -> C, T), slash-joined after a position ('chr11
    5227002 T/A' -> T, A) or after ref/reference/alt/alternate ('reference allele G' -> G)."""
    out = [(m.start(i), m.group(i)) for p in (_COLON_VARIANT, _SLASH_VARIANT) for m in p.finditer(text) for i in (1, 2)]
    out += [(m.start(1), m.group(1)) for m in _NAMED_ALLELE.finditer(text)]
    return _dedup((v for _, v in sorted(out)), strip=False)


_ON_SIGN = re.compile(r"\b(?:on|strand)\s+(?:the\s+)?([+-])(?=$|[\s,;.)])", re.I)
_AND_SIGN = re.compile(r"\b(?:on|strand)\s+(?:the\s+)?[+-]\s*(?:and|or|,|/)\s*([+-])(?=$|[\s,;.)])", re.I)
_SIGN_STRAND = re.compile(r"(?<![\w+-])([+-])\s+strands?\b", re.I)
_STRAND_LIST = re.compile(r"\bstrands?\s*:\s*([+\-.](?:\s*[;,]\s*[+\-.])+)", re.I)
_STRAND_WORD = re.compile(r"\b(plus|forward|positive|minus|reverse|negative)\s+strand\b|\b(unstranded)\b", re.I)
_STRAND_SIGN = {"plus": "+", "forward": "+", "positive": "+", "minus": "-", "reverse": "-", "negative": "-",
                "unstranded": "."}


def strand_marks(text):
    """Strand marks '+', '-', '.': a sign alone after 'on' or 'strand' ('on +', 'on the + and - strands'), or before
    'strand(s)', the items of a list after 'strands:' ('strands: .; .; +', each as often as written), and strand words
    (plus/forward/positive strand -> '+', minus/reverse/negative strand -> '-', unstranded -> '.')."""
    out = [(m.start(1), m.group(1)) for p in (_ON_SIGN, _AND_SIGN, _SIGN_STRAND) for m in p.finditer(text)]
    for m in _STRAND_LIST.finditer(text):
        out += [(m.start(1) + i, x) for i, x in enumerate(re.split(r"\s*[;,]\s*", m.group(1).strip()))]
    out += [(m.start(), _STRAND_SIGN[(m.group(1) or m.group(2)).lower()]) for m in _STRAND_WORD.finditer(text)]
    return _dedup((v for _, v in sorted(out)), strip=False)


SHARED = re.compile(r"\bboth\b|\b(?:on|for|in|per)\s+each\b", re.I)    # a value said to apply to several arguments

_LEN_FACTOR = {"bp": 1, "kb": 1000, "kbp": 1000, "kilobase": 1000, "kilobases": 1000, "mb": 1000000, "mbp": 1000000,
               "megabase": 1000000, "megabases": 1000000}
_LEN_UNIT = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(kbp|kb|mbp|mb|kilobases?|megabases?)\b", re.I)
_LEN_RANGE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:-|–|to|and)\s*(\d+(?:\.\d+)?)\s*(kbp|kb|mbp|mb|kilobases|megabases)\b",
                        re.I)
_PM_WINDOW = re.compile(r"(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*(bp|kb|mb)\s+(?:of|around|from|about|either\s+side\s+of)\s+"
                        + _POS, re.I)


def length_units(text):
    """Lengths in kb or Mb, in bp ('2 kb' -> 2000, '+5 kb' -> 5000), both ends of a range with one unit ('from 27.0 to
    28.0 Mb' -> 27000000, 28000000), and the ends of a window written with ± ('±50 kb of chr2:135,851,076' ->
    135801076, 135901076)."""
    out = [_num(float(m.group(1)) * _LEN_FACTOR[m.group(2).lower()]) for m in _LEN_UNIT.finditer(text)]
    for m in _LEN_RANGE.finditer(text):
        f = _LEN_FACTOR[m.group(3).lower()]
        out += [_num(float(m.group(1)) * f), _num(float(m.group(2)) * f)]
    for m in _PM_WINDOW.finditer(text):
        n, p = float(m.group(1)) * _LEN_FACTOR[m.group(2).lower()], int(m.group(3).replace(",", ""))
        out += [_num(p - n), _num(p + n)]
    return _dedup(out, strip=False)


_FLAT_LIST = re.compile(r"\[([^\[\]]+)\]")
_LIST_GAP = re.compile(r"\s*(?:,|;|and|or|\n)?\s*$", re.I)


def _item(x):
    x = x.strip().strip("'\"`")
    return _number(x) if re.fullmatch(r"-?\d+(?:\.\d+)?", x) else x


def bracket_lists(text):
    """Bracketed lists written one after another ('[chr1, 109274000, 109275500, SORT1_prom] and [chr1, ...]'), offered
    together as one list of lists, and each alone as a list of one; unquoted words are text, numbers numbers. A bracket
    with one item (AREA[StudyType], EM[+15.99]) is not a list."""
    groups, cur, end = [], [], None
    for m in _FLAT_LIST.finditer(text):
        items = [x for x in m.group(1).split(",")]
        if len(items) < 2 or not all(x.strip() for x in items):
            continue
        lst = [_item(x) for x in items]
        if cur and end is not None and _LIST_GAP.fullmatch(text[end:m.start()]):
            cur.append(lst)
        else:
            if cur:
                groups.append(cur)
            cur = [lst]
        end = m.end()
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        for v in ([g] if len(g) > 1 else []) + [[x] for x in g]:
            if v not in out:
                out.append(v)
    return out


_ROW_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def number_rows(text):
    """Lines holding two numbers each, short labels allowed between them ('m/z 85.028  int 40', '57.07 15'), as
    [[a, b], ...]; a line without numbers ('query', 'reference') or any other line ends a block. Blocks of two or more
    rows are offered."""
    blocks, cur = [], []
    for line in text.splitlines() + [""]:
        nums = _ROW_NUMBER.findall(line)
        rest = _ROW_NUMBER.sub(" ", line).split()
        if len(nums) == 2 and len(rest) <= 2 and all(len(w) <= 9 for w in rest):
            cur.append([_number(n) for n in nums])
            continue
        if len(cur) >= 2:
            blocks.append(cur)
        cur = []
    return blocks


def object_arrays(text):
    """The JSON arrays of objects the request writes ('[{"usi": "mzspec:..."}, {...}]'), parsed."""
    out = []
    for m in re.finditer(r"\[\s*\{", text):
        depth = 0
        for i in range(m.start(), len(text)):
            depth += {"[": 1, "]": -1}.get(text[i], 0)
            if depth == 0:
                for parse in (json.loads, ast.literal_eval):
                    try:
                        v = parse(text[m.start():i + 1])
                    except (ValueError, SyntaxError, TypeError):
                        continue
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v) and v not in out:
                        out.append(v)
                    break
                break
    return out


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_VERSION = re.compile(r"(?<![\w.])(v?\d+\.\d+\.\d+(?:\.\d+)?|v\d+\.\d+)(?![\w]|\.\d)")
_RESIDUE = re.compile(r"\b([A-Z][A-Za-z]+)\s+\(([A-Z]{1,3})\)")
_LETTER_SERIES = re.compile(r"(?<![\w/])([A-Za-z](?:/[A-Za-z])+)(?![\w/])")
_NOT_SERIES = {"w/o", "n/a", "w/"}


def joined_tokens(text):
    """Offered whole: parts joined by commas without spaces, each with a letter and a digit ('L858R,T790M'); dotted
    versions ('4.2.1', 'v2.0'); a word with one to three capitals in brackets ('Carbamidomethyl (C)'); tokens with
    superscript digits read as digits ('MS²PIP' -> MS2PIP); single letters joined by '/' ('b/y' -> by)."""
    out = []
    for m in _TOKEN.finditer(text):
        t = m.group().strip("'\"`“”‘’()").rstrip(".;:?!")
        parts = t.split(",")
        if len(parts) >= 2 and all(re.fullmatch(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+", p) for p in parts):
            out.append(t)
        if t.translate(_SUPERSCRIPT) != t:
            out.append(t.translate(_SUPERSCRIPT).rstrip(","))
    out += [m.group(1) for m in _VERSION.finditer(text)]
    out += [f"{a} ({b})" for a, b in _RESIDUE.findall(text)]
    out += [m.group(1).replace("/", "") for m in _LETTER_SERIES.finditer(text) if m.group(1).lower() not in _NOT_SERIES]
    return _dedup(out, strip=False)


# ---- search expressions ------------------------------------------------------------------------------------------------
_SEARCH_DROP = _STOP | COMMAND_WORDS | FILLER_WORDS
_CHUNK_SPLIT = re.compile(r"[,;:()\[\]\n.!?\"`]|\s[-–—]\s|^\s*[-*•]\s|\b(?:with|who|whose|which|that|for|in|at|from|into|on|"
                          r"and|or|but|near|than|as|by|to|about|matching|based|lives?|living)\b", re.I | re.M)
_SEARCH_WORD = re.compile(r"[A-Za-z0-9][\w'’+-]*")


def _trim(ws):
    while ws and ws[0].lower() in _SEARCH_DROP:
        ws = ws[1:]
    while ws and ws[-1].lower() in _SEARCH_DROP:
        ws = ws[:-1]
    return ws


def search_chunks(text):
    """For search arguments: phrases of up to three words that hold a token with a digit ('type 1 diabetes', 'KRAS G12C
    inhibitors'), and the runs of two to six words between punctuation, line breaks and function words, whole
    ('metastatic castration-resistant prostate cancer', 'spinal muscular atrophy type 1')."""
    masked = text
    for p in paths(text):
        if "/" in p:
            masked = masked.replace(p, " , ")
    out = []
    for chunk in _CHUNK_SPLIT.split(masked):
        if chunk is None:
            continue
        ws = _trim(_SEARCH_WORD.findall(chunk))
        if 2 <= len(ws) <= 6:
            out.append(" ".join(ws))
        kept = [w for w in ws if w.lower() not in _SEARCH_DROP]
        for n in (2, 3):
            for i in range(len(kept) - n + 1):
                g = kept[i:i + n]
                if any(re.search(r"\d", w) for w in g) and not re.fullmatch(r"[\d.,]+", g[0]) or (
                        any(re.search(r"\d", w) for w in g[1:-1])):
                    out.append(" ".join(g))
    return _dedup(out, strip=False)


_BOOL_TERM = r"(?:\([^()]*\)|[^\s\[\]()]*(?:\[[^\]]*\][^\s\[\]()]*)+|[A-Za-z0-9][\w.'-]*)"
_BOOL_EXPR = re.compile(rf"{_BOOL_TERM}(?:\s+(?:AND|OR|NOT)\s+{_BOOL_TERM})+")


def bracket_queries(text):
    """Boolean expressions whose terms include a bracketed group or a token with brackets, whole ('(psilocybin OR MDMA)
    AND depression', 'AREA[StudyType]INTERVENTIONAL AND AREA[EnrollmentCount]RANGE[200, MAX]')."""
    return _dedup((m.group(0) for m in _BOOL_EXPR.finditer(text) if re.search(r"[()\[\]]", m.group(0))), strip=False)


# ---- numbers in notations of their own ---------------------------------------------------------------------------------
_AGE = re.compile(r"(?<![\w.])(\d{1,3})\s?(?:F|M|yo|y/o|yrs?\s+old)(?![\w/])")


def age_notation(text):
    """'29F', '45M', '62yo' -> the number."""
    return _dedup((m.group(1) for m in _AGE.finditer(text)), strip=False)


def spaced_paths(text):
    """Paths that contain spaces, whole: quoted spans with a '/' whose last part has a file extension or that end in '/'
    ('urine study/batch2/U_014.mzML'), and unquoted paths starting with '/' or '~/' followed by up to three words and a
    relative token with a '/' ending in a data-file extension ('/lab/share/QC/2024-11 HeLa QC/psms.parquet'); a token
    that starts a path of its own ('/', '~', '.') does not continue one."""
    out = [q for q in quoted(text)
           if " " in q.strip() and "/" in q and (q.endswith("/") or _ANY_EXTENSION.search(q.rsplit("/", 1)[-1]))]
    toks = text.split()
    for i, t in enumerate(toks):
        if not t.startswith(("/", "~/")) or _has_data_extension(t.rstrip(".,;:?!)")):
            continue
        for j in range(i + 1, min(i + 5, len(toks))):
            last = toks[j].rstrip(".,;:?!)")
            if "/" in last:
                mids = toks[i + 1:j]
                if mids and _has_data_extension(last) and not last.startswith(("/", "~", ".")) \
                        and not any(w.lower() in _STOP or "/" in w for w in mids):
                    out.append(" ".join([t] + mids + [last]))
                break
            if toks[j].lower() in _STOP or toks[j][-1:] in ".,;:?!":
                break
    return _dedup(out, strip=False)


def _has_data_extension(t):
    m = re.search(r"\.([A-Za-z][A-Za-z0-9]{0,7})(?:\.gz)?$", t)
    return bool(m) and m.group(1).lower() in DATA_EXTENSIONS


_NAMED_AFTER = re.compile(r"\b(?:called|named|column|field)\s+[`'\"]?([A-Za-z_][\w./-]*)", re.I)


def named_words(text):
    """Words written right after called, named, column or field ('the RT column rt' -> rt, 'is called mz_obs' ->
    mz_obs), even when they are words of an argument's name."""
    return _dedup((m.group(1) for m in _NAMED_AFTER.finditer(text)
                   if m.group(1).lower() not in _STOP | COMMAND_WORDS | {"is", "was", "are"}), strip=True)


_SCALE_WORDS = {"thousand": 1000, "million": 1000000, "billion": 1000000000}
_SCALED = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s+(thousand|million|billion)\b", re.I)
_ORDINAL_NUM = re.compile(r"(?<![\w.])(\d+)(?:st|nd|rd|th)\b")
_SLASH_NUMBERS = re.compile(r"(?<![\w./])(-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?){2,})(?![\w/])")
_ONE_UNIT = re.compile(r"\b(?:a|an|one)\s+(?:full\s+|whole\s+|single\s+)?(minute|hour|second)\b", re.I)


def number_scales(text, name="", prop=None):
    """Digits with thousand, million or billion ('2 million' -> 2000000); ordinals ('95th' -> 95); three or more numbers
    joined by '/' ('0.1/0.25/0.5' -> each); for an argument in seconds (name and prop given), 'a minute', 'an hour', 'a
    full minute' -> 60, 3600."""
    out = [_num(float(m.group(1)) * _SCALE_WORDS[m.group(2).lower()]) for m in _SCALED.finditer(text)]
    out += [m.group(1) for m in _ORDINAL_NUM.finditer(text)]
    out += [x for m in _SLASH_NUMBERS.finditer(text) for x in m.group(1).split("/")]
    if prop is not None and (_SECONDS_ARG.search(name) or _SECONDS_ARG.search(prop.get("description") or "")):
        out += [str(_seconds(m.group(1))) for m in _ONE_UNIT.finditer(text)]
    return _dedup(out, strip=False)


_MILLI = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mDa|kDa|ms|millidaltons?|milliseconds?)\b")
_MILLI_FACTOR = {"mda": (0.001, "Da"), "kda": (1000, "Da"), "ms": (0.001, "s"), "millidalton": (0.001, "Da"),
                 "millidaltons": (0.001, "Da"), "millisecond": (0.001, "s"), "milliseconds": (0.001, "s")}


def milli_units(text):
    """A number with mDa, kDa or ms in Da or s ('10 mDa' -> 0.01). -> (numbers, base unit symbols)."""
    found = [(float(m.group(1)), _MILLI_FACTOR[m.group(2).lower()]) for m in _MILLI.finditer(text)]
    return (_dedup((_num(v * f) for v, (f, _) in found), strip=False), _dedup((u for _, (_, u) in found), strip=False))


_UNIT_ABBREV = re.compile(r"(?:\(\s*|\bin\s+)(mins?|hrs?|secs?)\b(?!['’])", re.I)
_UNIT_NAME_OF = {"min": "minutes", "mins": "minutes", "hr": "hours", "hrs": "hours", "h": "hours", "sec": "seconds",
                 "secs": "seconds", "s": "seconds"}
_UNIT_SYMBOLS = [(re.compile(r"\bparts\s+per\s+million\b", re.I), "ppm"), (re.compile(r"\bparts\s+per\s+billion\b", re.I), "ppb"),
                 (re.compile(r"\bmillidaltons?\b", re.I), "mDa"), (re.compile(r"\bdaltons?\b", re.I), "Da"),
                 (re.compile(r"\bbase\s+pairs?\b", re.I), "bp"), (re.compile(r"\bkilobases?\b", re.I), "kb")]


def unit_words(text):
    """Unit abbreviations of two letters or more after a bracket or 'in' give the unit's name ('(min)', 'RT in min' ->
    minutes; '(s)' is not one); unit names give the symbol ('parts per million' -> ppm, daltons -> Da, base pairs ->
    bp)."""
    out = [_UNIT_NAME_OF[m.group(1).lower()] for m in _UNIT_ABBREV.finditer(text)]
    out += [sym for p, sym in _UNIT_SYMBOLS if p.search(text)]
    return _dedup(out, strip=False)


_CHARGE = re.compile(r"(?<![\w.])(\d)\s?[+-](?![\w\d])")
_CHARGED = re.compile(r"\b(singly|doubly|triply|quadruply)\s+(?:charged|protonated)\b", re.I)
_TIMES = {"singly": 1, "doubly": 2, "triply": 3, "quadruply": 4}


def charge_forms(text):
    """Charges '3+', '2-' -> 3, 2; singly, doubly, triply or quadruply charged -> 1 to 4."""
    out = [m.group(1) for m in _CHARGE.finditer(text)] + [str(_TIMES[m.group(1).lower()]) for m in _CHARGED.finditer(text)]
    return _dedup(out, strip=False)


_ARROW_FORMAT = re.compile(r"(?:->|→)\s*([A-Za-z][A-Za-z0-9]{1,7})\b")
_LONG_EXTENSION = re.compile(r"(?<![\w/~.])(\.[A-Za-z][A-Za-z0-9]{1,7})(?![\w/])")


def renamed_outputs_ext(text, sources=()):
    """Same-stem outputs (renamed_outputs) with more sources of the new extension: a written extension ('.html
    extension', 'but .feather'), a format word after '->' or '→' ('-> mzML'), and the word after to/into/as; the input's
    own extension is not reused. Source files: the written files and `sources` (spaced_paths); each output is placed in
    its source's folder and in each written folder."""
    if not _SAME_STEM.search(text):
        return []
    written = paths(text)
    files = [p for p in written if _ANY_EXTENSION.search(os.path.basename(p))] + list(sources)
    folders = [p.rstrip("/") for p in written if p not in files and p.rstrip("/") and not _ANY_EXTENSION.search(p)
               and not any(p in s for s in sources)]
    masked = text
    for p in written + list(sources):
        masked = masked.replace(p, " ")
    exts = _LONG_EXTENSION.findall(masked) + ["." + w for w in _ARROW_FORMAT.findall(text)]
    exts += ["." + w for w in _FORMAT_WORD.findall(text) if w.lower() not in _STOP | COMMAND_WORDS | set(_UNITS) | set(_TENS)]
    out = []
    for f in files:
        stem, own = os.path.splitext(os.path.basename(f))
        for e in _dedup(exts, strip=False):
            if e.lower() == own.lower():
                continue
            out += [f"{d}/{stem}{e}" for d in [os.path.dirname(f)] + folders if d]
    return _dedup(out, strip=False)


_SPLIT = re.compile(r"(?<![\w./])(\d{1,2})\s*[/:]\s*(\d{1,2})(?![\w/.])")


def split_fractions(text):
    """'A/B' or 'A:B' with A + B = 100 ('70/30 train-test split') -> A/100, B/100."""
    return _dedup((_num(int(x) / 100) for m in _SPLIT.finditer(text) if int(m.group(1)) + int(m.group(2)) == 100
                   for x in m.groups()), strip=False)


_COMPLEMENT_DESC = re.compile(r"\b1\s*[-−–]\s*[A-Za-z]")


def complement_levels(text, prop):
    """For an argument whose description defines a complement ('1-alpha coverage'), a percentage p -> 1 - p/100 ('90%
    prediction intervals' -> 0.1)."""
    if not _COMPLEMENT_DESC.search(prop.get("description") or ""):
        return []
    return _dedup((_num(1 - float(m.group(1)) / 100) for m in _PERCENT.finditer(text) if float(m.group(1)) <= 100),
                  strip=False)


_LEVEL_TOKEN = re.compile(r"\b[A-Z]{2}(\d)\b")
_LEVEL_WORD = re.compile(r"level", re.I)


def level_digits(text, name, prop):
    """'MS1', 'MS2' (two capitals and one digit) -> the digit, for an argument whose name or description says level."""
    if not (_LEVEL_WORD.search(name) or _LEVEL_WORD.search(prop.get("description") or "")):
        return []
    return _dedup((m.group(1) for m in _LEVEL_TOKEN.finditer(text)), strip=False)


_GROUP_LINE = re.compile(r"^\s*[-*•]?\s*([A-Za-z][\w' -]{0,40}?)\s*:\s*(\S.*?)\s*$")
_GROUP_ITEM = re.compile(r"[\w.+-]+")
_BRACKET_GROUP = re.compile(r"\(([^()]+)\)")


def _group_items(s):
    items = [x.strip() for x in re.split(r"[,;]|\s+and\s+", s) if x.strip()]
    return items if items and all(_GROUP_ITEM.fullmatch(x) for x in items) else None


def grouped_lines(text):
    """Dictionaries of lists written one group per line ('T cells: CD3D, CD3E', two or more lines) or as 'key (a; b),
    key2 (c)' (two or more groups): {key: [items]}. Keys have up to three words, items are one token each."""
    out, d = [], {}
    for line in text.splitlines() + [""]:
        m = _GROUP_LINE.match(line)
        items = _group_items(m.group(2)) if m else None
        if m and items and len(m.group(1).split()) <= 3:
            d[m.group(1).strip()] = items
            continue
        if len(d) >= 2 and d not in out:
            out.append(d)
        d = {}
    d, last = {}, None
    for m in _BRACKET_GROUP.finditer(text):
        start = max(text.rfind(c, 0, m.start()) for c in "),;:-\n") + 1
        gap = text[last:start] if last is not None else None
        words = text[start:m.start()].split()
        if words and words[0].lower() == "and":
            words = words[1:]
        items = _group_items(m.group(1))
        ok = items and 1 <= len(words) <= 3 and all(re.fullmatch(r"[A-Za-z][\w'-]*", w) for w in words)
        if ok and (last is None or not d or re.fullmatch(r"\s*,?\s*", gap or "")):
            d[" ".join(words)] = items
        else:
            if len(d) >= 2 and d not in out:
                out.append(d)
            d = {" ".join(words): items} if ok else {}
        last = m.end()
    if len(d) >= 2 and d not in out:
        out.append(d)
    return out
