"""Verdicts of one tool call against the labelled request, the same for every agent.

A labelled request is {"id", "request", "tool": "server/tool" | null, "args": {only the values the request states},
"ask": [required arguments the request leaves out], ...}. The scorer judges only the values the request states: a value
the request does not state is never an error. For the call of the right tool, `argument_verdicts` gives

  args_correct            per labelled argument: the call's value equals the label after normalisation (case, number
                          form, a comma list against a JSON list), or the call leaves it out and the label is the
                          argument's documented default
  args_equivalent         also right when the server maps both values to the same thing (s1eval.equivalence)
  args_conditional        labelled arguments the call leaves out that the server may auto-detect (reported, not counted)
  call_exact              every value strictly right and nothing extra or missing (strict, for reference)
  call_exact_equivalent   the same with server-equivalent values
  call_exact_conditional  the same, auto-detected omissions allowed
  call_correct            the main verdict: every stated value right (server-equivalent or auto-detected) and the
                          required arguments left missing are exactly the request's `ask` (none for a complete request):
                          the agent asks for what the request leaves out instead of inventing it

A request is fully correct when the right tool is called with call_correct, or, out of scope, when no tool is called.
"""


def _norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 8)
    if isinstance(v, list):
        return frozenset(_norm(x) for x in v)
    if isinstance(v, dict):                             # object arguments (e.g. DICOM tag filters): key by key
        return frozenset((str(k).strip().lower(), _norm(x)) for k, x in v.items())
    s = str(v).strip()
    try:
        return round(float(s), 8)
    except ValueError:
        pass
    if "," in s:
        return frozenset(x.strip().lower() for x in s.split(",") if x.strip())
    return s.lower()


def same(pred, gold):
    """Equal after normalisation; a comma list equals a JSON list with the same items; '0' equals 0 for string folds."""
    p, g = _norm(pred), _norm(gold)
    if isinstance(g, frozenset) and not isinstance(p, frozenset) and p is not None:
        p = frozenset([p]) if not isinstance(p, float) else p
    if isinstance(p, frozenset) and not isinstance(g, frozenset):
        g = frozenset([g])
    return p == g


def argument_verdicts(tool, props, gold, args, extra, missing, ask=()):
    """Strict, server-equivalent and conditional verdicts for the labelled arguments of one call of `tool`.
    props: the tool's inputSchema properties; gold: the labelled values; args: the call's arguments; extra: arguments
    the call sets that the label does not have; missing: required arguments without a value; ask: the required
    arguments the request leaves out (the main verdict wants exactly these missing)."""
    from s1agent.compiler import is_default
    from s1eval import equivalence as E
    strict = {k: same(args.get(k), v) or (k not in args and is_default(v, props.get(k, {}))) for k, v in gold.items()}
    equiv = {k: ok or (k in args and E.equivalent(tool, k, args[k], gold[k])) for k, ok in strict.items()}
    conditional = sorted(k for k, ok in equiv.items() if not ok and k not in args and E.autodetected(tool, k, gold[k]))
    clean = not extra and not missing
    stated = all(equiv[k] or k in conditional for k in equiv)
    return dict(args_correct=strict, args_equivalent=equiv, args_conditional=conditional,
                call_exact=all(strict.values()) and clean, call_exact_equivalent=all(equiv.values()) and clean,
                call_exact_conditional=stated and clean, call_correct=stated and set(missing) == set(ask))
