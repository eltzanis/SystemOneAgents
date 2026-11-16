# Review guide: are these argument values grounded or invented?

## The task

Each line of `sample.jsonl` is one argument value that an AI agent put into a tool call. It holds:

- `request`: the user's request;
- `tool` and `tool_description`: the tool the agent called;
- `argument` and `argument_schema`: the argument and its schema (type, description, enum, default);
- `value`: the value the agent sent.

For a list or an object, `value` is one item, key or value of it.

An automatic check found that the value does not occur in the request or the schema. Your job is to decide, for each
line, whether that check was right.

## Verdicts (one per line)

- `grounded`: the value is stated in the request, or follows directly and unambiguously from what the request or
  the schema says. That covers:
  - the same value written differently: case, spacing, punctuation, singular/plural, a reordered or abbreviated
    phrase, an equivalent number format, the same date in another format;
  - a unit conversion of a stated quantity;
  - the canonical code or name of something the request names, such as "random forest" → `rf`, or a gene name →
    the identifier format the schema asks for;
  - a value listed or documented in the tool or argument description;
  - an item that is part of what the request names.
- `invented`: the value is neither stated nor directly derivable. That covers:
  - a guess or a "sensible" setting the request does not ask for (for example a batch size, a learning rate or a
    file path the request never mentions);
  - an identifier, path, name or number that appears nowhere in the request or the schema;
  - a value that only fits if one assumes facts the request does not give.
- `unclear`: you cannot decide from the information given.

Judge only whether the value is grounded. Do not judge whether it is the right value for the argument, and do not
judge the agent.

## Rules

1. Work only in this folder. Open no other file or folder, do not use the web, and run no model.
2. Write `verdicts.jsonl`, one line per sample line: `{"n": <n>, "verdict": "grounded|invented|unclear", "kind": "<optional: short category such as unstated setting, reformatted value, invented path>"}`.
   Write no request text or value into it.
3. Then write `REVIEW_SUMMARY.md` with counts only: the verdicts, the categories, and a short paragraph on how you
   judged and any line you found hard. Describe hard lines in general terms only, without quoting request text or
   values.
4. Your final message: the same counts. No request text, no values.
