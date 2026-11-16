# Review summary

200 sample lines reviewed; one verdict per line in `verdicts.jsonl`.

## Verdicts

| Verdict  | Count |
|----------|------:|
| grounded |    16 |
| invented |   148 |
| unclear  |    36 |

## Categories

| Verdict  | Category                                   | Count |
|----------|--------------------------------------------|------:|
| grounded | item of named set or scale                 |     5 |
| grounded | canonical name from abbreviation           |     5 |
| grounded | computed from stated values                |     2 |
| grounded | reformatted value                          |     2 |
| grounded | rephrased request terms                    |     2 |
| invented | unstated hyperparameter                    |    76 |
| invented | invented path or file name                 |    29 |
| invented | unstated option                            |    19 |
| invented | assumed domain fact                        |    11 |
| invented | invented name or label                     |     6 |
| invented | mis-derived stated quantity                |     5 |
| invented | invented data value                        |     2 |
| unclear  | refers to unseen earlier content           |    29 |
| unclear  | path of referenced but unspecified object  |     3 |
| unclear  | filter adds possibly implied restriction   |     1 |
| unclear  | structural key for stated fact             |     1 |
| unclear  | free-text label restating request          |     1 |
| unclear  | class name inferred from dataset name      |     1 |

## How I judged

I read every line in full (request, tool, tool description, argument schema and value) and compared the value only
with the request and the schema. A value was **grounded** if it was the stated value written differently, an item of a
range or a standard scale that the request names, a count or bin edge computed directly from stated numbers, the
canonical name behind an unambiguous abbreviation in a given path, or the request's own terms rephrased as a search.
A value was **invented** if the request gave no value for that argument: training settings, optional switches, output
names and paths, titles and model names. It was also invented if it relied on domain facts the request does not give,
such as class lists, marker genes, coordinates or class counts. Numbers that look like a stated quantity but are not
equal to it or validly converted from it also counted as invented (for example a dropped exponent or a percent misread
as a count). They have their own category, so they can be separated from outright guesses.

The hard lines were mostly ones where the request refers to data it does not include and presents as already
available. That covers a result from an earlier step, a spectrum or summary referred to with a definite article, or a
dataset or model whose location is not given. The value could have been copied from that unseen content or made up,
and I cannot tell which from the line alone, so I marked these unclear. One exception: a request that says the data
will be supplied later was judged invented. Four other lines were unclear because two readings are reasonable:

- a search filter whose extra restriction may or may not be implied by the wording;
- a key in an agent-built payload that names a stated fact;
- a free-text description that restates the request's own phrase;
- a class name inferred only from a dataset folder name.

Class names were grounded only when the request names the label scale itself.
