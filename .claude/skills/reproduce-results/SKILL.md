---
name: reproduce-results
description: Reproduce the System 1 agents study from this supplementary folder. Installs the code, runs the tests, replays every System 1 call from the stored decision-model answers, rescores every agent, rebuilds the manuscript's tables and the fabrication analysis, and checks the headline numbers. Optionally reruns the agents on a GPU. Use when asked to reproduce, verify or re-analyse the results of this study.
---

# Reproduce the System 1 agents results

Work from the folder that holds this `.claude/` directory (the supplementary folder). Report every step's outcome to
the user as it finishes: what ran, what it printed, and whether it matched. Never edit `data/` or `results/`; write new
outputs under `runs/` or a temporary folder.

## 0. Check the environment

- Python 3.11 or later, and conda or venv: `python3 --version`.
- Node.js on the PATH (the biomedical scorer runs the ClinicalTrials.gov server's own normalisers): `node --version`.
  If it is missing, ask the user to install it, or set `S1EVAL_NODE=/path/to/node`.
- A GPU is not needed for steps 1–5.

## 1. Install and test

```bash
cd code
python3 -m venv .venv && . .venv/bin/activate        # or: conda create -n s1agents python=3.12
pip install -e ".[test,figures]"        # or the exact versions: pip install -r requirements-lock.txt
python -m pytest
```

Expected: every test passes. The decider prompt tests are skipped unless decider-35b-a3b's tokenizer is present.

## 2. Check the data

```bash
cd ../data && sha256sum -c SHA256SUMS.txt && cd ../results && sha256sum -c SHA256SUMS.txt && cd ../code
```

## 3. Replay the System 1 agent (no model)

```bash
python scripts/replay_s1.py --catalogue ../data/catalogues/imaging_tools.json \
    --data ../data/requests/imaging_test.jsonl ../data/requests/external.jsonl \
    --answers ../results/system1/answers --records ../results/system1/records
python scripts/replay_s1.py --catalogue ../data/catalogues/biomedical_tools.json \
    --data ../data/requests/biomedical_test.jsonl \
    --answers ../results/system1/answers --records ../results/system1/records
```

Each of the six models is replaced by a back-end that only reads its stored answers and fails on any question it has
no answer for. Expected: for every model and set, every request identical and 0 missing answers (1,179 imaging, 123
external, 824 biomedical).

## 4. Rebuild the analysis

```bash
python scripts/analyse.py --sensitivity --out runs/analysis
```

Compare `runs/analysis/metrics.json` with `../results/analysis/metrics.json`; they should be equal. The headline
numbers to check in `runs/analysis/tables.md`:

- Evaluation set: 1,952 requests for tool calling and fully correct, and 6,056 stated values. The 174 underspecified
  requests are reported apart.
- Fully correct: pplx-decider-27b 1,832 (93.9%), OpenJev 1,813 (92.9%). Best LLM: Qwen3.5-27B with thinking off,
  1,782 (91.3%).
- The 108 System 1 × LLM pairs, each Holm-corrected, as System 1 ahead / LLM ahead / no difference:

  | measure | tally |
  |---|---|
  | fully correct | 73 / 11 / 24 |
  | tool calling | 76 / 5 / 27 |
  | argument filling | 46 / 19 / 43 |

## 5. Rebuild the fabrication analysis

```bash
python scripts/fabrication.py --check-sample --out runs/fabrication   # also re-draws the review sample and compares it
diff -r runs/fabrication ../results/fabrication && echo identical
```

It takes about 11 s, prints counts only, and reads the review verdicts from `../results/fabrication_review/` for the
precision step.

Expected:

- The System 1 agents have 0 nonexistent tools, 0 nonexistent arguments and 0 ungrounded values (33,945 judged).
- Precision of the check, from the independent review in `../results/fabrication_review/`: 0.873 with thinking on and
  0.643 with thinking off (lower bounds).
- After that adjustment, 5.5–19.7% of the LLM agents' calls contain at least one invented value (lower bound).

## 5a. Rebuild Table 1 (the scorecard)

```bash
python scripts/scorecard.py --out runs/scorecard && diff -r runs/scorecard ../results/scorecard && echo identical
```

## 5b. Rebuild the figures

```bash
pip install -e ".[figures]"
python scripts/figures.py --out runs/figures
diff <(python -c "import json;print(json.dumps(json.load(open('runs/figures/figure_data.json')),sort_keys=True))") \
     <(python -c "import json;print(json.dumps(json.load(open('../figures/figure_data.json')),sort_keys=True))") && echo identical
```

Expected: identical plotted numbers, and the line "worked examples: the printed options are the compiler's".

## 6. Optional: rerun the agents on a GPU

Only when the user asks, and only with their go-ahead for the downloads (tens of GB):

1. Get llama.cpp b11929 and the model files as `code/README.md` describes. Check every SHA-256.
2. Run `python scripts/run_s1.py …` and `python scripts/run_llm.py …` as in `code/README.md`, one model at a time.
   Each run takes hours.
3. Rescore with `analyse.py`.

On other hardware a few close decisions can differ; in the study, 0–2 of 117 verdicts per model differed between two
GPUs.

## If something does not match

- Report the exact command and the first difference.
- **Replay misses an answer:** the compiler asked a question the model was never asked. Check that `src/` is unmodified
  (`git status`, or compare with the released archive).
- **The analysis stops with an error about Node.js:** the biomedical scorer needs Node.js; put `node` on the PATH or set `S1EVAL_NODE=/path/to/node`.
- Do not change code, data or labels to make numbers match.
