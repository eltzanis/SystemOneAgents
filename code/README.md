# System 1 agents: code

MCP tool-calling agents whose decisions are made by one-pass **decision models**: a model that reads a state and a
question with a finite list of options and returns a probability for each option, in one forward pass, without
generating text. A deterministic **compiler** turns a tool's JSON schema and the user's request into typed
multiple-choice questions (which tool, which value for each argument, which item comes next in a list); code owns the
control flow. The baseline is a generating LLM used as a tool-calling agent, as its distributor ships it. Both are
scored by the same scorer on the same labelled requests.

```
code/
  src/s1agent/          the System 1 agent
    agent.py            the loop: tool selection, capability check, argument filling, asking, calling, continuing
    compiler.py         schema + request -> typed questions -> arguments (the rules that keep a call faithful)
    candidates.py       the values a request states, in the forms it writes them (paths, numbers, lists, ...)
    backends/           decision models served by llama-server, the lexical baseline, the answer store
    catalog.py          tool catalogues (JSON snapshots of the MCP servers' tool lists)
    mcp_hub.py, cli.py  MCP client and the `s1a` command
  src/s1eval/           evaluation shared by both agents
    scoring.py          verdicts of a call against the labelled request
    equivalence.py      values a server treats as the same (its own name tables and normalisers), data/ (sources, licences)
    records.py, meter.py  the System 1 record of a request; per-request cost (time, GPU energy, tokens)
  src/llm_agent/        the LLM agent baseline (`s1llm`): one completion per request, its scoring path
  configs/              agent.yaml (settings), models.yaml (six decision models), llm_models.yaml (nine LLMs),
                        servers.example.yaml (MCP servers)
  scripts/              run_s1.py, run_llm.py, replay_s1.py, analyse.py, fabrication.py, oracle.py, measure_context.py,
                        smoke_llm.py, extract_name_tables.py
  tests/                run without a GPU or a model
```

The evaluation set is in `../data/` (requests with labels, and the two tool catalogues), the released results in
`../results/` (see "Results" below).

## Install

Python 3.12 (3.11 or later), Linux. From this folder:

```bash
conda create -n s1agents python=3.12 && conda activate s1agents
pip install -e ".[test]"
python -m pytest                         # no GPU, no model; the replay tests use ../results and ../data
```

Versions used for the runs: httpx 0.28.1, jsonschema 4.26.0, mcp 2.3.0, numpy 2.5.4, tokenizers 0.23.3,
transformers 5.17.0, PyYAML, rich. The biomedical part of the scorer runs the ClinicalTrials.gov server's own
normalisers with **Node.js** (24 was used): `node` on the PATH, or `S1EVAL_NODE=/path/to/node`.

The scripts also run from a checkout without installing (they put `src/` on the path).

### llama.cpp

Every model runs in llama.cpp's `llama-server`, release **b11929**, CUDA 12.8 build for Ubuntu x64:

```bash
mkdir -p third_party/llama.cpp && cd third_party/llama.cpp
curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b11929/llama-b11929-bin-ubuntu-cuda-12.8-x64.tar.gz
curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b11929/cudart-llama-b11929-bin-ubuntu-cuda-12.8-x64.tar.gz
tar -xzf llama-b11929-bin-ubuntu-cuda-12.8-x64.tar.gz && tar -xzf cudart-llama-b11929-bin-ubuntu-cuda-12.8-x64.tar.gz
```

SHA-256: `deb661537301b760cdea3e3a8cfbacc350c630b39bbe6bbd766a48b7ad577c3a` (binaries),
`325660a546eb261244a345c06c1e9f9dec07c9c1498ac6ba810e300b97837359` (CUDA runtime). The paths are set in
`configs/models.yaml` and `configs/llm_models.yaml`. The runs used one NVIDIA RTX PRO 5000 Blackwell (48 GB, driver
580), Ubuntu 24.04; every model entirely on the GPU, one model at a time.

### Decision models (System 1)

Files go to `models/<folder>/` (folders as in `configs/models.yaml`). Hugging Face repositories, revisions and SHA-256:

| name (configs) | model | files | source |
|---|---|---|---|
| `decider-4b` | decider 4B (Qwen3.5-4B base), Q8_0 | `decider-4b-v3-Q8_0.gguf` (`d29bda3381301c81d1ecaf8adf2f9df94f724f0179be4ddb11c898f38713ffd0`), `tokenizer.json`, `tokenizer_config.json`, `decider_config.json` | `Mapika/decider-4b-GGUF` @ `d3a78581751d518bbfa9e871a6d84059554d53c5` |
| `pplx-decider-flash` | Perplexity pplx-decider-flash v1.2 (4B), Q8_0 | `decider-flash-v1.2-q8_0.gguf` (`ffb27ade8976765b0e6943a0b7f5f3fd0f7042d45ad90256a76d608bcd3942c2`); in `hf/`: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json`, `decision_config.json`, `joint_config.json`, `readout.safetensors` (`6751578171026190d4db8078e97162b404fd984646e0e9aec13844e4dd5b1711`) | GGUF: `perplexity-ai/pplx-decider-flash-gguf-v1.2` @ `2514d5f0b7fbbe1ee1bfec9ffb5801a60289b4cc`; `hf/`: `perplexity-ai/pplx-decider-flash-v1.2` @ `9ff7cf9838e59df461c276b7699d56b244d2af58` |
| `decider-35b-a3b` | decider 35B-A3B (Qwen3.5-35B-A3B base, mixture of experts), Q4_K_M | `decider-35b-a3b.Q4_K_M.gguf` (`a274da384a4e2f375d76792f43c8173c0319d3eaa9ce8e6be69d8fa9b2b2d8ca`); `tokenizer.json`, `tokenizer_config.json`, `decider_config.json`, `config.json` | GGUF: `mradermacher/decider-35b-a3b-GGUF`; the rest: `Mapika/decider-35b-a3b` @ `91470e65478656eb68fde8c0ccdfe575d69d0309` |
| `openjev` | OpenJev, Q4_K_M | `OpenJev-Q4_K_M.gguf` (`43b8d9cb89851f27c9d6bed8f073f917327bec05471da462eea77cbf896aa891`) | `ggml-org/OpenJev-GGUF` (from `openjev/openjev`; CC-BY-NC-4.0) |
| `clef` | Cloudflare Clef (Qwen3.5-27B + joint schema head), Q8_0 | `Clef-Q8_0.gguf` (`728801f58fcebb35e60bb2ac0cf63fc33e3137b07581da132c1f0f5c5e2149fe`) | `ggml-org/Clef-GGUF` @ `d7492270abd7af222e98dca986580772e6229242` (from `Cloudflare/clef`) |
| `pplx-decider-27b-embed` | Perplexity pplx-decider v1.2 (27B), Q8_0 | `pplx-decider-v1.2-Q8_0.gguf` (`f451f6a69f5481d82bc16b4c24ad6a00426a3b8895d1c42d297d6990f88347e0`); in `models/pplx-27b-hf/`: `readout.safetensors`, `decision_config.json`, `joint_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json` | `perplexity-ai/pplx-decider-v1.2` @ `cce0f3f80033e9d89d80831d5281fd72124aacce`, converted with llama.cpp b11929's converter (below) |

The Perplexity repositories may require accepting their terms on Hugging Face. The 27B GGUF is made with the converter
of the same llama.cpp release:

```bash
git clone --depth 1 --branch b11929 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-src
python -m venv third_party/convert-venv
third_party/convert-venv/bin/pip install -r third_party/llama.cpp-src/requirements/requirements-convert_hf_to_gguf.txt
third_party/convert-venv/bin/pip install -e third_party/llama.cpp-src/gguf-py
hf download perplexity-ai/pplx-decider-v1.2 --revision cce0f3f80033e9d89d80831d5281fd72124aacce --local-dir models/pplx-27b-hf
third_party/convert-venv/bin/python third_party/llama.cpp-src/convert_hf_to_gguf.py models/pplx-27b-hf \
    --outtype q8_0 --outfile models/pplx-decider-27b/pplx-decider-v1.2-Q8_0.gguf
```

How each model is read (prompt layout, readout, server flags) is documented in `src/s1agent/backends/` and
`configs/models.yaml`.

### LLMs (baseline)

Files go to `models/<name>/` (`configs/llm_models.yaml`, which also holds each file's SHA-256):

| name | model | file | source |
|---|---|---|---|
| `nemotron-bolt-e4b` | NVIDIA Nemotron 3.5 Bolt E4B | `NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q8_K_XL.gguf` | `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` @ `b738967d6da3441a7d1c71fcec15e49e32e16ecd` |
| `nemotron-bolt-74b` | NVIDIA Nemotron 3.5 Bolt 74B-A9B (24 GB release) | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF` @ `e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246` |
| `qwen3.8-27b` | Qwen3.8-27B | `Qwen3.8-27B-Q8_0.gguf` | `ggml-org/Qwen3.8-27B-GGUF` @ `a316ea527fde32c12487be7574e2db5ac2ad2a7e` |
| `muse-glimmer-30b` | Meta Muse Glimmer 30B | `Muse-Glimmer-30B-Q8_0.gguf` | `unsloth/Muse-Glimmer-30B-GGUF` @ `faa5b025c584459c13febfa5c59883516710ae39` |
| `gemma-4-e4b` | Google Gemma 4 E4B (it) | `gemma-4-E4B-it-Q8_0.gguf` | `ggml-org/gemma-4-E4B-it-GGUF` @ `a69f018d7678a2b9f5e705f8903d6fb29b9b25dc` |
| `gemma-4-26b-a4b` | Google Gemma 4 26B-A4B (it) | `gemma-4-26B-A4B-it-Q8_0.gguf` | `ggml-org/gemma-4-26B-A4B-it-GGUF` @ `9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2` |
| `qwen3.5-4b` | Qwen3.5-4B | `Qwen3.5-4B-Q8_0.gguf` | `unsloth/Qwen3.5-4B-GGUF` @ `e87f176479d0855a907a41277aca2f8ee7a09523` |
| `qwen3.5-27b` | Qwen3.5-27B | `Qwen3.5-27B-Q8_0.gguf` | `unsloth/Qwen3.5-27B-GGUF` @ `3221f178a6b842d04f1fb42f1c413534adcc0a6a` |
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B | `Qwen3.5-35B-A3B-Q4_K_M.gguf` | `unsloth/Qwen3.5-35B-A3B-GGUF` @ `bc014a17be43adabd7066b7a86075ff935c6a4e2` |

Each is used with the chat template inside its GGUF (`--jinja`).

## The System 1 agent

```bash
s1a serve decider-4b &                                    # llama-server of a model in configs/models.yaml
s1a plan "Segment the liver and the spleen in /data/ct/p01.nii.gz" \
    --catalogue ../data/catalogues/imaging_tools.json --model decider-4b
s1a run "..." --catalogue my_tools.json --servers my_servers.yaml --model decider-4b   # also calls the tool
s1a catalog --servers my_servers.yaml --out my_tools.json # list the tools of your MCP servers
```

`plan` prints every decision (question, top options, probabilities) and the call. Agent settings are in
`configs/agent.yaml`: abstain below a selection confidence of 0.3; set an optional value, or a list's first item, only
when P(the request gives it) reaches 0.7; an output folder the request does not name is filled by policy
(`{workspace}/{tool}_{timestamp}`). In Python: `from s1agent import build_agent`.

### Evaluation run

```bash
python scripts/run_s1.py --catalogue ../data/catalogues/imaging_tools.json \
    --data ../data/requests/imaging_test.jsonl ../data/requests/external.jsonl
python scripts/run_s1.py --catalogue ../data/catalogues/biomedical_tools.json --data ../data/requests/biomedical_test.jsonl
```

For each model: start its server, measure 20 s of idle power, plan every request (selection with the raw confidence,
arguments; nothing is executed) under the cost meter, stop the server. Records go to `runs/s1/<set>/<model>.jsonl`,
every answer of the model to `runs/s1_answers/<model>.jsonl` (record only: every question is asked).

### Replay without a model

```bash
python scripts/replay_s1.py --catalogue ../data/catalogues/imaging_tools.json \
    --data ../data/requests/imaging_test.jsonl ../data/requests/external.jsonl \
    --answers ../results/system1/answers --records ../results/system1/records
python scripts/replay_s1.py --catalogue ../data/catalogues/biomedical_tools.json \
    --data ../data/requests/biomedical_test.jsonl --answers ../results/system1/answers --records ../results/system1/records
```

Each model is replaced by a backend that only reads its recorded answers and fails on any question it has no answer
for. An answer is keyed by the SHA-256 of the model name, the question kind, the state, the question text and the
options, so the replay succeeds only if the compiler asks exactly the questions the model was asked. Every rebuilt
record is compared with the released one (tool, confidence, arguments, decisions, verdicts); only the folder and time
stamp of the policy output folder are masked.

## The LLM agent baseline

Each LLM sees the catalogue's tools as native function definitions (MCP description and input schema unchanged; name
`<server>__<tool>`), gets the request as the only message (no system prompt: the chat template's defaults apply), and
answers with one completion: seed 20261102, at most 16,384 generated tokens, sampling as published by the vendor, two
conditions (reasoning on: `think`; off: `nothink`). Its first function call is its decision; no call means no tool.
Nothing is added (no function to ask or decline, no retry) and nothing is executed.

```bash
python scripts/run_llm.py --catalogue ../data/catalogues/imaging_tools.json \
    --data ../data/requests/imaging_test.jsonl ../data/requests/external.jsonl
python scripts/run_llm.py --catalogue ../data/catalogues/biomedical_tools.json --data ../data/requests/biomedical_test.jsonl
s1llm score --model qwen3.5-4b --cond think --data ../data/requests/external.jsonl \
    --catalogue ../data/catalogues/imaging_tools.json --answers ../results/llm --out runs/llm_eval
```

`run_llm.py` starts each model's server in turn (context 65,536 tokens, or the larger one `ctx_by_catalogue` gives,
measured by `scripts/measure_context.py`), asks every request once under the cost meter, scores the answers and stops
the server. Completions are saved with the hash of everything that shapes them (the settings key), so a run continues
after an interruption and can be rescored without the model. `scripts/smoke_llm.py` checks that a model's tool calls come
back parsed.

## Analysis

```bash
python scripts/analyse.py                     # ../data + ../results -> ../results/analysis/tables.md, metrics.json
python scripts/analyse.py --sensitivity       # also the scoring-rule variants -> sensitivity.md
```

Every System 1 record and every LLM completion is scored again by `s1eval`, against the labels of `../data/requests`.
Requests are pooled over the three sets. Measures: tool decisions (complete in-scope and out-of-scope requests), stated
values (every labelled value of the complete requests), fully correct requests (the primary measure); each System 1
model against each LLM in both conditions (6 × 9 × 2 = 108 pairs per measure, one Holm correction per measure; exact
McNemar for requests, exact paired sign-flip test by request for values); each LLM in its better condition;
same-backbone pairs; subcategories; the underspecified requests; cost per request and per fully correct request.

## Fabrication analysis

```bash
python scripts/fabrication.py                 # ../data + ../results -> ../results/fabrication/
python scripts/fabrication.py --check-sample  # also re-draws the review sample and compares it with the released one
```

Can an agent send content that nothing grounds? For every call of every agent on the evaluation set (System 1: the
recorded calls; LLMs: the first call of each completion; the policy output folder left out): F1 a nonexistent tool, F2
a nonexistent argument, F3 a schema-invalid value (reported as schema validity, apart from fabrication), F4 an
ungrounded value (not the labelled value, not in the request, not a number the request writes, not an option the
compiler offers for that argument, not in the schema; each list item, object key and value judged), F5 a required
value set on an underspecified request. The precision of F4 comes from an independent review of 200 random flagged LLM
values (`../results/fabrication_review/`: guide, sample, verdicts, summary, and the key from sample number to agent);
`precision.md` gives the precision per thinking mode and the precision-adjusted counts per agent. Definitions in the
script's docstring.

## The compiler oracle

```bash
python scripts/oracle.py --data my_requests.jsonl --catalogue my_tools.json --list
```

A development tool for systems of this kind: it drives the compiler with a backend that answers every question the way
the label says, and reports the complete requests whose labelled call the compiler cannot express (a stated value that
no question offers, so no decision model could fill it), with the values out of reach.

## Results (`../results/`)

| path | content |
|---|---|
| `system1/records/<set>/<model>.jsonl` | one record per request (six models and `lexical`): selected tool, confidence, selection steps (tag, choice, confidence), arguments, argument decisions, verdicts, cost (time, GPU energy, power, tokens counted by the server). The request text is in `../data/requests/` |
| `system1/answers/<model>.jsonl` | every answer each decision model gave in the run (`{"key", "probs", "prompt_tokens", "missing_labels"}`): what the replay reads |
| `llm/<set>/<model>__<cond>.jsonl` | the LLM completions (`{"id", "settings", "turns", "cost"}`: message with reasoning text and tool calls, token usage, timings, cost) |
| `llm/<set>/<model>__<cond>.summary.json` | the run's settings key and settings, and counts scored by this code |
| `analysis/` | `tables.md`, `metrics.json` and `sensitivity.md` written by `scripts/analyse.py` |
| `fabrication/` | `fabrication.json`, `fabrication.md`, `precision.json`, `precision.md` written by `scripts/fabrication.py` |
| `fabrication_review/` | the independent review of flagged values: `REVIEW_GUIDE.md`, `sample.jsonl` (with request text), `verdicts.jsonl`, `REVIEW_SUMMARY.md`, `review_key.jsonl` (sample number -> agent) |
| `SHA256SUMS.txt` | checksums of every file above |

Sets: `imaging_test` and `external` (imaging catalogue), `biomedical_test` (biomedical catalogue).

## Licences of included third-party files

`src/s1eval/data/README.md`: ClinicalTrials.gov and ChEMBL MCP server files (Apache-2.0, cyanheads), scikit-learn's
English stop words (BSD-3-Clause), PyCaret's regression model names (MIT), and the name tables of the imaging MCP servers.
The MCP servers themselves are not included; the catalogues are JSON snapshots of their tool lists.

## Figures

```bash
pip install -e ".[figures]"                   # matplotlib >= 3.10
python scripts/figures.py --out runs/figures  # ../data + ../results -> the manuscript's figures
```

Writes every figure of the manuscript (PDF with embedded fonts, SVG, 600-dpi PNG) and `figure_data.json`, the numbers
the figures plot; it equals `../figures/figure_data.json`. Figure 1b shows the recorded decisions of one
development request (`../results/figure1_examples/`); the script checks that the options it prints are the ones this
compiler builds.

## Scorecard (Table 1 of the manuscript)

```bash
python scripts/scorecard.py --out runs/scorecard   # ../results/analysis + ../results/fabrication -> scorecard.md, .json
```

Each measure as a range over the six System 1 agents and the 18 LLM-agent settings, the 108-pair tallies of the three
accuracy measures, and the values of three leading agents. It equals `../results/scorecard/`.
