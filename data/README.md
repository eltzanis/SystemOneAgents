# Data: requests, labels and tool catalogues

## Files

| file | contents |
|---|---|
| `requests/imaging_test.jsonl` | 1,179 imaging test requests (896 complete, 180 out of scope, 103 underspecified) |
| `requests/external.jsonl` | 123 requests from the published mAIstro (50) and ReCLAIM (73) notebooks, verbatim |
| `requests/biomedical_test.jsonl` | 824 biomedical test requests (626 complete, 127 out of scope, 71 underspecified) |
| `catalogues/imaging_tools.json` | the 42 tools of the 11 imaging MCP servers: name, description, input schema |
| `catalogues/biomedical_tools.json` | the 115 tools of the 7 biomedical MCP servers |
| `SHA256SUMS.txt` | checksums of the files above |

The imaging and external requests use `imaging_tools.json`; the biomedical requests use `biomedical_tools.json`.

## Evaluation set

These requests are the evaluation set of the manuscript. It contains:

- every out-of-scope request;
- every underspecified request;
- every complete request whose stated argument values the System 1 compiler can express as options.

The last condition was checked with the compiler oracle (`code/scripts/oracle.py`).

## One request per line (JSON)

| field | meaning |
|---|---|
| `id` | request identifier |
| `group` | tool group (subcategory used in the analysis) |
| `request` | the request text, as given to every agent |
| `tool` | the labelled tool as `server/tool` (for a request with several steps, the first step's tool); `null` when no tool can do it (out of scope) |
| `args` | only the argument values the request states, in the form the server accepts |
| `ask` | underspecified requests only: the required arguments the request does not give |
| `later_steps` | the later calls of a request with several steps (documentation only, not scored) |
| `accept_tools` | other tools that also count as correct, when present |
| `tags` | phenomena of the request; out-of-scope requests carry exactly one of `oos-far`, `oos-near`, `oos-modality` |
| `source`, `source_name`, `source_section` | external requests only: the notebook a prompt was taken from |

Values the request does not state are not labelled, and setting them is never counted as an error. The output folder
is filled by the agent and is not scored.


## Tool catalogues and servers

| catalogue | servers (tools) |
|---|---|
| imaging | 2D image classification (10), nnU-Net (7), Merlin (7), 3D image classification (6), PyCaret classification (3), PyCaret regression (3), exploratory data analysis (2), feature importance (1), model performance comparison (1), PyRadiomics (1), TotalSegmentator (1) | 
| biomedical | proteomics-mcp v0.2.0 (53), AlphaGenome (22), MSMCP 1.0.0 (13), ClinicalTrials.gov server 2.11.1 (7), ChEMBL server 0.5.0 (7), Scanpy (7), TISSUE (6) | 


