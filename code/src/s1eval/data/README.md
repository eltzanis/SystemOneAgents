# Data files of the scorer

| file | what | source | licence |
|---|---|---|---|
| `reco_name_tables.json` | the name tables and auto-detected column lists of the imaging MCP servers (eight literals, each keyed by the server source file and variable it is assigned to) | the ReCo imaging MCP servers' Python sources (`feature_importance_analysis_tool.py`, `pycaret_classification_tool.py`, `pycaret_classification_inference_tool.py`, `pycaret_regression_tool.py`, `pyradiomics_feature_extraction_tool.py`, `totalsegmentator_tool.py`); extracted with `ast` (`scripts/extract_name_tables.py`) | the servers' |
| `biomed/clinicaltrials/query-helpers.js` | normalizeStatusFilter, normalizePhaseFilter, queryTermContent | clinicaltrialsgov-mcp-server 2.11.1 (cyanheads), `dist/mcp-server/tools/utils/query-helpers.js`, unchanged | Apache-2.0 (`biomed/clinicaltrials/LICENSE`) |
| `biomed/clinicaltrials/place-names.js` | registryCountry, registryState | the same package, `dist/mcp-server/tools/utils/place-names.js`, unchanged | Apache-2.0 |
| `biomed/chembl/input-normalization.js` | ASSAY_TYPE_CODES (read as text only, never executed) | @cyanheads/chembl-mcp-server 0.5.0, `dist/mcp-server/input-normalization.js`, unchanged | Apache-2.0 (`biomed/chembl/LICENSE`) |
| `biomed/stopwords_english.txt` | 318 English stop words | scikit-learn 1.4.2, `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` (from the Glasgow Information Retrieval Group list) | BSD-3-Clause (`biomed/LICENSE.scikit-learn`) |
| `biomed/normalise.mjs`, `biomed/package.json` | runs the two ClinicalTrials.gov files' exported functions (one JSON line in, one out); marks the folder as ES modules | this work | as this code |
| `pycaret_regression_models.json` | PyCaret 3.3.2 regression model IDs and names, used only by the scoring sensitivity analysis | PyCaret 3.3.2, `pycaret/containers/models/regression.py` | MIT (PyCaret) |

SHA-256 of the unchanged third-party files:

```
c506543e4b98c5072a3b1a1505a96246e32299823a2dcf1e8ff778d46c2298cf  biomed/clinicaltrials/place-names.js
bc33ceed4dac167668f039825f61be1c034d7a19402df50c3de0559ca491b359  biomed/clinicaltrials/query-helpers.js
ffe276971e9916a26f1696f6fc063d5b4077c10e28646c8b52caf8506d35d3fc  biomed/chembl/input-normalization.js
4e22be0ad71ae1c41dd7a8f944e851ead671d114edf4faad1ee8c698d2ba5084  biomed/stopwords_english.txt
```

Node.js (24 was used) runs `normalise.mjs`; put `node` on the PATH or set `S1EVAL_NODE`.
