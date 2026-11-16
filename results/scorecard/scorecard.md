| category | measure | System 1, six agents | LLM, 18 settings | pairs: S1 / LLM / n.s. | pplx-decider-27b | OpenJev | Qwen3.5-27B, reasoning off |
|---|---|---|---|---|---|---|---|
| Accuracy | Fully correct requests (% of 1,952) | 81.4–93.9 | 61.4–91.3 | 73 / 11 / 24 | 93.9 | 92.9 | 91.3 |
| Accuracy | Tool calling (% of 1,952) | 91.7–97.0 | 66.3–95.2 | 76 / 5 / 27 | 97.0 | 96.1 | 95.1 |
| Accuracy | Argument filling (% of 6,056 values) | 89.8–97.0 | 56.6–97.2 | 46 / 19 / 43 | 97.0 | 96.1 | 97.2 |
| Reliability | Calls with invented content (% of calls (LLM: adjusted, lower bound)) | 0.0 | 5.5–19.7 | — | 0.0 | 0.0 | 7.4 |
| Reliability | Calls with argument names absent from the schema (calls) | 0 | 0–53 | — | 0 | 0 | 0 |
| Reliability | Underspecified requests handled correctly (% of 174) | 86.2–93.7 | 0.0–5.7 | — | 93.7 | 92.0 | 0.0 |
| Reliability | Underspecified requests with an invented required value (% of 174) | 0.0 | 0.0–62.1 | — | 0.0 | 0.0 | 27.6 |
| Reliability | Out-of-scope requests declined (% of 307) | 87.0–94.5 | 41.4–97.4 | — | 91.2 | 91.2 | 80.8 |
| Reliability | Complete requests left without a call (% of 1,645) | 1.0–3.3 | 0.0–36.5 | — | 1.0 | 1.7 | 0.7 |
| Reliability | Calls with a schema-invalid value (% of calls) | 0.0–0.4 | 0.0–0.8 | — | 0.0 | 0.1 | 0.1 |
| Efficiency | GPU energy per fully correct request (Wh) | 0.072–0.407 | 0.080–2.481 | — | 0.407 | 0.228 | 0.413 |
| Efficiency | Time per request (s (mean)) | 1.2–10.1 | 1.1–26.1 | — | 10.1 | 2.6 | 4.5 |
| Efficiency | Generated tokens per request (tokens) | 0–15 | 123–952 | — | 0 | 0 | 156 |
| Efficiency | Peak GPU memory (MiB) | 7,368–35,057 | 12,044–45,532 | — | 35,057 | 21,852 | 36,002 |
