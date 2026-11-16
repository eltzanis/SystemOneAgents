# Precision of the ungrounded-value check

Independent review of 200 random flagged LLM values (sample drawn with seed 20261110).

| thinking mode | sample | invented | unclear | grounded | engine | lower bound (95% CI) | unclear left out | upper bound |
|---|---|---|---|---|---|---|---|---|
| on | 71 | 62 | 2 | 5 | 2 | 0.873 (0.776–0.932) | 0.899 | 0.901 |
| off | 129 | 83 | 34 | 11 | 1 | 0.643 (0.558–0.721) | 0.874 | 0.907 |

## Per agent (adjusted = flagged × precision, lower–upper bound)

| agent | calls | nonexistent tool | nonexistent argument | calls with a flagged value (%) | calls with an invented value, adjusted (%) | invented values, adjusted | underspecified requests with a required value filled: written elsewhere / no source | schema-invalid calls |
|---|---|---|---|---|---|---|---|---|
| System 1 decider-4b | 1807 | 0 | 0 | 0 (0.0) | 0 (0.0) | 0 | 5 / 0 | 7 |
| System 1 pplx-decider-flash | 1832 | 0 | 0 | 0 (0.0) | 0 (0.0) | 0 | 6 / 0 | 5 |
| System 1 decider-35b-a3b | 1796 | 0 | 0 | 0 (0.0) | 0 (0.0) | 0 | 2 / 0 | 3 |
| System 1 OpenJev | 1814 | 0 | 0 | 0 (0.0) | 0 (0.0) | 0 | 3 / 0 | 1 |
| System 1 Clef | 1811 | 0 | 0 | 0 (0.0) | 0 (0.0) | 0 | 3 / 0 | 6 |
| System 1 pplx-decider-27b | 1828 | 0 | 0 | 0 (0.0) | 0 (0.0) | 0 | 3 / 0 | 0 |
| NVIDIA Nemotron 3.5 Bolt E4B [think] | 1061 | 0 | 19 | 103 (9.7) | 90–93 (8.5–8.8) | 100–103 | 0 / 0 | 4 |
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | 1691 | 0 | 51 | 518 (30.6) | 333–470 (19.7–27.8) | 616–869 | 17 / 34 | 14 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | 1728 | 0 | 0 | 257 (14.9) | 224–232 (13.0–13.4) | 403–416 | 8 / 16 | 1 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | 1927 | 0 | 0 | 463 (24.0) | 298–420 (15.5–21.8) | 578–814 | 47 / 91 | 3 |
| Qwen3.8-27B [think] | 1832 | 0 | 0 | 272 (14.8) | 238–245 (13.0–13.4) | 362–374 | 10 / 54 | 0 |
| Qwen3.8-27B [nothink] | 1982 | 0 | 0 | 295 (14.9) | 190–268 (9.6–13.5) | 589–830 | 22 / 108 | 2 |
| Meta Muse Glimmer 30B [think] | 1803 | 0 | 0 | 385 (21.4) | 336–347 (18.6–19.2) | 698–720 | 15 / 33 | 7 |
| Meta Muse Glimmer 30B [nothink] | 1745 | 0 | 0 | 282 (16.2) | 181–256 (10.4–14.7) | 326–460 | 14 / 19 | 7 |
| Google Gemma 4 E4B (it) [think] | 1266 | 0 | 23 | 80 (6.3) | 70–72 (5.5–5.7) | 74–77 | 1 / 0 | 1 |
| Google Gemma 4 E4B (it) [nothink] | 1465 | 0 | 53 | 300 (20.5) | 193–272 (13.2–18.6) | 250–353 | 8 / 12 | 9 |
| Google Gemma 4 26B-A4B (it) [think] | 1732 | 0 | 22 | 219 (12.6) | 191–197 (11.0–11.4) | 209–215 | 7 / 16 | 4 |
| Google Gemma 4 26B-A4B (it) [nothink] | 1805 | 0 | 33 | 374 (20.7) | 241–339 (13.3–18.8) | 302–425 | 14 / 99 | 15 |
| Qwen3.5-4B [think] | 1747 | 0 | 0 | 346 (19.8) | 302–312 (17.3–17.9) | 488–504 | 7 / 28 | 4 |
| Qwen3.5-4B [nothink] | 1828 | 0 | 0 | 477 (26.1) | 307–433 (16.8–23.7) | 1090–1536 | 14 / 97 | 7 |
| Qwen3.5-27B [think] | 1751 | 0 | 0 | 201 (11.5) | 176–181 (10.0–10.3) | 316–326 | 7 / 31 | 2 |
| Qwen3.5-27B [nothink] | 1761 | 0 | 0 | 202 (11.5) | 130–183 (7.4–10.4) | 235–331 | 15 / 48 | 2 |
| Qwen3.5-35B-A3B [think] | 1702 | 0 | 0 | 188 (11.0) | 164–169 (9.6–10.0) | 255–263 | 7 / 19 | 3 |
| Qwen3.5-35B-A3B [nothink] | 1817 | 0 | 0 | 279 (15.4) | 180–253 (9.9–13.9) | 416–587 | 21 / 85 | 1 |
