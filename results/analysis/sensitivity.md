# Sensitivity of the comparison to the scoring rules

Each row is a full run of scripts/analyse.py; tallies are System 1 ahead / LLM ahead / no significant difference (108 pairs per measure, one Holm correction per measure).

| variant | fully correct | tool decisions | stated values |
|---|---|---|---|
| main | 73 / 11 / 24 | 76 / 5 / 27 | 46 / 19 / 43 |
| strict search text | 73 / 11 / 24 | 76 / 5 / 27 | 46 / 19 / 43 |
| PyCaret model lists | 73 / 11 / 24 | 76 / 5 / 27 | 45 / 19 / 44 |
