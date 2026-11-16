# The System 1 agent against the LLM agents

**scoring variant: PyCaret's model names for the regression server's model lists**

Requests, pooled over both fields and every source: 1645 complete in scope + 307 out of scope = 1952 (tool decisions, fully correct); 6056 stated values; 174 underspecified apart. Sets: imaging_test 1076, external 123, biomedical_test 753.

## 1. Each system

### System 1 (six decision models) and the lexical baseline

| system | tool decisions right /1952 (95% CI) | in scope, right tool /1645 | out of scope declined /307 | wrong tool | no call (in scope) | stated values right /6056 | fully correct /1952 (95% CI) | extras (right tool) |
|---|---|---|---|---|---|---|---|---|
| System 1 decider-4b | 1804 (92.4; 91.2–93.5) | 1537 | 267 | 61 | 47 | 5438 (89.8) | 1589 (81.4; 79.6–83.1) | 378 |
| System 1 pplx-decider-flash | 1857 (95.1; 94.1–96.0) | 1588 | 269 | 33 | 24 | 5727 (94.6) | 1752 (89.8; 88.3–91.0) | 466 |
| System 1 decider-35b-a3b | 1790 (91.7; 90.4–92.8) | 1519 | 271 | 72 | 54 | 5478 (90.5) | 1692 (86.7; 85.1–88.1) | 391 |
| System 1 OpenJev | 1876 (96.1; 95.2–96.9) | 1596 | 280 | 21 | 28 | 5817 (96.1) | 1813 (92.9; 91.7–93.9) | 415 |
| System 1 Clef | 1857 (95.1; 94.1–96.0) | 1567 | 290 | 54 | 24 | 5688 (93.9) | 1778 (91.1; 89.7–92.3) | 385 |
| System 1 pplx-decider-27b | 1894 (97.0; 96.2–97.7) | 1614 | 280 | 15 | 16 | 5876 (97.0) | 1832 (93.9; 92.7–94.8) | 241 |
| lexical baseline (no model) | 962 (49.3; 47.1–51.5) | 944 | 18 | 610 | 91 | 1244 (20.5) | 212 (10.9; 9.6–12.3) | 1206 |

### LLM agents, thinking on

| system | tool decisions right /1952 (95% CI) | in scope, right tool /1645 | out of scope declined /307 | wrong tool | no call (in scope) | stated values right /6056 | fully correct /1952 (95% CI) | extras (right tool) |
|---|---|---|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B [think] | 1295 (66.3; 64.2–68.4) | 996 | 299 | 49 | 600 | 3426 (56.6) | 1198 (61.4; 59.2–63.5) | 292 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | 1802 (92.3; 91.0–93.4) | 1567 | 235 | 53 | 25 | 5698 (94.1) | 1715 (87.9; 86.3–89.2) | 1542 |
| Qwen3.8-27B [think] | 1733 (88.8; 87.3–90.1) | 1522 | 211 | 121 | 2 | 5622 (92.8) | 1692 (86.7; 85.1–88.1) | 1170 |
| Meta Muse Glimmer 30B [think] | 1727 (88.5; 87.0–89.8) | 1513 | 214 | 128 | 4 | 5527 (91.3) | 1676 (85.9; 84.2–87.3) | 2070 |
| Google Gemma 4 E4B (it) [think] | 1493 (76.5; 74.6–78.3) | 1198 | 295 | 48 | 399 | 4210 (69.5) | 1375 (70.4; 68.4–72.4) | 227 |
| Google Gemma 4 26B-A4B (it) [think] | 1773 (90.8; 89.5–92.0) | 1523 | 250 | 108 | 14 | 5541 (91.5) | 1631 (83.6; 81.8–85.1) | 678 |
| Qwen3.5-4B [think] | 1798 (92.1; 90.8–93.2) | 1563 | 235 | 67 | 15 | 5609 (92.6) | 1627 (83.4; 81.6–84.9) | 1482 |
| Qwen3.5-27B [think] | 1812 (92.8; 91.6–93.9) | 1584 | 228 | 40 | 21 | 5796 (95.7) | 1736 (88.9; 87.5–90.3) | 992 |
| Qwen3.5-35B-A3B [think] | 1858 (95.2; 94.1–96.0) | 1593 | 265 | 34 | 18 | 5800 (95.8) | 1752 (89.8; 88.3–91.0) | 889 |

### LLM agents, thinking off

| system | tool decisions right /1952 (95% CI) | in scope, right tool /1645 | out of scope declined /307 | wrong tool | no call (in scope) | stated values right /6056 | fully correct /1952 (95% CI) | extras (right tool) |
|---|---|---|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | 1747 (89.5; 88.1–90.8) | 1486 | 261 | 100 | 59 | 5334 (88.1) | 1558 (79.8; 78.0–81.5) | 2270 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | 1714 (87.8; 86.3–89.2) | 1546 | 168 | 91 | 8 | 5513 (91.0) | 1550 (79.4; 77.6–81.1) | 1833 |
| Qwen3.8-27B [nothink] | 1710 (87.6; 86.1–89.0) | 1583 | 127 | 62 | 0 | 5753 (95.0) | 1639 (84.0; 82.3–85.5) | 989 |
| Meta Muse Glimmer 30B [nothink] | 1831 (93.8; 92.6–94.8) | 1583 | 248 | 53 | 9 | 5793 (95.7) | 1758 (90.1; 88.7–91.3) | 1686 |
| Google Gemma 4 E4B (it) [nothink] | 1634 (83.7; 82.0–85.3) | 1346 | 288 | 73 | 226 | 4754 (78.5) | 1420 (72.7; 70.7–74.7) | 1114 |
| Google Gemma 4 26B-A4B (it) [nothink] | 1778 (91.1; 89.7–92.3) | 1508 | 270 | 119 | 18 | 5418 (89.5) | 1597 (81.8; 80.0–83.5) | 801 |
| Qwen3.5-4B [nothink] | 1758 (90.1; 88.7–91.3) | 1534 | 224 | 91 | 20 | 5477 (90.4) | 1566 (80.2; 78.4–81.9) | 1689 |
| Qwen3.5-27B [nothink] | 1857 (95.1; 94.1–96.0) | 1609 | 248 | 25 | 11 | 5888 (97.2) | 1782 (91.3; 90.0–92.5) | 1068 |
| Qwen3.5-35B-A3B [nothink] | 1843 (94.4; 93.3–95.4) | 1606 | 237 | 29 | 10 | 5809 (95.9) | 1710 (87.6; 86.1–89.0) | 1022 |

## 2. Each System 1 model against each LLM in both modes, one Holm correction per measure over the 108 pairs (System 1 minus LLM in percentage points; Holm-adjusted * < 0.05, ** < 0.01, *** < 0.001)

### fully correct requests (/1952, exact McNemar): the primary measure

| LLM \ System 1 | decider-4b | pplx-decider-flash | decider-35b-a3b | OpenJev | Clef | pplx-decider-27b | S1 ahead / LLM ahead / n.s. |
|---|---|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B [think] | +20.0*** | +28.4*** | +25.3*** | +31.5*** | +29.7*** | +32.5*** | 6 / 0 / 0 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | -6.5*** | +1.9 | -1.2 | +5.0*** | +3.2* | +6.0*** | 3 / 1 / 2 |
| Qwen3.8-27B [think] | -5.3*** | +3.1* | +0.0 | +6.2*** | +4.4*** | +7.2*** | 4 / 1 / 1 |
| Meta Muse Glimmer 30B [think] | -4.5** | +3.9** | +0.8 | +7.0*** | +5.2*** | +8.0*** | 4 / 1 / 1 |
| Google Gemma 4 E4B (it) [think] | +11.0*** | +19.3*** | +16.2*** | +22.4*** | +20.6*** | +23.4*** | 6 / 0 / 0 |
| Google Gemma 4 26B-A4B (it) [think] | -2.2 | +6.2*** | +3.1* | +9.3*** | +7.5*** | +10.3*** | 5 / 0 / 1 |
| Qwen3.5-4B [think] | -1.9 | +6.4*** | +3.3* | +9.5*** | +7.7*** | +10.5*** | 5 / 0 / 1 |
| Qwen3.5-27B [think] | -7.5*** | +0.8 | -2.3 | +3.9*** | +2.2 | +4.9*** | 2 / 1 / 3 |
| Qwen3.5-35B-A3B [think] | -8.4*** | +0.0 | -3.1* | +3.1** | +1.3 | +4.1*** | 2 / 2 / 2 |
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | +1.6 | +9.9*** | +6.9*** | +13.1*** | +11.3*** | +14.0*** | 5 / 0 / 1 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | +2.0 | +10.3*** | +7.3*** | +13.5*** | +11.7*** | +14.4*** | 5 / 0 / 1 |
| Qwen3.8-27B [nothink] | -2.6 | +5.8*** | +2.7 | +8.9*** | +7.1*** | +9.9*** | 4 / 0 / 2 |
| Meta Muse Glimmer 30B [nothink] | -8.7*** | -0.3 | -3.4* | +2.8* | +1.0 | +3.8*** | 2 / 2 / 2 |
| Google Gemma 4 E4B (it) [nothink] | +8.7*** | +17.0*** | +13.9*** | +20.1*** | +18.3*** | +21.1*** | 6 / 0 / 0 |
| Google Gemma 4 26B-A4B (it) [nothink] | -0.4 | +7.9*** | +4.9*** | +11.1*** | +9.3*** | +12.0*** | 5 / 0 / 1 |
| Qwen3.5-4B [nothink] | +1.2 | +9.5*** | +6.5*** | +12.7*** | +10.9*** | +13.6*** | 5 / 0 / 1 |
| Qwen3.5-27B [nothink] | -9.9*** | -1.5 | -4.6*** | +1.6 | -0.2 | +2.6* | 1 / 2 / 3 |
| Qwen3.5-35B-A3B [nothink] | -6.2*** | +2.2 | -0.9 | +5.3*** | +3.5** | +6.2*** | 3 / 1 / 2 |

**All 108 pairs: System 1 ahead 73, LLM ahead 11, no significant difference 24** (within this correction: thinking on 37 / 6 / 11; thinking off 36 / 5 / 13).

### tool decisions (/1952, exact McNemar)

| LLM \ System 1 | decider-4b | pplx-decider-flash | decider-35b-a3b | OpenJev | Clef | pplx-decider-27b | S1 ahead / LLM ahead / n.s. |
|---|---|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B [think] | +26.1*** | +28.8*** | +25.4*** | +29.8*** | +28.8*** | +30.7*** | 6 / 0 / 0 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | +0.1 | +2.8*** | -0.6 | +3.8*** | +2.8** | +4.7*** | 4 / 0 / 2 |
| Qwen3.8-27B [think] | +3.6*** | +6.4*** | +2.9* | +7.3*** | +6.4*** | +8.2*** | 6 / 0 / 0 |
| Meta Muse Glimmer 30B [think] | +3.9*** | +6.7*** | +3.2* | +7.6*** | +6.7*** | +8.6*** | 6 / 0 / 0 |
| Google Gemma 4 E4B (it) [think] | +15.9*** | +18.6*** | +15.2*** | +19.6*** | +18.6*** | +20.5*** | 6 / 0 / 0 |
| Google Gemma 4 26B-A4B (it) [think] | +1.6 | +4.3*** | +0.9 | +5.3*** | +4.3*** | +6.2*** | 4 / 0 / 2 |
| Qwen3.5-4B [think] | +0.3 | +3.0*** | -0.4 | +4.0*** | +3.0** | +4.9*** | 4 / 0 / 2 |
| Qwen3.5-27B [think] | -0.4 | +2.3* | -1.1 | +3.3*** | +2.3* | +4.2*** | 4 / 0 / 2 |
| Qwen3.5-35B-A3B [think] | -2.8** | -0.1 | -3.5*** | +0.9 | -0.1 | +1.8* | 1 / 2 / 3 |
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | +2.9* | +5.6*** | +2.2 | +6.6*** | +5.6*** | +7.5*** | 5 / 0 / 1 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | +4.6*** | +7.3*** | +3.9*** | +8.3*** | +7.3*** | +9.2*** | 6 / 0 / 0 |
| Qwen3.8-27B [nothink] | +4.8*** | +7.5*** | +4.1*** | +8.5*** | +7.5*** | +9.4*** | 6 / 0 / 0 |
| Meta Muse Glimmer 30B [nothink] | -1.4 | +1.3 | -2.1 | +2.3* | +1.3 | +3.2*** | 2 / 0 / 4 |
| Google Gemma 4 E4B (it) [nothink] | +8.7*** | +11.4*** | +8.0*** | +12.4*** | +11.4*** | +13.3*** | 6 / 0 / 0 |
| Google Gemma 4 26B-A4B (it) [nothink] | +1.3 | +4.0*** | +0.6 | +5.0*** | +4.0*** | +5.9*** | 4 / 0 / 2 |
| Qwen3.5-4B [nothink] | +2.4 | +5.1*** | +1.6 | +6.0*** | +5.1*** | +7.0*** | 4 / 0 / 2 |
| Qwen3.5-27B [nothink] | -2.7** | +0.0 | -3.4*** | +1.0 | +0.0 | +1.9* | 1 / 2 / 3 |
| Qwen3.5-35B-A3B [nothink] | -2.0 | +0.7 | -2.7** | +1.7 | +0.7 | +2.6*** | 1 / 1 / 4 |

**All 108 pairs: System 1 ahead 76, LLM ahead 5, no significant difference 27** (within this correction: thinking on 41 / 2 / 11; thinking off 35 / 3 / 16).

### stated values (/6056, exact paired sign-flip test by request)

| LLM \ System 1 | decider-4b | pplx-decider-flash | decider-35b-a3b | OpenJev | Clef | pplx-decider-27b | S1 ahead / LLM ahead / n.s. |
|---|---|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Bolt E4B [think] | +33.2*** | +38.0*** | +33.9*** | +39.5*** | +37.4*** | +40.5*** | 6 / 0 / 0 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | -4.3*** | +0.5 | -3.6** | +2.0 | -0.2 | +2.9*** | 1 / 2 / 3 |
| Qwen3.8-27B [think] | -3.0* | +1.7 | -2.4 | +3.2** | +1.1 | +4.2*** | 2 / 1 / 3 |
| Meta Muse Glimmer 30B [think] | -1.5 | +3.3** | -0.8 | +4.8*** | +2.7 | +5.8*** | 3 / 0 / 3 |
| Google Gemma 4 E4B (it) [think] | +20.3*** | +25.0*** | +20.9*** | +26.5*** | +24.4*** | +27.5*** | 6 / 0 / 0 |
| Google Gemma 4 26B-A4B (it) [think] | -1.7 | +3.1*** | -1.0 | +4.6*** | +2.4 | +5.5*** | 3 / 0 / 3 |
| Qwen3.5-4B [think] | -2.8* | +1.9 | -2.2 | +3.4*** | +1.3 | +4.4*** | 2 / 1 / 3 |
| Qwen3.5-27B [think] | -5.9*** | -1.1 | -5.3*** | +0.3 | -1.8 | +1.3 | 0 / 2 / 4 |
| Qwen3.5-35B-A3B [think] | -6.0*** | -1.2 | -5.3*** | +0.3 | -1.8 | +1.3 | 0 / 2 / 4 |
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | +1.7 | +6.5*** | +2.4 | +8.0*** | +5.8*** | +8.9*** | 4 / 0 / 2 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | -1.2 | +3.5*** | -0.6 | +5.0*** | +2.9* | +6.0*** | 4 / 0 / 2 |
| Qwen3.8-27B [nothink] | -5.2*** | -0.4 | -4.5*** | +1.1 | -1.1 | +2.0 | 0 / 2 / 4 |
| Meta Muse Glimmer 30B [nothink] | -5.9*** | -1.1 | -5.2*** | +0.4 | -1.7 | +1.4 | 0 / 2 / 4 |
| Google Gemma 4 E4B (it) [nothink] | +11.3*** | +16.1*** | +12.0*** | +17.6*** | +15.4*** | +18.5*** | 6 / 0 / 0 |
| Google Gemma 4 26B-A4B (it) [nothink] | +0.3 | +5.1*** | +1.0 | +6.6*** | +4.5*** | +7.6*** | 4 / 0 / 2 |
| Qwen3.5-4B [nothink] | -0.6 | +4.1*** | +0.0 | +5.6*** | +3.5** | +6.6*** | 4 / 0 / 2 |
| Qwen3.5-27B [nothink] | -7.4*** | -2.7*** | -6.8*** | -1.2 | -3.3*** | -0.2 | 0 / 4 / 2 |
| Qwen3.5-35B-A3B [nothink] | -6.1*** | -1.4 | -5.5*** | +0.1 | -2.0* | +1.1 | 0 / 3 / 3 |

**All 108 pairs: System 1 ahead 45, LLM ahead 19, no significant difference 44** (within this correction: thinking on 23 / 8 / 23; thinking off 22 / 11 / 21).

## 3. Each LLM in its better mode (chosen on fully correct, which favours the LLMs; Holm over the 54 pairs of each measure)

Better mode: gemma-4-26b-a4b think (1631 vs 1597), gemma-4-e4b nothink (1375 vs 1420), muse-glimmer-30b nothink (1676 vs 1758), nemotron-bolt-74b think (1715 vs 1550), nemotron-bolt-e4b nothink (1198 vs 1558), qwen3.5-27b nothink (1736 vs 1782), qwen3.5-35b-a3b think (1752 vs 1710), qwen3.5-4b think (1627 vs 1566), qwen3.8-27b think (1692 vs 1639).

- full: System 1 ahead 33, LLM ahead 8, no significant difference 13
- tool: System 1 ahead 33, LLM ahead 4, no significant difference 17
- vals: System 1 ahead 19, LLM ahead 12, no significant difference 23

## 4. Same backbone (each LLM against the System 1 model built on it; p unadjusted)

- System 1 decider-4b vs Qwen3.5-4B [think]: tool 1804 vs 1798 (+0.3 pp, p=0.72); vals 5438 vs 5609 (-2.8 pp, p=0.00051); full 1589 vs 1627 (-1.9 pp, p=0.08)
- System 1 pplx-decider-flash vs Qwen3.5-4B [think]: tool 1857 vs 1798 (+3.0 pp, p=2.8e-06); vals 5727 vs 5609 (+1.9 pp, p=0.0064); full 1752 vs 1627 (+6.4 pp, p=3.4e-11)
- System 1 Clef vs Qwen3.5-27B [think]: tool 1857 vs 1812 (+2.3 pp, p=0.0017); vals 5688 vs 5796 (-1.8 pp, p=0.0099); full 1778 vs 1736 (+2.2 pp, p=0.021)
- System 1 pplx-decider-27b vs Qwen3.5-27B [think]: tool 1894 vs 1812 (+4.2 pp, p=5.4e-12); vals 5876 vs 5796 (+1.3 pp, p=0.0099); full 1832 vs 1736 (+4.9 pp, p=4.3e-10)
- System 1 decider-35b-a3b vs Qwen3.5-35B-A3B [think]: tool 1790 vs 1858 (-3.5 pp, p=1.3e-06); vals 5478 vs 5800 (-5.3 pp, p=2.2e-11); full 1692 vs 1752 (-3.1 pp, p=0.001)
- System 1 decider-4b vs Qwen3.5-4B [nothink]: tool 1804 vs 1758 (+2.4 pp, p=0.003); vals 5438 vs 5477 (-0.6 pp, p=0.45); full 1589 vs 1566 (+1.2 pp, p=0.31)
- System 1 pplx-decider-flash vs Qwen3.5-4B [nothink]: tool 1857 vs 1758 (+5.1 pp, p=5.2e-13); vals 5727 vs 5477 (+4.1 pp, p=1.8e-08); full 1752 vs 1566 (+9.5 pp, p=8.6e-21)
- System 1 Clef vs Qwen3.5-27B [nothink]: tool 1857 vs 1857 (+0.0 pp, p=1); vals 5688 vs 5888 (-3.3 pp, p=3e-07); full 1778 vs 1782 (-0.2 pp, p=0.86)
- System 1 pplx-decider-27b vs Qwen3.5-27B [nothink]: tool 1894 vs 1857 (+1.9 pp, p=0.00089); vals 5876 vs 5888 (-0.2 pp, p=0.71); full 1832 vs 1782 (+2.6 pp, p=0.00086)
- System 1 decider-35b-a3b vs Qwen3.5-35B-A3B [nothink]: tool 1790 vs 1843 (-2.7 pp, p=0.0002); vals 5478 vs 5809 (-5.5 pp, p=4.5e-14); full 1692 vs 1710 (-0.9 pp, p=0.36)

## 5. Subcategories (mean [min–max] in % over the systems of each family; Δ = System 1 mean minus LLM mean, pp; descriptive, no test)

| subcategory | requests | values | tool: System 1 | tool: LLM think | Δ | tool: LLM nothink | Δ | values: System 1 | values: LLM think | Δ | values: LLM nothink | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| source: generated (imaging and biomedical test sets) | 1829 | 5467 | 94.3 [91–97] | 86.5 [65–95] | +7.7 | 89.8 [83–95] | +4.4 | 93.1 [89–97] | 86.1 [54–96] | +7.0 | 90.8 [77–97] | +2.3 |
| source: mAIstro (external) | 50 | 214 | 100.0 [100–100] | 96.4 [90–100] | +3.6 | 97.1 [94–100] | +2.9 | 98.2 [97–99] | 92.5 [86–97] | +5.7 | 92.6 [88–95] | +5.6 |
| source: ReCLAIM (external) | 73 | 375 | 98.9 [93–100] | 93.3 [77–100] | +5.6 | 98.8 [95–100] | +0.1 | 98.8 [93–100] | 91.7 [73–100] | +7.1 | 96.9 [92–100] | +1.9 |
| group: automl | 112 | 374 | 99.9 [99–100] | 86.8 [45–100] | +13.0 | 98.0 [91–100] | +1.8 | 98.5 [95–100] | 84.8 [44–98] | +13.8 | 94.5 [85–98] | +4.0 |
| group: chemistry | 32 | 61 | 93.8 [84–97] | 95.1 [88–100] | -1.4 | 97.2 [91–100] | -3.5 | 89.6 [72–98] | 93.6 [85–98] | -4.0 | 91.6 [79–100] | -2.0 |
| group: clinical_trials | 41 | 114 | 89.8 [80–98] | 98.6 [98–100] | -8.8 | 96.5 [90–100] | -6.6 | 82.9 [59–95] | 95.4 [88–100] | -12.5 | 88.2 [78–96] | -5.3 |
| group: dl_2d | 322 | 1679 | 98.7 [97–100] | 89.5 [53–99] | +9.2 | 96.0 [80–100] | +2.7 | 97.8 [95–99] | 88.8 [55–98] | +9.0 | 94.6 [80–99] | +3.2 |
| group: dl_3d | 172 | 771 | 98.8 [98–100] | 81.5 [20–100] | +17.4 | 93.5 [66–100] | +5.4 | 98.2 [97–99] | 80.4 [22–99] | +17.8 | 91.2 [65–100] | +7.0 |
| group: eda | 35 | 110 | 95.7 [89–100] | 91.4 [63–100] | +4.3 | 100.0 [100–100] | -4.3 | 95.9 [86–100] | 91.5 [63–100] | +4.4 | 99.6 [99–100] | -3.7 |
| group: evaluation | 41 | 132 | 99.6 [98–100] | 99.7 [98–100] | -0.1 | 99.7 [98–100] | -0.1 | 98.5 [97–100] | 99.3 [97–100] | -0.8 | 98.8 [94–100] | -0.3 |
| group: feature_selection | 30 | 125 | 98.9 [93–100] | 89.3 [47–100] | +9.6 | 98.5 [93–100] | +0.4 | 96.1 [90–100] | 87.0 [46–99] | +9.1 | 94.6 [87–99] | +1.6 |
| group: genomics | 135 | 618 | 91.2 [86–96] | 91.7 [59–99] | -0.5 | 95.0 [86–99] | -3.7 | 84.5 [78–93] | 89.6 [57–98] | -5.1 | 91.5 [83–98] | -7.0 |
| group: merlin | 66 | 185 | 97.5 [94–98] | 96.6 [82–100] | +0.8 | 100.0 [100–100] | -2.5 | 97.4 [94–99] | 95.9 [84–100] | +1.5 | 99.2 [98–99] | -1.8 |
| group: metabolomics | 46 | 92 | 91.7 [78–98] | 97.1 [83–100] | -5.4 | 96.4 [80–100] | -4.7 | 88.6 [72–98] | 97.8 [89–100] | -9.2 | 96.6 [85–100] | -8.0 |
| group: nnunet | 116 | 403 | 94.7 [91–100] | 78.6 [53–94] | +16.0 | 82.5 [69–98] | +12.2 | 93.8 [89–98] | 80.5 [57–95] | +13.4 | 83.0 [72–95] | +10.8 |
| group: proteomics | 281 | 792 | 89.4 [81–97] | 82.8 [64–92] | +6.7 | 84.9 [63–96] | +4.5 | 83.6 [73–92] | 78.1 [56–90] | +5.5 | 79.2 [55–94] | +4.4 |
| group: radiomics | 38 | 165 | 99.1 [97–100] | 99.1 [95–100] | -0.0 | 98.8 [92–100] | +0.3 | 96.0 [93–98] | 97.7 [93–99] | -1.8 | 96.6 [89–99] | -0.6 |
| group: segmentation | 41 | 131 | 99.2 [95–100] | 95.9 [85–100] | +3.3 | 98.1 [95–100] | +1.1 | 98.6 [95–100] | 94.2 [84–99] | +4.4 | 94.8 [87–98] | +3.8 |
| group: single_cell | 42 | 106 | 96.8 [95–100] | 91.3 [45–100] | +5.6 | 97.6 [90–100] | -0.8 | 95.9 [92–98] | 91.5 [51–100] | +4.4 | 97.3 [92–100] | -1.4 |
| group: spatial | 35 | 136 | 91.4 [77–97] | 81.6 [17–100] | +9.8 | 90.8 [54–100] | +0.6 | 92.4 [78–99] | 79.7 [22–99] | +12.7 | 90.3 [64–99] | +2.1 |
| group: workflow | 60 | 62 | 96.4 [88–100] | 91.5 [88–95] | +4.9 | 90.4 [87–97] | +6.0 | 95.4 [85–100] | 96.2 [92–100] | -0.8 | 94.6 [90–100] | +0.8 |
| out of scope: oos-near | 153 | 0 | 97.3 [95–99] | 81.9 [62–99] | +15.4 | 73.3 [18–97] | +23.9 | — | — | — | — | — |
| out of scope: oos-far | 92 | 0 | 99.8 [99–100] | 98.9 [97–100] | +0.9 | 98.2 [93–100] | +1.6 | — | — | — | — | — |
| out of scope: oos-modality | 62 | 0 | 57.3 [40–82] | 51.1 [27–95] | +6.2 | 44.4 [3–77] | +12.8 | — | — | — | — | — |
| tag: multi-arg | 1123 | 4687 | 96.0 [93–98] | 88.3 [58–98] | +7.7 | 94.1 [82–99] | +1.9 | 94.3 [90–97] | 86.7 [56–96] | +7.6 | 91.5 [78–97] | +2.7 |
| tag: distractor | 938 | 3970 | 96.4 [93–98] | 89.8 [63–98] | +6.6 | 94.4 [83–98] | +1.9 | 94.5 [92–97] | 88.2 [60–96] | +6.3 | 92.4 [80–98] | +2.1 |
| tag: number | 778 | 3538 | 95.3 [92–98] | 87.8 [57–98] | +7.5 | 93.2 [79–98] | +2.1 | 93.7 [90–97] | 86.2 [53–96] | +7.6 | 90.9 [77–97] | +2.9 |
| tag: quoting | 356 | 1430 | 97.3 [94–99] | 90.9 [68–98] | +6.3 | 95.6 [86–98] | +1.7 | 96.6 [92–98] | 89.4 [65–97] | +7.2 | 93.8 [84–97] | +2.8 |
| tag: minimal | 355 | 687 | 96.5 [94–99] | 87.9 [64–94] | +8.6 | 90.9 [75–97] | +5.6 | 96.4 [95–98] | 88.1 [59–96] | +8.4 | 91.5 [72–98] | +4.9 |
| tag: path-form | 314 | 1013 | 94.7 [91–98] | 88.0 [61–96] | +6.8 | 92.4 [81–97] | +2.4 | 92.3 [88–96] | 85.5 [57–95] | +6.8 | 89.8 [76–97] | +2.5 |
| tag: list | 284 | 1217 | 95.7 [93–98] | 91.2 [66–99] | +4.4 | 95.7 [89–99] | -0.0 | 89.9 [82–95] | 88.3 [63–96] | +1.6 | 91.0 [83–97] | -1.1 |
| tag: style | 280 | 955 | 95.5 [92–99] | 88.5 [59–97] | +7.0 | 93.1 [79–97] | +2.4 | 94.1 [90–98] | 87.2 [54–97] | +6.8 | 92.1 [76–98] | +1.9 |
| tag: paraphrase | 245 | 1150 | 94.9 [91–98] | 86.9 [47–99] | +8.0 | 94.1 [79–100] | +0.7 | 92.9 [88–97] | 84.6 [44–98] | +8.3 | 90.9 [73–98] | +2.0 |
| tag: identifier | 239 | 899 | 90.0 [84–95] | 91.2 [70–97] | -1.2 | 92.4 [84–99] | -2.4 | 84.9 [77–93] | 88.8 [62–96] | -3.8 | 89.0 [79–98] | -4.1 |
| tag: noise | 197 | 728 | 96.4 [93–99] | 87.9 [62–96] | +8.5 | 92.0 [83–98] | +4.4 | 95.0 [92–97] | 87.2 [61–96] | +7.8 | 90.4 [80–95] | +4.7 |
| tag: negation | 177 | 850 | 93.0 [86–97] | 87.9 [51–98] | +5.1 | 93.7 [83–99] | -0.7 | 91.1 [83–96] | 87.0 [51–97] | +4.1 | 90.6 [77–98] | +0.5 |
| tag: alias | 166 | 826 | 96.2 [93–98] | 89.6 [52–99] | +6.6 | 97.1 [85–100] | -0.9 | 92.9 [90–96] | 88.3 [54–99] | +4.6 | 93.7 [81–99] | -0.8 |
| tag: multi-step | 142 | 441 | 92.0 [86–96] | 91.9 [77–99] | +0.1 | 94.4 [88–99] | -2.3 | 90.8 [84–94] | 91.7 [77–99] | -0.8 | 93.2 [87–98] | -2.4 |
| tag: derived | 84 | 315 | 94.6 [89–98] | 86.4 [55–100] | +8.3 | 93.7 [75–99] | +1.0 | 92.8 [87–97] | 81.3 [50–95] | +11.4 | 86.8 [70–93] | +6.0 |
| tag: conditional | 17 | 51 | 100.0 [100–100] | 99.3 [94–100] | +0.7 | 99.3 [94–100] | +0.7 | 100.0 [100–100] | 98.3 [92–100] | +1.7 | 96.9 [84–100] | +3.1 |

## 6. Underspecified requests (174, pooled; correct = right tool, stated values right, the missing values left unfilled)

| system | no tool call | wrong tool | right tool, missing value invented | right tool, missing value left out (correct) | right tool, other |
|---|---|---|---|---|---|
| System 1 decider-4b | 5 | 3 | 5 | 150 | 11 |
| System 1 pplx-decider-flash | 1 | 3 | 6 | 157 | 7 |
| System 1 decider-35b-a3b | 5 | 8 | 2 | 153 | 6 |
| System 1 OpenJev | 4 | 4 | 3 | 160 | 3 |
| System 1 Clef | 1 | 4 | 3 | 163 | 3 |
| System 1 pplx-decider-27b | 2 | 2 | 3 | 163 | 4 |
| lexical baseline (no model) | 2 | 46 | 46 | 39 | 41 |
| NVIDIA Nemotron 3.5 Bolt E4B [think] | 166 | 7 | 0 | 1 | 0 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | 138 | 12 | 24 | 0 | 0 |
| Qwen3.8-27B [think] | 81 | 31 | 62 | 0 | 0 |
| Meta Muse Glimmer 30B [think] | 105 | 21 | 48 | 0 | 0 |
| Google Gemma 4 E4B (it) [think] | 166 | 7 | 1 | 0 | 0 |
| Google Gemma 4 26B-A4B (it) [think] | 130 | 21 | 23 | 0 | 0 |
| Qwen3.5-4B [think] | 129 | 9 | 35 | 1 | 0 |
| Qwen3.5-27B [think] | 126 | 10 | 38 | 0 | 0 |
| Qwen3.5-35B-A3B [think] | 141 | 7 | 26 | 0 | 0 |
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | 115 | 7 | 50 | 2 | 0 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | 23 | 11 | 138 | 2 | 0 |
| Qwen3.8-27B [nothink] | 17 | 16 | 130 | 10 | 1 |
| Meta Muse Glimmer 30B [nothink] | 124 | 17 | 33 | 0 | 0 |
| Google Gemma 4 E4B (it) [nothink] | 147 | 3 | 20 | 4 | 0 |
| Google Gemma 4 26B-A4B (it) [nothink] | 33 | 21 | 110 | 9 | 1 |
| Qwen3.5-4B [nothink] | 54 | 10 | 110 | 0 | 0 |
| Qwen3.5-27B [nothink] | 106 | 5 | 63 | 0 | 0 |
| Qwen3.5-35B-A3B [nothink] | 62 | 5 | 106 | 1 | 0 |

## 7. Cost per request (every request of every set, 2126; whole-board GPU energy) and per fully correct request (over the 1952)

| system | s/request mean (median) | GPU energy J/request (above idle) | context tokens (LLM) | prompt tokens computed | generated tokens (reasoning) | model calls | Wh total | hours total | Wh per fully correct | s per fully correct | mean power W (idle W) | peak GPU memory MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| System 1 decider-4b | 1.17 (1.21) | 210 (148) | — | 5963 | 13 (—) | 11.9 | 123.8 | 0.69 | 0.072 | 1.45 | 179 (52) | 7368 |
| System 1 pplx-decider-flash | 4.59 (4.50) | 441 (201) | — | 6479 | 0 (—) | 11.9 | 260.2 | 2.71 | 0.137 | 5.14 | 100 (52) | 12294 |
| System 1 decider-35b-a3b | 2.24 (2.16) | 413 (294) | — | 5952 | 15 (—) | 11.8 | 243.7 | 1.32 | 0.133 | 2.59 | 178 (53) | 23490 |
| System 1 OpenJev | 2.56 (2.54) | 758 (617) | — | 6308 | 0 (—) | 11.8 | 447.5 | 1.51 | 0.228 | 2.77 | 291 (55) | 21852 |
| System 1 Clef | 3.12 (3.10) | 927 (762) | — | 7859 | 0 (—) | 11.8 | 547.2 | 1.85 | 0.284 | 3.45 | 292 (53) | 31107 |
| System 1 pplx-decider-27b | 10.14 (9.99) | 1367 (795) | — | 6434 | 0 (—) | 11.8 | 807.5 | 5.99 | 0.407 | 10.86 | 153 (56) | 35057 |
| NVIDIA Nemotron 3.5 Bolt E4B [think] | 6.15 (5.77) | 1801 (1374) | 25097 | 80 | 686 (566) | — | 1063.8 | 3.63 | 0.827 | 10.16 | 290 (70) | 12364 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think] | 5.48 (3.97) | 1517 (1136) | 38871 | 100 | 584 (403) | — | 895.6 | 3.24 | 0.467 | 6.08 | 266 (69) | 25515 |
| Qwen3.8-27B [think] | 26.09 (10.45) | 7793 (6085) | 31337 | 90 | 952 (779) | — | 4602.4 | 15.41 | 2.480 | 29.88 | 295 (66) | 34955 |
| Meta Muse Glimmer 30B [think] | 19.82 (16.42) | 5924 (4605) | 32173 | 79 | 733 (560) | — | 3498.4 | 11.71 | 1.899 | 22.88 | 297 (65) | 35212 |
| Google Gemma 4 E4B (it) [think] | 5.85 (5.38) | 1710 (1331) | 25097 | 80 | 656 (563) | — | 1009.9 | 3.45 | 0.685 | 8.42 | 289 (65) | 12798 |
| Google Gemma 4 26B-A4B (it) [think] | 7.87 (4.98) | 2321 (1816) | 25097 | 80 | 711 (590) | — | 1370.7 | 4.65 | 0.752 | 9.19 | 288 (66) | 45341 |
| Qwen3.5-4B [think] | 3.20 (2.96) | 933 (721) | 31299 | 90 | 456 (303) | — | 550.9 | 1.89 | 0.310 | 3.84 | 285 (66) | 12597 |
| Qwen3.5-27B [think] | 11.76 (10.62) | 3494 (2762) | 31299 | 90 | 418 (274) | — | 2063.5 | 6.94 | 1.095 | 13.26 | 295 (62) | 36038 |
| Qwen3.5-35B-A3B [think] | 2.60 (2.34) | 641 (487) | 31299 | 90 | 417 (271) | — | 378.4 | 1.53 | 0.198 | 2.89 | 241 (59) | 27572 |
| NVIDIA Nemotron 3.5 Bolt E4B [nothink] | 1.48 (0.93) | 398 (306) | 25095 | 70 | 150 (0) | — | 235.3 | 0.87 | 0.139 | 1.86 | 249 (62) | 14362 |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink] | 1.87 (1.35) | 461 (345) | 38871 | 100 | 169 (0) | — | 272.2 | 1.11 | 0.166 | 2.42 | 226 (62) | 26930 |
| Qwen3.8-27B [nothink] | 3.71 (2.86) | 1081 (852) | 31301 | 92 | 127 (0) | — | 638.4 | 2.19 | 0.362 | 4.47 | 284 (62) | 35976 |
| Meta Muse Glimmer 30B [nothink] | 9.25 (8.34) | 2751 (2209) | 32173 | 79 | 340 (168) | — | 1624.8 | 5.46 | 0.853 | 10.32 | 295 (58) | 36030 |
| Google Gemma 4 E4B (it) [nothink] | 1.32 (0.91) | 349 (268) | 25095 | 70 | 132 (0) | — | 206.2 | 0.78 | 0.135 | 1.83 | 246 (62) | 14208 |
| Google Gemma 4 26B-A4B (it) [nothink] | 1.49 (1.00) | 405 (312) | 25099 | 84 | 123 (0) | — | 239.1 | 0.88 | 0.142 | 1.88 | 253 (62) | 45532 |
| Qwen3.5-4B [nothink] | 1.22 (0.92) | 337 (253) | 31301 | 92 | 164 (0) | — | 199.2 | 0.72 | 0.117 | 1.52 | 261 (69) | 12044 |
| Qwen3.5-27B [nothink] | 4.54 (3.32) | 1330 (1055) | 31301 | 92 | 156 (0) | — | 785.2 | 2.68 | 0.413 | 5.07 | 286 (60) | 36002 |
| Qwen3.5-35B-A3B [nothink] | 1.07 (0.75) | 247 (175) | 31301 | 92 | 157 (0) | — | 145.8 | 0.63 | 0.080 | 1.25 | 215 (67) | 27573 |

## 8. Checks

Scorer check: System 1 records whose rescored verdicts (tool, call_correct, value verdicts) differ from the verdicts stored with the run, per set.

| system | imaging_test | external | biomedical_test |
|---|---|---|---|
| System 1 decider-4b | 0 | 0 | 0 |
| System 1 pplx-decider-flash | 0 | 0 | 0 |
| System 1 decider-35b-a3b | 0 | 0 | 0 |
| System 1 OpenJev | 0 | 0 | 0 |
| System 1 Clef | 0 | 0 | 0 |
| System 1 pplx-decider-27b | 0 | 0 | 0 |
| lexical baseline (no model) | 0 | 0 | 0 |

LLM completions (all requests of every set): NVIDIA Nemotron 3.5 Bolt E4B [think]: errors 0, no call 1065; NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [think]: errors 0, no call 398; Qwen3.8-27B [think]: errors 0, no call 294; Meta Muse Glimmer 30B [think]: errors 0, no call 323; Google Gemma 4 E4B (it) [think]: errors 0, no call 860; Google Gemma 4 26B-A4B (it) [think]: errors 0, no call 394; Qwen3.5-4B [think]: errors 0, no call 379; Qwen3.5-27B [think]: errors 0, no call 375; Qwen3.5-35B-A3B [think]: errors 0, no call 424; NVIDIA Nemotron 3.5 Bolt E4B [nothink]: errors 0, no call 435; NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB release) [nothink]: errors 0, no call 199; Qwen3.8-27B [nothink]: errors 0, no call 144; Meta Muse Glimmer 30B [nothink]: errors 0, no call 381; Google Gemma 4 E4B (it) [nothink]: errors 0, no call 661; Google Gemma 4 26B-A4B (it) [nothink]: errors 0, no call 321; Qwen3.5-4B [nothink]: errors 0, no call 298; Qwen3.5-27B [nothink]: errors 0, no call 365; Qwen3.5-35B-A3B [nothink]: errors 0, no call 309
