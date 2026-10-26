# WP-B3 data statistics

Exact metadata from `cache/musique/manifest.json`; selected and dropped IDs remain in that manifest.

Source MuSiQue revision: `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`. Tokenizer revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

Context cap: 640 tokens. No model output enters selection. Counts and coverage below use the default supporting-only setting.

## train

| Hop type | Before cap | Before % | Retained | Retained % | Dropped | Substituted | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2hop | 13655 | 72.10 | 13470 | 73.76 | 185 | 13338 | 99.02 |
| 3hop1 | 3550 | 18.75 | 3386 | 18.54 | 164 | 3254 | 96.10 |
| 3hop2 | 617 | 3.26 | 583 | 3.19 | 34 | 566 | 97.08 |
| 4hop1 | 615 | 3.25 | 428 | 2.34 | 187 | 411 | 96.03 |
| 4hop2 | 121 | 0.64 | 94 | 0.51 | 27 | 92 | 97.87 |
| 4hop3 | 380 | 2.01 | 302 | 1.65 | 78 | 288 | 95.36 |

Token lengths before cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 68 / 237 / 486 / 664 / 2089 | 10 / 21 / 29 / 34 / 51 | 1 / 4 / 10 / 14 / 26 |
| 3hop1 | 126 / 366 / 630 / 830 / 1099 | 12 / 27 / 37 / 41 / 54 | 1 / 4 / 10 / 15 / 18 |
| 3hop2 | 153 / 414 / 648 / 748 / 1803 | 19 / 30 / 40 / 45 / 51 | 1 / 4 / 13 / 16 / 16 |
| 4hop1 | 221 / 537 / 887 / 1054 / 1698 | 18 / 35 / 46 / 50 / 55 | 1 / 4 / 11 / 12 / 17 |
| 4hop2 | 288 / 521 / 1215 / 1281 / 1304 | 27 / 37 / 49 / 50 / 54 | 1 / 7 / 9 / 10 / 12 |
| 4hop3 | 231 / 475 / 812 / 910 / 953 | 19 / 35 / 50 / 60 / 64 | 1 / 3 / 16 / 16 / 16 |

Token lengths after cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 68 / 235 / 461 / 581 / 639 | 10 / 21 / 29 / 34 / 51 | 1 / 4 / 10 / 14 / 26 |
| 3hop1 | 126 / 360 / 563 / 618 / 640 | 12 / 26 / 37 / 42 / 54 | 1 / 4 / 10 / 15 / 18 |
| 3hop2 | 153 / 404 / 609 / 635 / 637 | 19 / 31 / 40 / 45 / 51 | 1 / 4 / 13 / 16 / 16 |
| 4hop1 | 221 / 476 / 614 / 633 / 640 | 18 / 35 / 46 / 50 / 55 | 1 / 4 / 10 / 12 / 17 |
| 4hop2 | 288 / 465 / 612 / 639 / 639 | 27 / 37 / 49 / 54 / 54 | 1 / 7 / 10 / 12 / 12 |
| 4hop3 | 231 / 430 / 612 / 632 / 639 | 19 / 36 / 49 / 58 / 64 | 1 / 3 / 13 / 16 / 16 |

Original paragraph order versus the deduplicated decomposition order:

| Hop type | Examples | Matching orders | Chance expected matches |
| --- | ---: | ---: | ---: |
| 2hop | 13655 | 6746 | 6827.500 |
| 3hop1 | 3550 | 569 | 591.667 |
| 3hop2 | 617 | 106 | 102.833 |
| 4hop1 | 615 | 26 | 25.625 |
| 4hop2 | 121 | 3 | 5.042 |
| 4hop3 | 380 | 15 | 15.833 |

Fixed generation budget: **26 tokens**, greedy, stop at newline or EOS. Budget includes the longest retained gold or alias target.

## val

| Hop type | Before cap | Before % | Retained | Retained % | Dropped | Substituted | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2hop | 721 | 72.10 | 718 | 74.40 | 3 | 707 | 98.47 |
| 3hop1 | 187 | 18.70 | 174 | 18.03 | 13 | 169 | 97.13 |
| 3hop2 | 33 | 3.30 | 31 | 3.21 | 2 | 29 | 93.55 |
| 4hop1 | 33 | 3.30 | 22 | 2.28 | 11 | 22 | 100.00 |
| 4hop2 | 6 | 0.60 | 4 | 0.41 | 2 | 4 | 100.00 |
| 4hop3 | 20 | 2.00 | 16 | 1.66 | 4 | 13 | 81.25 |

Token lengths before cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 67 / 230 / 472 / 603 / 1156 | 11 / 21 / 29 / 33 / 45 | 1 / 4 / 10 / 13 / 19 |
| 3hop1 | 126 / 357 / 680 / 1053 / 1068 | 13 / 27 / 37 / 40 / 43 | 1 / 4 / 10 / 15 / 17 |
| 3hop2 | 158 / 358 / 677 / 764 / 764 | 23 / 33 / 41 / 49 / 49 | 1 / 4 / 13 / 16 / 16 |
| 4hop1 | 301 / 551 / 1040 / 1094 / 1094 | 26 / 35 / 48 / 53 / 53 | 2 / 6 / 11 / 16 / 16 |
| 4hop2 | 378 / 547 / 1258 / 1258 / 1258 | 31 / 35 / 41 / 41 / 41 | 1 / 2 / 7 / 7 / 7 |
| 4hop3 | 169 / 554 / 768 / 855 / 855 | 26 / 36 / 44 / 63 / 63 | 2 / 5 / 16 / 16 / 16 |

Token lengths after cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 67 / 229 / 462 / 582 / 640 | 11 / 21 / 29 / 33 / 45 | 1 / 4 / 10 / 13 / 19 |
| 3hop1 | 126 / 351 / 566 / 617 / 624 | 13 / 27 / 37 / 40 / 43 | 1 / 4 / 10 / 15 / 17 |
| 3hop2 | 158 / 355 / 626 / 636 / 636 | 23 / 33 / 41 / 49 / 49 | 1 / 4 / 13 / 16 / 16 |
| 4hop1 | 301 / 472 / 618 / 621 / 621 | 26 / 33 / 47 / 53 / 53 | 2 / 6 / 11 / 11 / 11 |
| 4hop2 | 378 / 522 / 588 / 588 / 588 | 32 / 35 / 41 / 41 / 41 | 2 / 7 / 7 / 7 / 7 |
| 4hop3 | 169 / 541 / 637 / 637 / 637 | 26 / 38 / 63 / 63 / 63 | 2 / 5 / 16 / 16 / 16 |

Original paragraph order versus the deduplicated decomposition order:

| Hop type | Examples | Matching orders | Chance expected matches |
| --- | ---: | ---: | ---: |
| 2hop | 721 | 338 | 360.500 |
| 3hop1 | 187 | 37 | 31.167 |
| 3hop2 | 33 | 7 | 5.500 |
| 4hop1 | 33 | 3 | 1.375 |
| 4hop2 | 6 | 0 | 0.250 |
| 4hop3 | 20 | 0 | 0.833 |

Fixed generation budget: **19 tokens**, greedy, stop at newline or EOS. Budget includes the longest retained gold or alias target.

## test

| Hop type | Before cap | Before % | Retained | Retained % | Dropped | Substituted | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2hop | 409 | 50.62 | 407 | 55.30 | 2 | 401 | 98.53 |
| 3hop1 | 193 | 23.89 | 168 | 22.83 | 25 | 164 | 97.62 |
| 3hop2 | 62 | 7.67 | 48 | 6.52 | 14 | 48 | 100.00 |
| 4hop1 | 92 | 11.39 | 74 | 10.05 | 18 | 73 | 98.65 |
| 4hop2 | 22 | 2.72 | 14 | 1.90 | 8 | 14 | 100.00 |
| 4hop3 | 30 | 3.71 | 25 | 3.40 | 5 | 25 | 100.00 |

Token lengths before cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 77 / 231 / 466 / 567 / 893 | 11 / 21 / 30 / 34 / 41 | 1 / 4 / 10 / 16 / 24 |
| 3hop1 | 209 / 418 / 831 / 986 / 1598 | 17 / 29 / 38 / 44 / 49 | 1 / 4 / 10 / 13 / 13 |
| 3hop2 | 215 / 478 / 746 / 857 / 857 | 21 / 30 / 40 / 48 / 48 | 2 / 5 / 9 / 9 / 9 |
| 4hop1 | 264 / 478 / 834 / 910 / 910 | 22 / 31 / 40 / 47 / 47 | 1 / 4 / 10 / 13 / 13 |
| 4hop2 | 230 / 451 / 878 / 1010 / 1010 | 27 / 35 / 56 / 61 / 61 | 4 / 5 / 10 / 16 / 16 |
| 4hop3 | 322 / 492 / 748 / 749 / 749 | 25 / 36 / 40 / 42 / 42 | 2 / 5 / 9 / 9 / 9 |

Token lengths after cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 77 / 231 / 436 / 558 / 583 | 11 / 20 / 30 / 34 / 41 | 1 / 4 / 10 / 16 / 24 |
| 3hop1 | 209 / 394 / 611 / 630 / 634 | 17 / 28 / 39 / 44 / 49 | 1 / 5 / 10 / 13 / 13 |
| 3hop2 | 215 / 431 / 591 / 606 / 606 | 22 / 30 / 39 / 41 / 41 | 2 / 5 / 9 / 9 / 9 |
| 4hop1 | 264 / 459 / 616 / 630 / 630 | 22 / 31 / 40 / 47 / 47 | 1 / 3 / 11 / 13 / 13 |
| 4hop2 | 230 / 383 / 640 / 640 / 640 | 27 / 40 / 61 / 61 / 61 | 4 / 9 / 16 / 16 / 16 |
| 4hop3 | 322 / 474 / 604 / 615 / 615 | 25 / 36 / 40 / 42 / 42 | 2 / 5 / 9 / 9 / 9 |

Original paragraph order versus the deduplicated decomposition order:

| Hop type | Examples | Matching orders | Chance expected matches |
| --- | ---: | ---: | ---: |
| 2hop | 409 | 201 | 204.500 |
| 3hop1 | 193 | 27 | 32.167 |
| 3hop2 | 62 | 11 | 10.333 |
| 4hop1 | 92 | 4 | 3.833 |
| 4hop2 | 22 | 2 | 0.917 |
| 4hop3 | 30 | 1 | 1.250 |

Fixed generation budget: **24 tokens**, greedy, stop at newline or EOS. Budget includes the longest retained gold or alias target.

## confirm

| Hop type | Before cap | Before % | Retained | Retained % | Dropped | Substituted | Coverage % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2hop | 843 | 52.39 | 832 | 55.76 | 11 | 825 | 99.16 |
| 3hop1 | 375 | 23.31 | 332 | 22.25 | 43 | 326 | 98.19 |
| 3hop2 | 130 | 8.08 | 114 | 7.64 | 16 | 114 | 100.00 |
| 4hop1 | 154 | 9.57 | 122 | 8.18 | 32 | 122 | 100.00 |
| 4hop2 | 42 | 2.61 | 30 | 2.01 | 12 | 30 | 100.00 |
| 4hop3 | 65 | 4.04 | 62 | 4.16 | 3 | 62 | 100.00 |

Token lengths before cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 78 / 241 / 497 / 667 / 1519 | 11 / 20 / 30 / 35 / 37 | 1 / 4 / 11 / 16 / 18 |
| 3hop1 | 209 / 391 / 785 / 921 / 1822 | 15 / 28 / 38 / 41 / 43 | 1 / 4 / 13 / 13 / 18 |
| 3hop2 | 206 / 445 / 693 / 839 / 984 | 23 / 30 / 41 / 47 / 47 | 2 / 5 / 8 / 9 / 9 |
| 4hop1 | 215 / 450 / 905 / 1042 / 1060 | 21 / 31 / 41 / 46 / 49 | 1 / 4 / 9 / 10 / 13 |
| 4hop2 | 267 / 423 / 884 / 950 / 950 | 26 / 32 / 58 / 60 / 60 | 4 / 5 / 11 / 16 / 16 |
| 4hop3 | 327 / 485 / 615 / 741 / 741 | 24 / 36 / 50 / 55 / 55 | 2 / 5 / 9 / 9 / 9 |

Token lengths after cap: each cell is **min / P50 / P95 / P99 / max**; nearest-rank percentiles.

| Hop type | Context | Question | Gold answer |
| --- | --- | --- | --- |
| 2hop | 78 / 239 / 468 / 553 / 599 | 11 / 20 / 30 / 35 / 37 | 1 / 4 / 10 / 15 / 18 |
| 3hop1 | 209 / 374 / 603 / 630 / 635 | 15 / 27 / 38 / 41 / 43 | 1 / 5 / 13 / 13 / 18 |
| 3hop2 | 206 / 422 / 579 / 626 / 635 | 23 / 30 / 41 / 42 / 44 | 2 / 5 / 8 / 9 / 9 |
| 4hop1 | 215 / 408 / 590 / 636 / 637 | 21 / 31 / 41 / 46 / 49 | 1 / 3 / 9 / 10 / 13 |
| 4hop2 | 267 / 389 / 582 / 583 / 583 | 28 / 33 / 58 / 60 / 60 | 4 / 9 / 11 / 16 / 16 |
| 4hop3 | 327 / 478 / 588 / 615 / 615 | 24 / 36 / 50 / 55 / 55 | 2 / 5 / 9 / 9 / 9 |

Original paragraph order versus the deduplicated decomposition order:

| Hop type | Examples | Matching orders | Chance expected matches |
| --- | ---: | ---: | ---: |
| 2hop | 843 | 411 | 421.500 |
| 3hop1 | 375 | 61 | 62.500 |
| 3hop2 | 130 | 26 | 21.667 |
| 4hop1 | 154 | 7 | 6.417 |
| 4hop2 | 42 | 2 | 1.750 |
| 4hop3 | 65 | 2 | 2.708 |

Fixed generation budget: **18 tokens**, greedy, stop at newline or EOS. Budget includes the longest retained gold or alias target.

## Provenance

MuSiQue preparation commit: `8bc909a966c7505d9483362efd6742a4666a339b`, job `3712971.opbs` on `mg0919`.

Source file blake2b-128:

- train: `4c35d71fae509e0d689429941416e36f` (19938 rows).
- dev: `da564a3ccdf6f1d4a9132b6e410e2a18` (2417 rows).

The derived validation split shares 1449 single-hop IDs with derived training before length filtering. Only the official split boundary carries the source dataset leakage restriction.

## Retention suite

Prepared on Miyabi-G in job `3713132.opbs`, host `mg0856`, source commit `bf95246`. PyArrow 21.0.0 was installed by the orchestrator; no worker-side `uv sync` ran.

All subsets use validation data. ARC-Challenge is the explicitly approved n = 299 exception; every other suite has 500 examples. MMLU was selected from all 1,531 validation rows spanning 57 subjects, not the initially cached medical-only subjects.

| Dataset | Source rows | Unique items | Selected | Cached revision | Output blake2b-128 |
| --- | ---: | ---: | ---: | --- | --- |
| arc_challenge | 299 | 299 | 299 | `210d026faf9955653af8916fad021475a3f00453` | `7ff793e1a1c3bb8c276b3c99f82fd7b9` |
| arc_easy | 570 | 570 | 500 | `210d026faf9955653af8916fad021475a3f00453` | `367299ab2d73a97d768a816b3b81956d` |
| hellaswag | 10042 | 10042 | 500 | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | `fbd1f5cd55bdd45aeb017adbe486efa4` |
| mmlu | 1531 | 1531 | 500 | `c30699e8356da336a370243923dbaf21066bb9fe` | `4bcc657798bc53c3ac0393f57759e8fd` |
| piqa | 1838 | 1838 | 500 | `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | `e9c6ded9f0b3420f7147f7e085ba02b6` |
| winogrande | 1267 | 1267 | 500 | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | `0a1b498de736e6ec0aaf0d84d00824d7` |

IDs are ascending blake2b-128 rank with an ID tie-break. Missing IDs use stable context/choice/config fingerprints; MMLU IDs carry the subject. HellaSwag uses `ind:blake2b(context,choices,config)` because its 10,042 validation rows contain only 9,609 distinct raw `ind` values. The source index alone must not collapse unrelated examples.

The full source file list, source hashes, selected IDs and output hashes are recorded in `cache/retention/manifest.json`.

| Dataset | Cached input path | Input blake2b-128 |
| --- | --- | --- |
| arc_challenge | `ARC-Challenge/validation-00000-of-00001.parquet` | `994769cbc4d8297fee7eed7bcab09813` |
| arc_easy | `ARC-Easy/validation-00000-of-00001.parquet` | `9e61f6f3399ce7ebd2a1810727533855` |
| hellaswag | `data/validation-00000-of-00001.parquet` | `544501ae7c6f6ba173566f17daff816b` |
| mmlu | `all/validation-00000-of-00001.parquet` | `25dab382c6acb50872717e9cb62f4bf8` |
| piqa | `piqa_validation.parquet` | `a53327207c6c1d687c116c725d403704` |
| winogrande | `winogrande_xl/validation-00000-of-00001.parquet` | `50058ebd427c3e162018824e5cbfda45` |

MMLU complete-validation download used `timeout 300 curl` on login/control and the pinned URL `https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/validation-00000-of-00001.parquet`.

MC choices are continuations for log-likelihood scoring of `context + choice`. WinoGrande choices include the sentence suffix after the blank. Data preparation performs no model calls or scoring.
