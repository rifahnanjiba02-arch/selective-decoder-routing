# Week 2 feature checks

**Status: notebook prepared; Colab run and observed results pending.** [Notebook 01](../../notebooks/01_feature_engineering.ipynb) has code for the full train and validation feature audit, but its cells have not been run. Fill the observed fields below from its saved aggregate outputs; do not treat planned sample sizes as measured counts.

## Run settings

| Item | Prepared value |
| --- | --- |
| Repository source commit | `58b14a3afe79d214ab0c302c79ce80a843624a4b` (Task 1 extractor and loader) |
| Dataset | `Bauxitiego/surface-code-syndromes` |
| Dataset revision | `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3` |
| Published splits requested | `train`, `validation` independently; no `test` request |
| Planned samples | 50,000 train; 10,000 validation |
| Sampling seed | 42 for each named split; published split seed 20260806 |
| Feature configuration | `configs/features.yaml` at the source commit; `max_time_steps: 26` |
| Package versions | Pending Colab output |

The source commit pins reusable code and configuration. When this notebook is run from a later code revision, update the source pin and this note together.

## Feature definitions and column order

`extract_features()` reads only `syndrome`, `round_widths`, `time_steps`, `width`, `distance`, `rounds`, and `basis`. The flattened syndrome is reshaped to `(time_steps, width)`; at step `t`, only the first `round_widths[t]` entries are real. Padded entries contribute to no count or density, even if they contain 1.

| Output | Definition |
| --- | --- |
| `detection_event_count` | Total 1s in real positions. |
| `event_density` | Event count / `sum(round_widths)`; 0.0 when there are zero real positions. |
| `activity_round_00` through `activity_round_25` | Real event count at each time step, including the final measurement step. Short sequences have trailing zeros; more than 26 steps raises an error. |
| `active_rounds` | Number of time steps with one or more real events, including the final measurement step. |
| `maximum_round_activity` | Largest real event count in any one time step. |
| `distance`, `rounds` | Permitted numeric metadata. |
| `basis` | Permitted categorical metadata (`X` or `Z`), retained as a string. |

Exact 33-column order: `detection_event_count`, `event_density`, `activity_round_00`, `activity_round_01`, `activity_round_02`, `activity_round_03`, `activity_round_04`, `activity_round_05`, `activity_round_06`, `activity_round_07`, `activity_round_08`, `activity_round_09`, `activity_round_10`, `activity_round_11`, `activity_round_12`, `activity_round_13`, `activity_round_14`, `activity_round_15`, `activity_round_16`, `activity_round_17`, `activity_round_18`, `activity_round_19`, `activity_round_20`, `activity_round_21`, `activity_round_22`, `activity_round_23`, `activity_round_24`, `activity_round_25`, `active_rounds`, `maximum_round_activity`, `distance`, `rounds`, `basis`.

No scaling or category encoding is fitted in notebook 01. The label, stored decoder predictions, decoder correctness, routing target, and experiment identifier are outside `X` and remain reserved for target construction or evaluation in later work.

## Checks to record after the Colab run

| Check | Train result | Validation result |
| --- | --- | --- |
| Selected rows equal planned size | Pending | Pending |
| Exact configured feature names and order | Pending | Pending |
| Finite numeric values and no missing feature values | Pending | Pending |
| Observed `basis` categories within `X`, `Z` | Pending | Pending |
| Event count, density, and activity agree with independent padding mask | Pending | Pending |
| Padded positions; nonzero padded positions | Pending | Pending |
| Aggregate min, median, mean, max for six summary features | Pending | Pending |
| Basis counts | Pending | Pending |

The notebook also checks the hand-calculated padded-1, all-zero, and zero-real-position examples; verifies that changing only forbidden fields leaves features identical; and asserts that only `train` and `validation` were requested. The Task 1 offline tests already cover these synthetic cases, but the notebook's real-data checks are pending.

## Findings and unresolved questions

No train or validation feature summaries have been measured for Task 2 yet. After running notebook 01, paste only its aggregate outputs here, record any failed assertion or new schema question, and save the executed notebook. Do not commit shot-level rows, syndromes, or feature tables.
