# Week 2 feature checks

**Status: complete.** [Notebook 01](../../notebooks/01_feature_engineering.ipynb) has execution counts for all nine code cells and saved aggregate outputs, with no error outputs. The 50,000 train and 10,000 validation shots passed the feature audit on 8 November 2026. These are feature checks, not router performance results.

## Run settings

| Item | Prepared value |
| --- | --- |
| Repository source commit | `58b14a3afe79d214ab0c302c79ce80a843624a4b` (Task 1 extractor and loader) |
| Saved notebook run commit | `fb9590943d58dc54ae7ec309cc55e30f55d1fed9` |
| Dataset | `Bauxitiego/surface-code-syndromes` |
| Dataset revision | `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3` |
| Published splits requested | `train`, `validation` independently; no `test` request |
| Published split sizes observed | 4,550,000 train; 650,000 validation |
| Samples checked | 50,000 train; 10,000 validation |
| Sampling seed | 42 for each named split; published split seed 20260806 |
| Feature configuration | `configs/features.yaml` at the source commit; `max_time_steps: 26` |
| Package versions | `datasets` 4.8.5, `numpy` 2.4.6, `pandas` 2.2.3, `PyYAML` 6.0.3 |

The source commit pins reusable code and configuration. The dataset loader requested only train and validation rows; the Hugging Face builder also downloaded/prepared test files internally, but the notebook did not select or analyze test rows. When this notebook is rerun from a later code revision, update the source pin and this note together.

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

## Saved checks

| Check | Train result | Validation result |
| --- | --- | --- |
| Selected rows equal planned size | 50,000 / 50,000 | 10,000 / 10,000 |
| Exact configured feature names and order | Passed; 33 columns | Passed; 33 columns |
| Nonfinite numeric values; missing feature values | 0; 0 | 0; 0 |
| Observed `basis` categories | `X`, `Z` | `X`, `Z` |
| Event count, density, and activity agree with independent padding mask | Passed on all 50,000 rows | Passed on all 10,000 rows |
| Padded positions; nonzero padded positions | 514,200; 0 | 103,552; 0 |
| Activity sum and other feature identity checks | Passed | Passed |

The hand-calculated padded-1, all-zero, and zero-real-position examples passed. Changing only label, stored decoder predictions, experiment ID, and other forbidden fields left the synthetic feature row unchanged. The notebook checked that no forbidden field was in `X` and that only `train` and `validation` were requested. No target, decoder prediction, or experiment identifier was used as a feature.

## Aggregate feature summaries

| Split | Feature | Min | Median | Mean | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | Detection event count | 0 | 16 | 21.098880 | 189 |
| Train | Event density | 0 | 0.133333 | 0.133045 | 0.625 |
| Train | Active rounds | 0 | 8 | 8.621060 | 26 |
| Train | Maximum round activity | 0 | 4 | 3.879480 | 16 |
| Train | Distance | 3 | 3 | 3.395480 | 5 |
| Train | Rounds | 1 | 13 | 12.992080 | 25 |
| Validation | Detection event count | 0 | 16 | 21.387800 | 146 |
| Validation | Event density | 0 | 0.134615 | 0.133554 | 0.625 |
| Validation | Active rounds | 0 | 8 | 8.671600 | 26 |
| Validation | Maximum round activity | 0 | 4 | 3.906700 | 16 |
| Validation | Distance | 3 | 3 | 3.403400 | 5 |
| Validation | Rounds | 1 | 13 | 13.029600 | 25 |

| Split | Basis X | Basis Z | Total |
| --- | ---: | ---: | ---: |
| Train | 25,094 | 24,906 | 50,000 |
| Validation | 5,068 | 4,932 | 10,000 |

## Findings and limitations

All recorded checks passed; no new schema question arose in these samples. The observed validation ranges fit the training-derived 26-step activity representation. This within-experiment train/validation audit does not establish performance on unseen experiments or on the untouched test split. No shot-level rows, syndromes, or generated feature tables were committed.
