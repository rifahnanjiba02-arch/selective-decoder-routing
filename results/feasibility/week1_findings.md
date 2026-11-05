# Week 1 feasibility findings

The [executed Colab notebook](../../notebooks/00_dataset_and_feasibility.ipynb) combines the data inspection, decoder outcomes, and routing-target checks. Its saved run executed all 16 code cells in order with no error outputs and passed the final assertions.

## Reproducible run

| Setting | Value |
| --- | --- |
| Dataset | `Bauxitiego/surface-code-syndromes` |
| Dataset revision | `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3` |
| Repository source commit | `2e2b527e00dfc2da59236d50a65abefcbe25f8fa` |
| Selected split | `train` |
| Loaded train rows | 4,550,000 |
| Development sample | 50,000 distinct rows |
| Sampling seed | 42 |
| Published split seed | 20260806 |

The notebook used a temporary config with the dataset revision pinned; `configs/data.yaml` retains `dataset_revision: null`. Repeating the selection matched every field of all 50,000 sampled rows in the same order. The repository's 15 offline unit tests and the notebook's test cell passed.

The dataset card lists 650,000 validation and 1,300,000 test rows; these are published split sizes, not rows selected for this run.

## Sample schema

All 12 expected fields were present. The 11 fields used in this analysis had zero missing rows; `tensor_network` was null in every sampled row and was outside the two-decoder scope. `label`, `pymatching`, and `correlated_matching` each contained only 0 or 1; `basis` contained X or Z; `distance` contained 3 or 5; `rounds` contained odd values from 1 through 25.

All sampled rows passed the checks for positive dimensions, `time_steps == rounds + 1`, syndrome and `round_widths` lengths, valid round widths, binary syndrome values, and zero padding. Of 514,200 padded positions checked, none was nonzero. The [data dictionary](../../docs/DATA_DICTIONARY.md) records field types and full missing counts.

## Decoder outcomes

Each sampled row belongs to exactly one outcome. Counts sum to 50,000.

| Outcome | Count | Share of sample |
| --- | ---: | ---: |
| Both correct | 32,055 | 0.64110 |
| Rescue: PM wrong, CM correct | 4,625 | 0.09250 |
| Harmful switch: PM correct, CM wrong | 3,728 | 0.07456 |
| Both wrong | 9,592 | 0.19184 |

| Metric | Rate |
| --- | ---: |
| PyMatching logical error rate | 0.28434 |
| Correlated-matching logical error rate | 0.26640 |
| Rescue opportunity | 0.09250 |
| Decoder disagreement | 0.16706 |
| Two-decoder oracle logical error rate | 0.19184 |

The positive routing target `y_router = 1` occurs on the 4,625 rescue rows only. It is 0 for both-correct, harmful-switch, and both-wrong rows. The notebook checks the target against the rescue mask. Oracle LER is the fraction where both stored predictions are wrong; it is a lower bound for selecting between them, not the performance of a trained router.

## Scope and limitations

These are measurements on sampled **training** shots from the published experiments. Only train rows were selected; the Hugging Face loader prepared validation and test files internally but no rows from those splits informed this analysis. The provided splits are stratified within experiments, so these results do not establish generalization to unseen experiments. No routing model was trained, and correlated-matching selection fraction is not a measured runtime or energy saving. True labels and stored decoder predictions were used only for targets and evaluation; Week 1 did not implement router input features.

The [Task 2 note](task2_outcomes.md) retains the separately verified decoder counts and rates.
