# Week 2 task guide: features and classical routers

Week 2 builds on the [completed Week 1 analysis](../results/feasibility/week1_findings.md). The goal is to turn each shot's syndrome into a small, safe feature row and compare simple classical models that predict whether correlated matching would rescue PyMatching. [Notebook 01](../notebooks/01_feature_engineering.ipynb) has saved train/validation feature checks. [Notebook 02](../notebooks/02_classical_routers.ipynb) and the classical-model files remain scaffolds. See the [seven-day schedule](WEEK_2_SCHEDULE.md).

## The two things a model needs

- **Inputs `X`:** information available before choosing a decoder, such as syndrome activity, distance, rounds, and basis.
- **Target `y_router`:** `1` only when PyMatching is wrong and correlated matching is correct; `0` otherwise. Compute it with the existing `route_to_correlated()` function.

Build `X` from an explicit source-field allowlist: `syndrome`, `round_widths`, `time_steps`, `width`, `distance`, `rounds`, and `basis`. Use `round_widths`, `time_steps`, and `width` to interpret the syndrome; only the documented derived features and permitted metadata become model columns. Do **not** build `X` by copying a whole dataset row and dropping a few known fields. `label`, `pymatching`, `correlated_matching`, decoder correctness, `y_router`, and `experiment` must never be columns of `X`. Keep labels and decoder predictions in a separate target/evaluation path; `experiment` remains excluded pending a grouping review. Start with the published train and validation splits only. The test split stays closed until final evaluation.

Use the Week 1 dataset revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3` as the starting pin. `configs/data.yaml` proposes 50,000 train rows, 10,000 validation rows, and sampling seed 42 for development. Record the exact revision, sample sizes, seeds, repository commit, feature definitions, and model settings in both notebooks. Sample each named split independently; do not create a new random train/validation split.

| Task | Main files | Required result |
| --- | --- | --- |
| 1. Build safe features | `src/router/features.py`, `configs/features.yaml`, `tests/test_features.py` | Deterministic features with padding masked and hand-checked tests. |
| 2. Verify features in notebook 01 | `notebooks/01_feature_engineering.ipynb`, `results/tables/week2_feature_checks.md` | Saved train/validation checks and a compact feature note. |
| 3. Build classical routers | `src/router/models/classical.py`, `configs/models.yaml`, `tests/test_classical.py` | Dummy, Logistic Regression, Random Forest, and XGBoost trained on the same target. |
| 4. Compare routers in notebook 02 | `notebooks/02_classical_routers.ipynb`, `src/router/evaluate.py`, `results/model_scores/week2_classical.md` | Saved validation comparison, system checks, and a compact findings note. |

**Progress — 8 November 2026:** Tasks 1 and 2 are complete. Notebook 01's saved Colab run checked 50,000 train and 10,000 validation rows with no failed feature assertions; aggregate findings are in [the feature note](../results/tables/week2_feature_checks.md). Tasks 3–4 remain pending.

## Task 1 — Build safe features

**What to do:** implement reusable feature extraction in `src/router/features.py`. A `syndrome` row is flattened; reshape it with `time_steps` and `width`. At time step `t`, only the first `round_widths[t]` positions are real. Ignore the padded positions even if a test row deliberately puts a `1` there.

The current [feature config](../configs/features.yaml) names these candidates:

| Feature | Plain definition |
| --- | --- |
| `detection_event_count` | Total number of `1` values in real syndrome positions. |
| `event_density` | Event count divided by the number of real positions. Define and test the zero-denominator behavior. |
| `activity_across_rounds` | Number of events at each time step. Give models a documented fixed-length representation; pad shorter sequences and never silently truncate longer ones. |
| `active_rounds` | Number of time steps with at least one real event. State clearly that the final measurement time step is included. |
| `maximum_round_activity` | Largest event count in any one time step. |
| `distance`, `rounds`, `basis` | Permitted experiment metadata; encode `basis` as a category in the model pipeline. |

Choose the fixed length for `activity_across_rounds` from training data or a documented schema limit, not by looking at validation or test rows. Check dimensions, lengths, widths, binary syndrome values, and missing fields. If an input does not fit the documented representation, raise a clear error.

**Hand checks:** include a tiny row with real and padded `1` values, a row with no events, and rows with different widths and time steps. Assert the expected event count, density, activity, and exact output column order. Change only `label`, decoder predictions, or `experiment` on a copy of a row and assert that `X` is unchanged. Assert that no forbidden field appears in the feature output.

**Deliver:** one deterministic feature extractor with named numeric activity columns and a `basis` category, a clear definition for every feature, and passing tests. Use the existing `router` package; do not copy feature calculations into notebooks.

**Verified result:** `extract_features()` reads only the seven allowlisted source fields and returns 33 stable columns. It masks padding, defines zero-position density as 0, pads activity to 26 time steps without truncation, and rejects invalid shapes, nonpositive `distance`/`rounds`, or `time_steps != rounds + 1`. Fifteen feature tests pass, including forbidden-field mutation and metadata checks; the full offline suite has 30 passing tests. Real train/validation feature checks belong to Task 2.

## Task 2 — Verify features in notebook 01

**What to do:** complete the `01_feature_engineering.ipynb` scaffold. Load and sample the named `train` and `validation` splits with the existing loader and sampler; call the Task 1 extractor on both. Show the run settings, small aggregate feature summaries, and checks that both splits have the exact configured feature names and order, expected row counts, finite numeric values, and valid `basis` categories. Check that padded positions do not contribute to counts or density. Assert that no `test` split was requested.

Keep target construction separate from feature extraction. If any encoding or scaling is shown in this notebook, fit it on train only and apply the same fitted transformation to validation. Do not print or commit shot-level syndrome or feature tables.

**Deliver:** a notebook that runs top to bottom in Colab with saved outputs, plus `results/tables/week2_feature_checks.md` containing the feature definitions, sample settings, aggregate checks, and any unresolved schema questions.

**Verified result:** All nine code cells have execution counts and no error outputs. Both published development samples produced the configured 33 columns, finite numeric values, no missing feature values, and `X`/`Z` basis categories. Independent masking checks passed on every sampled row; the synthetic padding, zero-event, and forbidden-field mutation checks passed. No test rows were selected. See the [aggregate findings](../results/tables/week2_feature_checks.md).

## Task 3 — Build classical routers

**What to do:** implement reusable model setup and fitting in `src/router/models/classical.py`. Use one feature matrix from Task 1 and the same rescue target for every model. Fit on train only. Compare a `DummyClassifier` baseline, Logistic Regression, Random Forest, and XGBoost as listed in [the model config](../configs/models.yaml).

Fit any imputer, scaler, one-hot encoder, feature selector, or probability calibrator on train only. Give the preprocessing pipeline an explicit list of allowed feature columns and use `remainder="drop"`; never pass through unexpected columns. The `basis` category needs encoding before models use it. Set and record random seeds and hyperparameters. Each candidate should return one rescue probability in `[0, 1]` per validation row (for example, from `predict_proba`) so later routing decisions can be made. Do not describe those probabilities as calibrated unless calibration was actually checked.

**Hand checks:** use a tiny synthetic train/validation example containing both target classes to verify feature/target alignment and valid score shape and range. Inject a forbidden column and confirm it is rejected before fitting or dropped by the explicit preprocessor; it must never reach a model. Include a category or extreme numeric value only in validation and confirm the fitted preprocessing state still comes only from train. Do not fit on validation or access test.

**Deliver:** reproducible training and prediction functions, configuration for each candidate, and passing offline tests.

## Task 4 — Compare routers in notebook 02

**What to do:** complete the `02_classical_routers.ipynb` scaffold. Reuse notebook 01's feature definitions and sampling settings. Call the Task 3 functions, train on train, and compare all four candidates on the same validation rows. Report the validation rescue prevalence and each model's average precision (AP), plus precision, recall, and F1 at the provisional 0.5 threshold. AP is useful because rescue rows are a minority; use and name `average_precision_score` from scikit-learn.

At a clearly labeled **provisional threshold of 0.5**, use `choose_correlated()` and `routed_prediction()` to compare always-PyMatching, always-correlated-matching, each router, and the two-decoder oracle on those same validation rows. Report logical error rate, fraction routed to correlated matching, number of rescue rows captured, and number of harmful switches. Put reusable metric calculations in `src/router/evaluate.py`; keep the notebook focused on presentation.

Logical error rate is the fraction of final predictions that disagree with the true label. Correlated-matching usage is the fraction of rows sent to that decoder. A captured rescue is a rescue row sent to correlated matching; a harmful switch is a row where PyMatching was correct, correlated matching was wrong, and the router switched. The oracle is a reference that chooses the correct stored prediction whenever either decoder is correct; keep PyMatching when both choices are equally correct or equally wrong.

The 0.5 result is a diagnostic, not the final operating threshold. The full threshold sweep belongs in notebook 04, and the untouched test evaluation belongs in notebook 05. Record any validation-based model or hyperparameter choices; validation scores used to make those choices are development estimates, not an unbiased final result. Do not claim performance on unseen experiments from these within-experiment splits.

**Deliver:** a top-to-bottom Colab run with readable model and system tables, plus `results/model_scores/week2_classical.md` containing settings, validation results, and limitations. Save aggregate results only; keep model files and shot-level data out of Git.

## Complete by the end of Day 7

- [x] Feature code masks padding, returns stable named columns, and passes hand-checked tests.
- [x] Notebook 01 runs on pinned train and validation samples and saves feature checks.
- [ ] All four classical candidates train on the same train target and produce aligned validation scores.
- [ ] Notebook 02 reports target-class and system metrics on the same validation rows.
- [ ] Both compact findings notes record revision, sample settings, feature definitions, model settings, and limitations.
- [ ] The feature allowlist and model preprocessing exclude target, decoder, and experiment fields; mutation and validation-only-value tests prove the boundary.
- [ ] No test rows guide development; no true label, stored decoder prediction, correctness flag, routing target, or experiment identifier enters `X`.

For the full split and leakage rules, use the [experiment protocol](EXPERIMENT_PROTOCOL.md) and [data dictionary](DATA_DICTIONARY.md).
