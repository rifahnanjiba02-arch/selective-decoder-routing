# Week 2 classical-router validation results

**Status: Week 2 comparison complete.** [Notebook 02](../../notebooks/02_classical_routers.ipynb) has execution counts for all nine code cells and saved Colab results for all four classical routers, the target and system tables, and the final feature-boundary check. It contains no saved error output. These are development results, not final test results.

## Run settings

| Item | Recorded value |
| --- | --- |
| Project source commit | `d6ac6681380c227f0153ab59c5ee32d92526015e` |
| Dataset | `Bauxitiego/surface-code-syndromes` at revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3` |
| Published splits requested | `train`, `validation`; published split seed 20260806 |
| Published split sizes observed | 4,550,000 train; 650,000 validation |
| Samples | 50,000 train; 10,000 validation; independently sampled with seed 42 |
| Input features | The 33 columns below, in `FEATURE_COLUMNS` order; `basis` is categorical |
| Router target | `route_to_correlated()`: 1 only for PyMatching-wrong, correlated-matching-correct rows |
| Model seed | 42 for all four pipelines |
| Provisional decision threshold | 0.5 for precision, recall, F1, and selective routing; AP uses continuous scores |
| Package versions | `datasets` 4.8.5; `numpy` 2.4.6; `pandas` 2.2.3; `PyYAML` 6.0.3; `scikit-learn` 1.6.1; `xgboost` 3.4.1; `tqdm` 4.67.3 |

The source allowlist is `syndrome`, `round_widths`, `time_steps`, `width`, `distance`, `rounds`, and `basis`. The flattened syndrome is reshaped by time step; only the first `round_widths[t]` positions at each step are real. The output order is `detection_event_count`, `event_density`, `activity_round_00` through `activity_round_25`, `active_rounds`, `maximum_round_activity`, `distance`, `rounds`, `basis`. Event density is real event count divided by real positions (0 if there are none); active rounds count time steps with an event, including the final measurement step. The 26 activity slots are fixed from the training schema, with unused trailing slots set to zero and longer rows rejected. See the [feature-check note](../tables/week2_feature_checks.md) for the full feature audit.

## Fixed model settings

All models use train-fitted median imputation for numeric features and train-fitted most-frequent imputation plus one-hot encoding for `basis`. The preprocessor uses an explicit feature allowlist and `remainder="drop"`. Numeric scaling is enabled only for Logistic Regression. No calibrator or feature selector was fitted.

| Model | Settings from `configs/models.yaml` |
| --- | --- |
| Dummy | Prior strategy; constant score equal to train rescue prevalence. |
| Logistic Regression | `C=1.0`, `max_iter=1000`, `class_weight=balanced`, `solver=lbfgs`; numeric scaling. |
| Random Forest | 300 trees, unrestricted depth, `min_samples_leaf=5`, `class_weight=balanced_subsample`, `n_jobs=-1`. |
| XGBoost | 300 trees, depth 4, learning rate 0.1, subsample 0.8, column subsample 0.8, L2 regularization 1.0, histogram tree method, `n_jobs=-1`; `scale_pos_weight` computed from train negatives divided by train positives. |

## Rescue-target results on the same 10,000 validation rows

Train rescue prevalence was 0.0925. Validation rescue prevalence was **0.0967 (967/10,000)**. AP is scikit-learn `average_precision_score` on continuous rescue scores; the other metrics use the provisional 0.5 threshold. Six decimal places reproduce the saved notebook table.

| Model | Average precision | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.096700 | 0.000000 | 0.000000 | 0.000000 |
| Logistic Regression | 0.187496 | 0.163719 | 0.746639 | 0.268551 |
| Random Forest | 0.159806 | 0.166667 | 0.013444 | 0.024880 |
| XGBoost | 0.172334 | 0.155393 | 0.703206 | 0.254539 |

## System results on the same validation rows

Selective routers use the **provisional 0.5 threshold**. Logical error rate (LER) is final-prediction errors divided by 10,000; correlated-matching usage is the fraction sent to that decoder. Rescue capture and harmful switches are counts. The oracle selects correlated matching on rescue rows and keeps PyMatching when both decoders are equally correct or wrong.

| Router | LER | Correlated-matching usage | Rescues captured | Harmful switches |
| --- | ---: | ---: | ---: | ---: |
| Always PyMatching | 0.2816 | 0.0000 | 0 | 0 |
| Always correlated matching | 0.2622 | 1.0000 | 967 | 773 |
| Dummy | 0.2816 | 0.0000 | 0 | 0 |
| Logistic Regression | 0.2693 | 0.4410 | 722 | 599 |
| Random Forest | 0.2817 | 0.0078 | 13 | 14 |
| XGBoost | 0.2712 | 0.4376 | 680 | 576 |
| Two-decoder oracle | 0.1849 | 0.0967 | 967 | 0 |

At 0.5, Logistic Regression has the highest AP and lowest LER among the trained routers. It routes 44.10% of shots to correlated matching, captures 722 rescues, and makes 599 harmful switches. Always using correlated matching has lower LER (0.2622) but uses it for every shot. Random Forest switches on very few rows at this threshold and has one more harmful switch than captured rescue. These observations do not select a final model or operating threshold.

## Checks and limitations

- [x] All four pipelines used the same 50,000 train feature rows and rescue target, then scored the same 10,000 validation rows.
- [x] `X` had the configured 33 columns; forbidden fields were rejected, and every preprocessor used `remainder="drop"`.
- [x] Pipelines were fitted on train only; a temporary validation-only category and extreme numeric value did not change the fitted train transformation or encoded categories.
- [x] Score shapes and ranges, selected decoder predictions, rescue labels, baseline LER, and oracle LER matched the same validation row order.
- [x] The notebook selected only `train` and `validation` rows. The Hugging Face builder also downloaded/prepared test files internally, but the notebook did not select or analyze test rows.
- [x] All nine notebook code cells have execution counts; result cells have saved outputs and no error output.

No model or hyperparameter selection was made from these results. Rescue rows are a minority, and scores were **not checked for probability calibration**. The 0.5 comparison is diagnostic; a validation threshold sweep belongs in notebook 04. The published splits are within-experiment, so these results do not establish performance on unseen experiments. Final evaluation on untouched test rows belongs in notebook 05. No shot-level rows, generated feature tables, or model files are included in the repository.
