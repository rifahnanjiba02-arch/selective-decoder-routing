# Week 2 classical-router comparison

**Status: notebook 02 and reusable metric code prepared; published-sample run and results pending.** Complete this Task 4 note from [notebook 02](../../notebooks/02_classical_routers.ipynb) after its Colab run. Follow the [Week 2 guide](../../docs/WEEK_2_TASK_GUIDE.md). No validation result has been measured in this note.

## Run settings to record

| Item | Recorded value |
| --- | --- |
| Repository commit | Pending publication and notebook run; record resolved `SOURCE_REF` commit. |
| Dataset revision and published splits | Planned: `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3`; named `train` and `validation` only. Verify in run. |
| Train and validation sample sizes and seed | Planned: 50,000 train, 10,000 validation, seed 42 independently for each split; published split seed 20260806. Verify in run. |
| Feature note and column order | [Week 2 feature checks](../tables/week2_feature_checks.md); 33 columns in `FEATURE_COLUMNS` order, including 26 activity slots. Verify in run. |
| Model versions, seeds, and hyperparameters | Seed 42; exact settings from `configs/models.yaml` printed by notebook; record installed package versions and resolved source commit after run. |
| Provisional routing threshold | 0.5; diagnostic only. |

## Validation results to add

Record rescue prevalence. Compare Dummy, Logistic Regression, Random Forest, and XGBoost on the **same validation rows** using average precision, precision, recall, and F1. At the provisional 0.5 threshold, compare their final logical error rate, correlated-matching usage, rescues captured, and harmful switches with always-PyMatching, always-correlated-matching, and the two-decoder oracle. Do not enter estimates or training-set scores as validation results.

## Checks and limitations

- [ ] Every model uses the same train features and rescue target.
- [ ] Preprocessing has an explicit feature allowlist with `remainder="drop"`; injected forbidden columns do not reach models.
- [ ] Preprocessing and models are fit on train only; validation-only values do not change fitted train state.
- [ ] Probabilities and final predictions align with the same validation rows.
- [ ] Model or hyperparameter choices made from validation are recorded.
- [ ] Test is untouched; no shot-level rows or model files are committed.

Pending the notebook run. Describe class imbalance, any calibration checks, the within-experiment split limit, and why the 0.5 result is not the final threshold study.
