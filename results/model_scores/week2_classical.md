# Week 2 classical-router comparison

**Status: Task 3 model code and offline tests complete; published-sample training and comparison pending.** Complete this Task 4 note from [notebook 02](../../notebooks/02_classical_routers.ipynb). Follow the [Week 2 guide](../../docs/WEEK_2_TASK_GUIDE.md).

## Run settings to record

| Item | Recorded value |
| --- | --- |
| Repository commit | Pending |
| Dataset revision and published splits | Pending |
| Train and validation sample sizes and seed | Pending |
| Feature note and column order | Pending |
| Model versions, seeds, and hyperparameters | Pending |
| Provisional routing threshold | 0.5; verify and label in notebook |

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
