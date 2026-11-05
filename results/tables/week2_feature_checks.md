# Week 2 feature checks

**Status: scaffold — no Week 2 feature run has been recorded.** Complete this note from the saved outputs of [notebook 01](../../notebooks/01_feature_engineering.ipynb). Follow Tasks 1 and 2 in the [Week 2 guide](../../docs/WEEK_2_TASK_GUIDE.md).

## Run settings to record

| Item | Recorded value |
| --- | --- |
| Repository commit | Pending |
| Dataset name and revision | Pending |
| Train and validation sample sizes | Pending |
| Sampling seed | Pending |
| Feature configuration and fixed activity length | Pending |

## Feature definitions

Record the exact formulas and output column order for event count, density, per-time-step activity, active time steps, maximum activity, distance, rounds, and basis. State how zero real positions and longer-than-expected sequences are handled. Confirm that `round_widths` masks padded positions.

## Checks to report

- [ ] Train and validation row counts match the selected sample sizes.
- [ ] Feature names and order exactly match the configured allowlist-derived schema across train and validation.
- [ ] Numeric values are finite; basis categories are valid.
- [ ] Hand-checked padded and all-zero examples agree with the extractor.
- [ ] Changing only label, decoder predictions, or experiment ID leaves the feature row unchanged.
- [ ] No `label`, stored decoder prediction, correctness flag, `y_router`, or experiment ID appears in model inputs.
- [ ] Only aggregate summaries are saved here; no shot-level rows are committed.

## Findings and unresolved questions

Pending the notebook run. Summarize observed ranges and any failed checks here, and link to the saved notebook output.
