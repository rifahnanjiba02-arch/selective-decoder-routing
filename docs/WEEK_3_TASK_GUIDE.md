# Week 3 task guide: four major tasks

Week 2 produced [classical-router validation results](../results/model_scores/week2_classical.md). The newest [decoder runtime experiment](../notebooks/decoder_runtime_motivation.ipynb) found correlation-enabled PyMatching slower than standard PyMatching in six configurations on one Colab CPU. At the provisional 0.5 threshold, Logistic Regression used correlated matching on 44.10% of validation shots, but **44.10% usage is not a measured time saving**. Week 3 must answer whether the router's lower decoder use saves time after feature extraction and model inference.

One member can choose each task below. Tasks 1–3 can begin together; Task 4 uses their frozen outputs. The [seven-day schedule](WEEK_3_SCHEDULE.md) gives handoff dates.

| Task | Member's main question | Main files | Required handoff |
| --- | --- | --- | --- |
| **1. Decoder cost and timing method** | Can we time PM and CM on the exact shots and decoder implementations used for router decisions? | `notebooks/decoder_runtime_motivation.ipynb`, new `src/router/benchmark.py`, `results/tables/week3_runtime_method.md` | A checked shot/decoder alignment report and a reusable timing harness for always-PM, always-CM, and routed decoding. |
| **2. QML router candidate** | Does a bounded VQC provide a useful additional routing option? | `src/router/models/qml.py`, `notebooks/03_qml_router.ipynb`, `results/model_scores/week3_qml.md` | Validation scores on the same 10,000 rows as the classical models, or a timed and documented stopping result by Day 3. |
| **3. Validation tradeoff and model freeze** | Which model and threshold give the best LER within the correlated-matching budget? | `src/router/evaluate.py`, `src/router/plots.py`, `notebooks/04_routing_tradeoff.ipynb`, `artifacts/models/`, `results/figures/` | LER-versus-CM-usage curves, a saved fitted router with preprocessing, and a machine-readable freeze record before test access. |
| **4. Final benefit and time-saving evaluation** | Does the frozen router improve accuracy at lower CM use, and does it run faster than always-CM? | `notebooks/05_final_evaluation.ipynb`, `results/model_scores/week3_final.md`, `results/figures/`, `README.md`, `paper/README.md` | Load the saved router for one held-out comparison plus a measured LER-versus-runtime curve **if Task 1 establishes aligned decoding**; otherwise a clear limit on the time-saving claim. |

## Shared evaluation rules

- Use dataset revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3`. Reuse 50,000 train and 10,000 validation rows, independently sampled with seed 42. `configs/data.yaml` fixes a 100,000-row test sample with seed 42. Keep test closed until Task 3 freezes the candidate set, threshold, settings, and cost-curve method.
- The claim concerns new shots from the **published experiments**, not unseen experiments. Exclude `label`, decoder predictions, target, and `experiment` from router inputs. Train on train; choose models and thresholds on validation; run test once.
- Use a 50% CM-usage budget for the primary operating point. Among validation model/threshold pairs within budget, choose the lowest system LER. Break ties by lower usage, then by model order: always-PyMatching, Logistic Regression, Random Forest, XGBoost, Dummy, VQC; then by the higher threshold. Enumerate distinct validation score thresholds with `>=`, include threshold 0 for all-CM, and include the explicit always-PM baseline.
- At every reported point show system LER, CM usage, rescues captured, and harmful switches. Compare LER with both fixed-decoder baselines. The oracle is a reference bound, not a deployable result. Report paired uncertainty for final LER differences and subgroup diagnostics by basis, distance, and rounds.

## Task 1 — Decoder cost and timing method

**What to do:** extend the merged runtime notebook into a method the final router evaluation can call. The saved benchmark already reports PM and CM time for the same archive shots. Its reconstructed CM predictions differ from the archive's stored CM predictions on 381–2,579 of 10,000 sampled shots per configuration; reconstructed PM also differs on 304–338 shots for the two distance-5, 25-round settings. First check whether the relevant Hugging Face rows can be matched to archive detector events, labels, and configurations. Record match counts and any prediction mismatch. Do not combine stored-prediction LER with measured-decoder runtime.

Build a reusable harness that accepts the **same ordered raw shots**, a fitted router, and a frozen threshold. Primary mode is sequential per-shot routing, because the router chooses before decoding each shot. Time three paths on the same CPU: always PM, always CM, and feature extraction + model inference + decision + selected decoder. Preload data and build decoder graphs outside all three timers; include any batching or grouping overhead if a batch result is also reported. Alternate run order, warm up equally, repeat timing blocks, and record median, spread, package versions, CPU, experiment, shot count, and decoder construction time separately. Check routed predictions against direct decoding on the corresponding selected shots.

**Deliver:** alignment report, timing harness with focused checks, and a compact method note. Hand the harness and alignment verdict to Task 4 by Day 5. If archived predictions do not match PyMatching 2.4.0, Task 4 may still measure an end-to-end system using **recomputed** PM/CM predictions and LER on the aligned shots, clearly labeled as that implementation. It must not present that as the stored-decoder test result.

## Task 2 — QML router candidate

**What to do:** fit a small VQC on leakage-safe compact features and the same rescue target. Fit preprocessing on train only. Predeclare the subset, qubit count, circuit, optimizer, seed, and stopping budget; record wall-clock training and scoring time. Score the same 10,000 validation rows if feasible so Task 3 can compare it fairly with the classical models.

**Deliver:** notebook 03, reusable QML code, and a compact validation note by Day 3. If the run cannot finish within its recorded budget, document the limit and let Task 3 freeze the classical candidates. QML work must not delay the test freeze.

## Task 3 — Validation tradeoff and model freeze

**What to do:** fit or load candidates using the recorded train sample and score them on the same validation rows. Sweep thresholds and plot **LER versus CM usage**. Include always-PM, always-CM, and oracle references. At the selected point report LER improvement over always-PM, LER gap to always-CM, CM calls per 1,000 shots, rescues, and harms. State explicitly if no candidate improves on always-PM under the 50% budget.

Use the shared rule above to freeze the model and threshold. Save the **already fitted selected router and its train-fitted preprocessing** under the Git-ignored `artifacts/models/` directory (`joblib` for a classical pipeline; a loadable parameter and preprocessing bundle for VQC). Save a small machine-readable manifest with the model filename, dataset and source revisions, train/validation sample sizes and seed, feature schema, model settings, threshold, package versions, tie rule, and timing mode. If always-PyMatching wins, record that baseline choice without a model file. Reload the saved model in Colab and confirm its validation scores and decisions match the recorded results. Share the model file and manifest with Task 4 through Drive, WhatsApp, or another team channel. Do this before Task 4 requests test rows; do not commit model binaries or generated feature tables.

**Deliver:** notebook 04, reusable sweep/plot functions with focused tests, validation figure and table, load-checked model artifact, and freeze manifest by Day 4. The validation curve shows the **accuracy-versus-decoder-use** tradeoff; it does not establish time saving.

## Task 4 — Final benefit and time-saving evaluation

**What to do:** load Task 3's saved fitted router and threshold in Colab, then confirm its validation results match the manifest. Do not refit or alter preprocessing on validation or test data. After the freeze, select the predeclared 100,000 test rows once. On those same ordered rows compare always-PM, always-CM, the frozen router, and oracle using stored predictions. Report LER, CM usage, rescues, harms, paired LER differences, and descriptive group results. Produce the final LER-versus-CM-usage figure without changing the operating point after seeing test outcomes.

Then use Task 1's aligned timing setup to answer the runtime question directly. For the same shots and implementation, measure `T_PM`, `T_CM`, and `T_router` **including** feature extraction, inference, decision, and selected decoding. Report absolute time per shot and `100 × (T_CM − T_router) / T_CM` percent saving. Plot LER against measured time per shot for always-PM, always-CM, and the router; show several thresholds fixed from validation if a curve is feasible. Keep batch and sequential results separate. If this timing workload differs from the 100,000 stored-prediction test rows, label it as a separate controlled experiment and report its own LER. The router saves time only where its measured end-to-end time is below always-CM on the same workload.

**Deliver:** notebook 05, final figure and note, README and paper outline update by Day 7. If Task 1 cannot establish aligned shots and a consistent LER for the timed decoder implementations, still complete the stored-prediction LER-versus-usage result and report measured router overhead and a configuration-specific break-even estimate, `(1 − CM usage) × (T_CM − T_PM)`. Label that estimate clearly; do **not** claim an observed time saving or mix it into the stored-prediction LER curve.

## Complete by the end of Week 3

- [ ] Task 1 records shot/prediction alignment and provides a repeatable same-shot timing method.
- [ ] Task 2 hands over a comparable VQC validation result or a documented stopping result by Day 3.
- [ ] Task 3 saves the validation LER-versus-CM-usage curve, load-checks the selected fitted model and preprocessing, and freezes the model filename and threshold before test access.
- [ ] Task 4 runs the held-out comparison once, with identical ordered shots for every stored-prediction method.
- [ ] Task 4 reports **measured end-to-end time saving versus always-CM** when alignment permits; otherwise it states why that claim is not supported and reports the break-even estimate separately.
- [ ] The final note and figures show exactly how much LER improves, how much CM use falls, and whether wall-clock time actually falls.

See the [experiment protocol](EXPERIMENT_PROTOCOL.md) for split and leakage rules.
