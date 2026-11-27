# Week 3 schedule: four member tasks

Each member can choose one of the [four tasks](WEEK_3_TASK_GUIDE.md#week-3-task-guide-four-major-tasks). Tasks 1–3 start in parallel. Task 4 prepares its final notebook early, then uses the frozen model from Task 3 and timing method from Task 1. Keep test closed through Day 5.

| Day | Task 1: decoder timing | Task 2: QML candidate | Task 3: validation tradeoff | Task 4: final evaluation |
| --- | --- | --- | --- | --- |
| **1** | Reproduce the merged benchmark; define shot and prediction alignment checks. | Fix compact features, circuit, sample, seed, and time budget. | Implement threshold sweep and metric checks. | Prepare final notebook structure and paired uncertainty method without test data. |
| **2** | Match archive detector events to dataset configurations and inspect prediction mismatches. | Train the bounded VQC on train only. | Score classical models on validation; plot LER versus CM usage. | Prepare same-shot baseline and reporting code on synthetic data. |
| **3** | Implement same-shot PM, CM, and routed timing harness. | Deliver comparable validation scores or a timed stopping result. | Add any completed VQC scores; close the candidate set. | Check the final report and figure templates. |
| **4** | Verify warm-up, repeated timing, and routed prediction checks. | Document settings and result. | Select the 50%-budget operating point; save the fitted model, preprocessing, threshold, and run settings. | Receive the saved model and freeze manifest; keep test closed. |
| **5** | Deliver timing harness, alignment verdict, and overhead method. | Review the QML limitations in the final narrative. | Reload the model and verify identical validation scores and decisions. | Load the shared model in Colab, confirm its validation results, and integrate the timing harness. |
| **6** | Help diagnose runtime measurements if needed. | Review QML comparison wording. | Audit that no threshold changes follow test access. | Run the single predeclared test comparison and aligned end-to-end timing, if available. |
| **7** | Check the timing interpretation. | Check the QML result. | Check the accuracy-versus-usage curve. | Finish LER-versus-runtime result or stated limit, final note, figures, README, and paper outline. |

The final time-saving claim requires `T_router < T_CM` from repeated measurements on the same shots and decoder implementation, with feature extraction and inference included. The saved decoder-only benchmark establishes why routing may help; it does not establish that the router already saves time.
