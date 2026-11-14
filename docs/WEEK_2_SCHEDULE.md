# Week 2 schedule: feature engineering and classical routers

Week 2 has four tasks: build safe features, verify them in notebook 01, build classical routers, and compare them in notebook 02. Tasks 1–3 are complete as of 14 November 2026; Task 4's published-sample notebook run remains pending. The [task guide](WEEK_2_TASK_GUIDE.md) gives the definitions and deliverables. Begin with the pinned Week 1 dataset revision, use train for fitting and validation for comparison, and leave test untouched.

| Day | Task | Main work | Checkpoint |
| --- | --- | --- | --- |
| **1** | **1 — Safe features** | Agree on exact feature definitions, the `round_widths` padding mask, and a fixed-length activity representation. List forbidden model inputs. | Feature names, formulas, and leakage boundary are written down. |
| **2** | **1 — Safe features** | Implement `src/router/features.py` and hand-checked `tests/test_features.py`. Include padded `1` values, all-zero events, and invalid shapes in tests. | Feature tests pass; output column names and order are stable. |
| **3** | **2 — Notebook 01** | Load pinned train and validation samples, call the feature code, check row counts, values, missingness, and consistent columns. Save aggregate findings. | `01_feature_engineering.ipynb` runs and `results/tables/week2_feature_checks.md` is drafted. |
| **4** | **3 — Classical routers** | Create train-only preprocessing and a Dummy baseline plus Logistic Regression, Random Forest, and XGBoost setup. Record seeds and model settings. | Each model can fit synthetic train features and return synthetic validation rescue scores. |
| **5** | **3 — Classical routers** | Add offline tests for alignment, score shape, forbidden inputs, and train-only preprocessing using synthetic rows with the Task 1 feature schema. | Model tests pass; published development-sample training and comparison happen in Task 4. |
| **6** | **4 — Notebook 02** | Compare candidates on the same validation rows. Show AP, precision, recall, F1, and provisional 0.5-threshold routing LER, correlated-matching usage, rescues captured, and harmful switches. | `02_classical_routers.ipynb` has readable target and system tables. |
| **7** | **4 — Final review** | Run notebooks 01 then 02 from a fresh Colab runtime. Check settings, leakage, results, and limitations; finish both compact notes. | The [Week 2 checklist](WEEK_2_TASK_GUIDE.md#complete-by-the-end-of-day-7) is satisfied. |

Tasks can overlap. If a model is too slow on the proposed development sample, reduce **train and validation sample sizes with recorded settings** and compare every candidate on the same selected validation rows. Do not use test data to resolve a Week 2 problem. Threshold sweeps and final test evaluation remain later work.
