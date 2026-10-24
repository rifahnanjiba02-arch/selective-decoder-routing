# Week 1 schedule: dataset and decoder feasibility

The assignment called “Day 1” in `cse491_project.md` is planned as a **seven-day task**. Day 1 begins when the team starts; the deadline is the **end of Day 7**. The [task guide](WEEK_1_TASK_GUIDE.md) defines the calculations and final deliverables.

The loader, sampler, config settings, offline tests, and initial Colab notebook are in place. The 50,000-row train sample has now passed full-row repeat, value/missingness, and syndrome shape/padding checks in a local run; the measured findings are in the data dictionary. The notebook's new check cells can be rerun in Colab. Decoder results and the final findings note remain.

| Day | Main work | Checkpoint by end of day |
| --- | --- | --- |
| **1 — Define the workflow** | Read the dataset card and project protocol. Confirm split names, fields to inspect, sample size, seed, and loader/sampler interface. Outline the Colab notebook. | Agreed sample plan and notebook outline. |
| **2 — Verify the data path** | Use the existing loader, sampler, config values, and offline tests. Review the manual decoder and routing-target examples. | Data functions are already implemented and offline-tested; real-data verification and outcome results remain. |
| **3 — Review completed data inspection** | Reuse the existing 50,000-row train sample and review the measured value sets, missing counts, field types, and syndrome checks in the data dictionary. Rerun the new notebook cells in Colab if needed. | Task 1 results are recorded; no data-path reimplementation is needed. |
| **4 — Verify calculations** | Add outcome counts and rates to the sample analysis. Check that `y_router = 1` only for PM wrong + CM correct and `0` otherwise. The offline tests already passed in Colab. | Verified four counts, five rates, and target rule on the real sample. |
| **5 — Complete the notebook results** | Fill the existing `00_dataset_and_feasibility.ipynb` result and interpretation sections using the package functions. | Notebook contains the full analysis and a draft interpretation. |
| **6 — Reproduce the run** | Run the notebook top to bottom in a fresh Colab runtime. Fix setup or data-interface problems. Draft the compact findings note. | Reproducible notebook run and findings draft. |
| **7 — Final review** | Check formulas, sample settings, test results, notebook output, and limitations. Finalize the notebook and findings note. | All items in the task guide's completion checklist are satisfied. |

The work may overlap across days. If a checkpoint slips, record the actual state and remaining issue, then continue with the independent parts of the task. Do not present an unverified calculation as a final result.
