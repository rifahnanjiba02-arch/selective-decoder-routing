# Week 1 schedule: dataset and decoder feasibility

The assignment called “Day 1” in `cse491_project.md` is planned as a **seven-day task**. Day 1 begins when the team starts; the deadline is the **end of Day 7**. The [task guide](WEEK_1_TASK_GUIDE.md) defines the calculations and final deliverables.

| Day | Main work | Checkpoint by end of day |
| --- | --- | --- |
| **1 — Define the workflow** | Read the dataset card and project protocol. Confirm split names, fields to inspect, sample size, seed, and loader/sampler interface. Outline the Colab notebook. | Agreed sample plan and notebook outline. |
| **2 — Implement the data path** | Build split loading and reproducible training sampling. Set config values. Check the existing decoder and routing-target functions with manual arrays. | Loader and sampler available; expected results for all four outcomes written down. |
| **3 — Inspect real rows** | Sample training rows. Record split sizes, columns, key values, missingness, and shape checks. Update the data dictionary where needed. | Reproducible data sample and schema findings. |
| **4 — Verify calculations** | Run outcome counts and rates on the sample. Test that `y_router = 1` only for PM wrong + CM correct and `0` otherwise. Run `make test`. | Verified four counts, five rates, target rule, and tests. |
| **5 — Build the notebook results** | Connect data loading, inspection, outcome tables, target checks, and explanatory text in `00_dataset_and_feasibility.ipynb`. | Notebook contains the full analysis and a draft interpretation. |
| **6 — Reproduce the run** | Run the notebook top to bottom in a fresh Colab runtime. Fix setup or data-interface problems. Draft the compact findings note. | Reproducible notebook run and findings draft. |
| **7 — Final review** | Check formulas, sample settings, test results, notebook output, and limitations. Finalize the notebook and findings note. | All items in the task guide's completion checklist are satisfied. |

The work may overlap across days. If a checkpoint slips, record the actual state and remaining issue, then continue with the independent parts of the task. Do not present an unverified calculation as a final result.
