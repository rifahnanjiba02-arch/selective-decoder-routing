# Week 1 schedule: dataset and decoder feasibility

The assignment called “Day 1” in `cse491_project.md` is planned as a **seven-day task**. Day 1 begins when the team starts; the deadline is the **end of Day 7**. The [task guide](WEEK_1_TASK_GUIDE.md) defines the calculations and final deliverables.

All four Week 1 tasks are complete. The saved [Colab notebook](../notebooks/00_dataset_and_feasibility.ipynb) ran all 16 code cells in order and verified the pinned 50,000-row train sample, schema checks, decoder outcomes, and rescue-only routing target. The [combined findings](../results/feasibility/week1_findings.md) record the run settings, counts, rates, and limitations; field-level findings remain in the [data dictionary](DATA_DICTIONARY.md).

| Day | Main work | Checkpoint by end of day |
| --- | --- | --- |
| **1 — Define the workflow** | Read the dataset card and project protocol. Confirm split names, fields to inspect, sample size, seed, and loader/sampler interface. Outline the Colab notebook. | Agreed sample plan and notebook outline. |
| **2 — Verify the data path** | Use the existing loader, sampler, config values, and offline tests. Review the manual decoder and routing-target examples. | Data functions and the pinned training sample are verified; Task 2's outcome results are recorded. |
| **3 — Review completed data inspection** | Review the saved Colab outputs and measured value sets, missing counts, field types, and syndrome checks in the data dictionary. | Task 1 results are recorded; no data-path reimplementation is needed. |
| **4 — Verify calculations** | Add outcome counts and rates to the sample analysis. Check that `y_router = 1` only for PM wrong + CM correct and `0` otherwise. | Four counts, five rates, and the rescue-only target are verified and shown in the notebook. |
| **5 — Complete the notebook results** | Fill the existing `00_dataset_and_feasibility.ipynb` result and interpretation sections using the package functions. | Notebook contains the full analysis and interpretation. |
| **6 — Reproduce the run** | Run the notebook top to bottom in a fresh Colab runtime. Fix setup or data-interface problems. Integrate the saved Task 2 findings. | Saved Colab run has consecutive execution counts and agrees with the Task 2 findings. |
| **7 — Final review** | Check formulas, sample settings, test results, notebook output, and limitations. Finalize the notebook and findings note. | The task guide's checklist and combined findings note are complete. |

The work may overlap across days. If a checkpoint slips, record the actual state and remaining issue, then continue with the independent parts of the task. Do not present an unverified calculation as a final result.
