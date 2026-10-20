# Week 1 schedule: complete the original Day 1 task

The work called **“Day 1”** in `cse491_project.md` is planned here as a **seven-day task**. Day 1 is the start date the team chooses; the deadline is the **end of Day 7**. The scope stays the same: inspect a reproducible training sample, verify the two stored decoders' outcomes, check the rescue target, and present the findings in a Colab notebook. No router model training is part of this week.

The detailed steps and acceptance checks are in the [member work guide](DAY_1_MEMBER_WORK_GUIDE.md). **Member 4** is responsible for the notebook and final integration. The schedule gives Members 1–3 their handoff dates so notebook work can progress throughout the week.

| Day | Member 1 — data | Member 2 — decoder outcomes | Member 3 — target and tests | Member 4 — notebook |
| --- | --- | --- | --- | --- |
| **1: Plan and inspect** | Check dataset card, split names, and planned loader/sampler interface. | Review existing outcome functions and prepare four manual cases. | Review target function and existing tests; write expected results for four cases. | Create notebook outline, introduction, and Colab setup cells. Agree on the sample and interface with Member 1. |
| **2: Build foundations** | Implement split loading and seeded sampling; set train sample size and seed in config. | Verify four masks and metric formulas on manual arrays. | Test that the routing target is 1 for PM wrong + CM correct and 0 for all other outcomes. | Add placeholder cells that import package functions and record run settings. |
| **3: First data handoff** | Run the training sample; report split sizes, columns, value ranges, missingness, and unusual rows. Send exact API calls, size, and seed to Member 4. | Run outcome functions on Member 1's sample once available; check counts sum to `N`. | Run `make test`; check Member 2's first sample summary against the manual formulas. | Connect the loader and sampler; add compact schema and split summaries. |
| **4: Verified metrics handoff** | Resolve data or encoding questions and update the data dictionary. | Send final four counts, five rates, denominator, and any assumptions to Members 3 and 4. | Confirm rescue target and metrics; send test result and corrections to Member 4. | Add outcome tables and notebook assertions; check numbers match the verified functions. |
| **5: Explain findings** | Help resolve any data issues found during notebook integration. | Check the notebook's metric names and interpretation. | Check that the notebook describes rescue and oracle LER correctly. | Write interpretation and draft the compact note in `results/feasibility/`. |
| **6: Reproduce and review** | Confirm another run can reproduce the same sample. | Review final tables against the code output. | Re-run tests and review any fixes. | Run the notebook top to bottom in a fresh Colab runtime; fix setup or integration problems. |
| **7: Finish and hand off** | Confirm dataset notes and config are complete. | Confirm decoder summary is complete. | Confirm tests pass and target definition is correct. | Finalize and commit the notebook and findings note; share the run settings and results with the team. |

## End-of-week deliverables

- **Member 1:** `load_surface_code_dataset()`, `sample_split()`, config values, and documented schema/split findings.
- **Member 2:** verified outcome masks and a summary with `both_correct`, `rescue`, `harmful_switch`, `both_wrong`, both decoder LERs, rescue opportunity, decoder disagreement, and oracle LER.
- **Member 3:** tested `route_to_correlated()` and manual tests covering all four outcomes and the summary math; `make test` passes.
- **Member 4:** `notebooks/00_dataset_and_feasibility.ipynb` runs from top to bottom in Colab, and `results/feasibility/` contains a concise findings note with sample size, seed, counts, rates, and limitations.

The team finishes this task by **the end of Day 7**. If a handoff slips, the owner sends the current code, actual findings, and remaining issue to the next member immediately so integration can continue. The final notebook records any unresolved issue plainly instead of presenting incomplete checks as results.
