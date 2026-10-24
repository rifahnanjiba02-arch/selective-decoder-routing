# Week 1 task guide: dataset and decoder feasibility

The assignment called “Day 1” in `cse491_project.md` is planned as a **one-week task**. By the end of Day 7, the repository should contain a reproducible training sample workflow, verified decoder outcome calculations, a binary routing target, and a Colab notebook explaining the findings. See the [seven-day schedule](WEEK_1_SCHEDULE.md) for checkpoints.

## What this task answers

Each dataset row is one surface-code shot with a true logical `label` and two stored decoder predictions: `pymatching` (PM) and `correlated_matching` (CM). This task checks:

1. Can the dataset and a small training sample be loaded reproducibly?
2. How often is each decoder correct or wrong on that sample?
3. How often could selecting CM rescue PM, and how often would switching hurt?
4. What is the lowest logical error rate possible when selecting between these two stored predictions?

This is a **feasibility analysis**. It does not train a routing model or measure decoder runtime savings.

## Progress snapshot — 24 October 2026

The **Task 1 code foundation is merged into `main`** (pull request #1). The initial Colab notebook is also in `main` and has run through training-sample loading. Use these files and complete the pending inspection and analysis cells rather than rebuilding them.

| Work | Status now | Continue with |
| --- | --- | --- |
| Load a named split | **Implemented and run:** `load_surface_code_dataset()` loaded `train`; the notebook reports 4,550,000 train rows and the expected 12 columns. | Check field values, missingness, and syndrome shapes on the sample. |
| Reproducible sample | **Implemented and run:** `sample_split()` selected 50,000 train rows with seed 42. A repeat matched the `label`, `pymatching`, and `correlated_matching` columns; five offline data tests also pass. | Confirm full-row/order reproducibility if required for the final run record. |
| Sample settings | **Recorded for the notebook run:** repository source commit `f05da60`, dataset revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3`, train sample size 50,000, and seed 42. The repository config still leaves `dataset_revision: null`. | Freeze the notebook's `SOURCE_REF` to the recorded commit before final reproduction; decide whether to pin the shared config. |
| Dataset inspection | **Partly done:** the notebook shows the train count, column list, and five compact sample rows. Validation/test sizes are cited from the dataset card. | Complete value-set, missing-count, data-type, and `round_widths`/syndrome shape checks; update `docs/DATA_DICTIONARY.md`. |
| Decoder outcomes and routing target | **Starter functions and a manual test exist;** the notebook's full offline test command passed. No real-sample counts, rates, or `y_router` values have been calculated. | Verify field encoding and run the functions on the sampled rows. |
| Colab presentation | **Started:** `notebooks/00_dataset_and_feasibility.ipynb` has executed setup, tests, train loading, repeated-sample checks, and a five-row preview. Later analysis sections are headings only; no findings note exists. | Fill the remaining cells, rerun top to bottom, and save a compact findings note. |

The saved notebook output is evidence of the first Colab run, not a completed feasibility result. Its Hugging Face builder downloaded/prepared validation and test files even though the notebook selected only train rows; no validation or test rows were used for analysis.

## Leakage boundary for this task

- The dataset's published train, validation, and test splits are [stratified within each experiment](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes). Use them as provided. Results from these splits concern new shots from the same experiments; a claim about unseen experiments needs a separate group-held-out study.
- `label`, `pymatching`, and `correlated_matching` are present in loaded rows because they are needed for the routing target and outcome evaluation. They must never become router input features. Feature extraction is not implemented in Week 1, so this boundary must be checked again when feature code is written.
- Training samples come from `train`. The loader and sampler require an explicit `allow_test=True` for test access; this is reserved for final evaluation after all choices are frozen. Hugging Face may still prepare other split files internally when loading train, as the saved Colab output shows.

## Tasks and locations

| Task | Where to work | Required result |
| --- | --- | --- |
| 1. Load and inspect data | `src/router/data.py`, `configs/data.yaml`, `docs/DATA_DICTIONARY.md` | Reproducible train sample and documented schema/splits. |
| 2. Calculate decoder outcomes | `src/router/decoders.py` | Four outcome masks/counts and five rates. |
| 3. Verify the routing target | `src/router/labels.py`, `tests/test_routing.py` | `y_router = 1` only for rescue, with hand-checked tests. |
| 4. Present the analysis | `notebooks/00_dataset_and_feasibility.ipynb`, `results/feasibility/` | Runnable Colab notebook and compact findings note. |

Tasks 2 and 3 can be checked on manual arrays while Task 1 prepares real data. Task 4 connects the verified functions and sample.

## Task 1 — Load and inspect the dataset

**Where:** `src/router/data.py` for loading/sampling, `configs/data.yaml` for settings, and `docs/DATA_DICTIONARY.md` for findings.

**Already done in code and the first notebook run:**

- `load_surface_code_dataset()` reads the config, requests one named split from Hugging Face, defaults to `train`, and returns a split mapping plus the config. The notebook has loaded the real train split and confirmed its 4,550,000-row count.
- `sample_split()` selects distinct rows using an explicit size and seed. The notebook created a 50,000-row train sample with seed 42, and a repeated call matched the three stored label/prediction columns. Offline reproducibility and input-check tests passed.
- Both functions require `allow_test=True` before accessing `test`; that option is reserved for final evaluation. The loader's `split` request controls its returned data, while actual download behavior depends on the dataset builder.

**Still to do with real data:**

- Extend the repeat check beyond the three stored label/prediction columns if the final record will claim that every sampled row matches in the same order.
- Keep the notebook's dataset revision pinned and freeze its repository source commit. The validation/test row counts currently come from the dataset card; no validation/test rows were selected for analysis.
- Complete missing counts, data types, and observed values for `label`, `pymatching`, `correlated_matching`, `basis`, `distance`, and `rounds`. The current notebook has only a five-row preview.
- Check `syndrome`, `round_widths`, `time_steps`, and `width` for shape consistency without printing many full syndrome arrays. Record confirmed interpretations and unresolved questions in [the data dictionary](DATA_DICTIONARY.md).

The full dataset, Hugging Face cache, and shot-level output files must stay out of Git.

**Deliver:** the functions, settings, pinned notebook run, train count, column list, and preview already exist. Task 1 is complete only after the remaining value, missingness, and shape checks are recorded and the run can be reproduced from frozen settings.

**Study:** Hugging Face `load_dataset`, train/validation/test splits, and the dataset card referenced in the [data dictionary](DATA_DICTIONARY.md).

## Task 2 — Calculate decoder outcomes

**Where:** `src/router/decoders.py`. The functions `outcome_masks()` and `summarize_outcomes()` already exist as starter code.

**What to do:** verify those functions against the actual value encoding. For each row, compare PM and CM with the true `label` and assign exactly one outcome:

| Outcome | PM | CM | Meaning |
| --- | --- | --- | --- |
| `both_correct` | Correct | Correct | PM already succeeds. |
| `rescue` | Wrong | Correct | Selecting CM fixes PM's error. |
| `harmful_switch` | Correct | Wrong | Selecting CM introduces an error. |
| `both_wrong` | Wrong | Wrong | Neither stored prediction succeeds. |

The four counts must sum to the number of sampled shots, `N`. Report counts and these five rates with `N` as the denominator:

```text
pymatching_ler          = (rescue + both_wrong) / N
correlated_matching_ler = (harmful_switch + both_wrong) / N
rescue_opportunity      = rescue / N
decoder_disagreement    = count(PM prediction != CM prediction) / N
oracle_ler              = both_wrong / N
```

**Rescue opportunity** is the share of rows where CM can fix PM. **Oracle LER** is the share where both decoders fail, even if a selector always chooses the better prediction. They are different quantities.

**Deliver:** `outcome_masks()` and `summarize_outcomes()` verified on manual and sampled rows. Report `both_correct`, `rescue`, `harmful_switch`, `both_wrong`, `pymatching_ler`, `correlated_matching_ler`, `rescue_opportunity`, `decoder_disagreement`, and `oracle_ler` with the sample size `N`.

**Study:** Boolean masks, NumPy array comparisons, logical error rate, and the [project guide](PROJECT_GUIDE.md).

## Task 3 — Verify the routing target

**Where:** `src/router/labels.py` for `route_to_correlated()` and `tests/test_routing.py` for hand-checked tests.

**What to do:** verify the existing function and fix it if the checks fail. The routing target is a **new label derived from** the dataset's true `label` and both decoder predictions. It is not the dataset's logical `label` itself.

| Outcome | Routing target `y_router` |
| --- | ---: |
| PM wrong + CM correct (`rescue`) | **1** |
| PM correct + CM correct | 0 |
| PM correct + CM wrong | 0 |
| PM wrong + CM wrong | 0 |

The function currently returns `True` for rescue and `False` for every other row; these correspond to `1` and `0`. Decoder disagreement alone is **not** the positive target.

Use hand-checked examples in the tests. One useful six-row case is already in `tests/test_routing.py`:

```text
label = [0, 0, 0, 0, 0, 0]
PM    = [0, 1, 0, 1, 1, 0]
CM    = [0, 0, 1, 1, 1, 0]
```

Expected counts are `both_correct = 2`, `rescue = 1`, `harmful_switch = 1`, and `both_wrong = 2`. The routing target is `[0, 1, 0, 0, 0, 0]`; PM LER and CM LER are each `3/6`, rescue opportunity is `1/6`, decoder disagreement is `2/6`, and oracle LER is `2/6`. The tests should also confirm equal-length inputs and that the four masks cover every row without overlap. Run `make test` from the repository root.

**Deliver:** verified `route_to_correlated()` and tests for both correct, rescue, harmful switch, and both wrong. The tests must confirm the routing target and Task 2's metric formulas on a tiny manual dataset.

**Study:** the [experiment protocol](EXPERIMENT_PROTOCOL.md), basic test assertions, and data leakage.

## Task 4 — Put the analysis in Colab

**Where:** `notebooks/00_dataset_and_feasibility.ipynb` for the analysis and `results/feasibility/` for a compact findings note.

**What to do:** continue the existing notebook. Its executed cells fetch the reusable `router` modules at a resolved repository commit, install the required runtime libraries, run tests, and load a real train sample. The remaining cells must call the reusable functions; do not copy their calculations into notebook cells.

The notebook should run in this order:

1. **Introduction:** dataset source, question, and scope; explain that the two decoder predictions are stored in the dataset.
2. **Setup:** use a recorded repository commit and install the required dependencies. The current notebook downloads the needed modules from that commit and adds `src/` to the import path; a full clone and editable install are also valid.
3. **Run settings:** dataset name/revision if known, repository commit, split, sample size, and sampling seed.
4. **Data inspection:** split sizes, columns, compact sample rows, value sets, missing counts, and unresolved schema notes.
5. **Outcome results:** a four-count table and a five-rate table using `outcome_masks()` and `summarize_outcomes()`.
6. **Routing target:** call `route_to_correlated()` and show that only rescue rows receive `1`.
7. **Checks and interpretation:** assert that counts sum to `N`, the target matches the rescue mask, and rates lie between 0 and 1. Explain what the sample does and does not show.

For a column-indexable sample, the core calculation is:

```python
import numpy as np
from router.decoders import outcome_masks, summarize_outcomes
from router.labels import route_to_correlated

true_label = np.asarray(sample["label"])
pm = np.asarray(sample["pymatching"])
cm = np.asarray(sample["correlated_matching"])
masks = outcome_masks(true_label, pm, cm)
summary = summarize_outcomes(true_label, pm, cm)
y_router = route_to_correlated(true_label, pm, cm).astype(int)

assert sum(summary[name] for name in masks) == summary["shots"]
assert np.array_equal(y_router, masks["rescue"].astype(int))
```

Save a compact findings note in `results/feasibility/` with the run settings, schema findings, counts, rates, and limitations. Do not save or commit shot-level rows.

**Deliver:** a notebook that runs top to bottom in a fresh Colab runtime, shows the schema and decoder results, explains `y_router` and oracle LER, and records enough settings to reproduce the sample; plus the compact findings note.

**Study:** Colab setup, Python package imports, and the [project guide](PROJECT_GUIDE.md).

## Complete by the end of Day 7

- [x] The loader and sampler are implemented; five offline data tests pass.
- [x] The notebook loaded the real train split and selected 50,000 rows with recorded seed and dataset revision; a repeat matched the three stored label/prediction columns.
- [ ] The repository source commit is frozen for the final run, and any claimed full-row reproducibility is verified.
- [ ] Split sizes, columns, key values, missingness, and open data questions are recorded.
- [ ] The four outcome counts sum to `N`, and all five rates use the correct denominator.
- [ ] `y_router` is 1 only for PM wrong + CM correct; manual tests and `make test` pass.
- [ ] The Colab notebook runs top to bottom and matches the verified functions.
- [ ] A compact findings note records settings and results without shot-level data.
- [ ] No test rows informed development, and no true label or stored decoder prediction is used as a router input feature.

For background and constraints, see the [project guide](PROJECT_GUIDE.md), [data dictionary](DATA_DICTIONARY.md), and [experiment protocol](EXPERIMENT_PROTOCOL.md).
