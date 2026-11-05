# Week 1 task guide: dataset and decoder feasibility

The assignment called “Day 1” in `cse491_project.md` is planned as a **one-week task**. By the end of Day 7, the repository should contain a reproducible training sample workflow, verified decoder outcome calculations, a binary routing target, and a Colab notebook explaining the findings. See the [seven-day schedule](WEEK_1_SCHEDULE.md) for checkpoints.

## What this task answers

Each dataset row is one surface-code shot with a true logical `label` and two stored decoder predictions: `pymatching` (PM) and `correlated_matching` (CM). This task checks:

1. Can the dataset and a small training sample be loaded reproducibly?
2. How often is each decoder correct or wrong on that sample?
3. How often could selecting CM rescue PM, and how often would switching hurt?
4. What is the lowest logical error rate possible when selecting between these two stored predictions?

This is a **feasibility analysis**. It does not train a routing model or measure decoder runtime savings.

## Completion snapshot — 5 November 2026

All four Week 1 tasks are complete. Tasks 1–3 are merged into `main`; the [executed notebook](../notebooks/00_dataset_and_feasibility.ipynb) and [combined findings note](../results/feasibility/week1_findings.md) complete Task 4. The notebook's 16 code cells ran in order with no error outputs, and its decoder counts and rates match the separately verified [Task 2 findings](../results/feasibility/task2_outcomes.md).

| Work | Verified result |
| --- | --- |
| Load a named split | `load_surface_code_dataset()` loaded `train`: 4,550,000 rows and 12 columns. |
| Reproducible sample | `sample_split()` selected 50,000 train rows with seed 42; a second selection matched every field of every row in order. |
| Sample settings | Dataset revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3`, train sample size 50,000, and seed 42 are saved in the notebook and combined note. The notebook used a temporary pinned config; `configs/data.yaml` still leaves `dataset_revision: null`. |
| Dataset inspection | The [data dictionary](DATA_DICTIONARY.md) records all 12 field types and missing counts, relevant value sets, and syndrome/`round_widths` shape and padding checks. All required shape checks had zero failing sampled rows. |
| Decoder outcomes | `outcome_masks()` and `summarize_outcomes()` produced four counts summing to 50,000 and five rates. The saved notebook agrees with the [Task 2 findings](../results/feasibility/task2_outcomes.md). |
| Routing target | `route_to_correlated()` returns the rescue mask. Hand-checked tests cover all four outcomes, both logical-label values, disagreement, mask coverage, and unequal input lengths. All 15 repository tests pass; the notebook shows 4,625 positive targets, all rescue rows. |
| Colab presentation | The notebook includes saved data checks, outcome and rate tables, the routing-target table, final assertions, and interpretation. The [combined findings note](../results/feasibility/week1_findings.md) records settings, schema, results, and limitations. |

Hugging Face downloaded/prepared validation and test files internally during the Colab run; only train rows were selected for analysis. The combined note contains no shot-level rows.

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

Task 1 now supplies a checked training sample. Tasks 2 and 3 use it for decoder outcomes and the routing target; Task 4 presents the combined result.

## Task 1 — Load and inspect the dataset

**Where:** `src/router/data.py` for loading/sampling, `configs/data.yaml` for settings, and `docs/DATA_DICTIONARY.md` for findings.

**Done in code and the pinned training-sample run:**

- `load_surface_code_dataset()` reads the config, requests one named split from Hugging Face, defaults to `train`, and returns a split mapping plus the config. The notebook has loaded the real train split and confirmed its 4,550,000-row count.
- `sample_split()` selects distinct rows using an explicit size and seed. The 50,000-row train sample with seed 42 matched a repeated selection across every field and row in order. Offline reproducibility and input-check tests passed.
- Both functions require `allow_test=True` before accessing `test`; that option is reserved for final evaluation. The loader's `split` request controls its returned data, while actual download behavior depends on the dataset builder.
- The sample had the expected binary `label` and decoder predictions; observed `basis`, `distance`, and `rounds` values were within the published domains. All 11 fields used here had zero missing rows; `tensor_network` was null in all 50,000 sampled rows and is outside scope.
- All sampled rows passed the dimensions, sequence-length, binary-syndrome, round-width, and zero-padding checks. See [the measured counts and types](DATA_DICTIONARY.md#week-1-task-1-findings-24-october-2026).

The saved Colab outputs use the pinned dataset revision. Validation/test counts remain dataset-card metadata; no validation/test rows were selected. Task 1 findings cover the 50,000 sampled train rows, not every row in the full dataset.

The full dataset, Hugging Face cache, and shot-level output files must stay out of Git.

**Deliver:** Task 1's loader, sampler, pinned sample settings, full-row repeat check, schema and missingness results, shape/padding checks, and data-dictionary findings are complete for the 50,000-row training sample.

**Study:** Hugging Face `load_dataset`, train/validation/test splits, and the dataset card referenced in the [data dictionary](DATA_DICTIONARY.md).

## Task 2 — Calculate decoder outcomes

**Where:** `src/router/decoders.py`. The functions `outcome_masks()` and `summarize_outcomes()` are implemented and verified.

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

**Verified result:** On the pinned 50,000-row train sample, the counts were 32,055 both correct, 4,625 rescue, 3,728 harmful switch, and 9,592 both wrong. The five rates and reproducibility settings are in [the Task 2 findings](../results/feasibility/task2_outcomes.md). All four counts sum to `N`; Task 2 is complete.

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

The verified function returns `True` for rescue and `False` for every other row; these correspond to `1` and `0`. Decoder disagreement alone is **not** the positive target.

Use hand-checked examples in the tests. One useful six-row case is already in `tests/test_routing.py`:

```text
label = [0, 0, 0, 0, 0, 0]
PM    = [0, 1, 0, 1, 1, 0]
CM    = [0, 0, 1, 1, 1, 0]
```

Expected counts are `both_correct = 2`, `rescue = 1`, `harmful_switch = 1`, and `both_wrong = 2`. The routing target is `[0, 1, 0, 0, 0, 0]`; PM LER and CM LER are each `3/6`, rescue opportunity is `1/6`, decoder disagreement is `2/6`, and oracle LER is `2/6`. The tests also confirm that unequal-length inputs are rejected and that the four masks cover every row without overlap. Run `make test` from the repository root where `make` is available.

**Deliver:** verified `route_to_correlated()` and tests for both correct, rescue, harmful switch, and both wrong. The tests must confirm the routing target and Task 2's metric formulas on a tiny manual dataset.

**Verified result:** The existing function returns `True` only for rescue, corresponding to `y_router = 1`. Commit `1054fa1` added hand-checked tests without changing the function, and pull request #3 merged Task 3 into `main`. On 5 November 2026, the Makefile's equivalent commands in PowerShell (`$env:PYTHONPATH='src'; python -m unittest discover -s tests`) passed all 15 tests; `make` was unavailable in that environment.

**Study:** the [experiment protocol](EXPERIMENT_PROTOCOL.md), basic test assertions, and data leakage.

## Task 4 — Put the analysis in Colab

**Where:** `notebooks/00_dataset_and_feasibility.ipynb` for the analysis and `results/feasibility/` for a compact findings note.

**Implementation:** The executed notebook fetches the reusable `router` modules, installs the required runtime libraries, runs tests, and loads a real train sample. The result cells call the reusable functions without copying their calculations into notebook cells.

The notebook should run in this order:

1. **Introduction:** dataset source, question, and scope; explain that the two decoder predictions are stored in the dataset.
2. **Setup:** install the required dependencies. The current notebook downloads the needed modules and adds `src/` to the import path; a full clone and editable install are also valid.
3. **Run settings:** dataset name/revision, split, sample size, and sampling seed.
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

**Verified result:** The saved Colab run executed all 16 code cells in order without error outputs. It reproduced the pinned sample, matched the Task 2 counts and rates, showed `y_router = 1` only for the 4,625 rescue rows, and passed the final assertions. The [combined findings note](../results/feasibility/week1_findings.md) records the run settings, schema findings, outcome counts, rates, and limitations without shot-level rows.

**Study:** Colab setup, Python package imports, and the [project guide](PROJECT_GUIDE.md).

## Complete by the end of Day 7

- [x] The loader and sampler are implemented; five offline data tests pass.
- [x] The pinned train split yielded 50,000 sampled rows; a repeat matched all fields of every row in order.
- [x] Split metadata, columns, key values, missingness, and shape/padding findings are recorded for the sampled train rows.
- [x] The four outcome counts sum to `N`, and all five rates use the correct denominator.
- [x] `y_router` is 1 only for PM wrong + CM correct; manual tests and the repository unittest suite pass.
- [x] The Colab notebook runs top to bottom and matches the verified functions.
- [x] The final combined findings note records settings, schema, and results without shot-level data; a Task 2-only note is saved.
- [x] No test rows informed development, and no true label or stored decoder prediction is used as a router input feature.

For background and constraints, see the [project guide](PROJECT_GUIDE.md), [data dictionary](DATA_DICTIONARY.md), and [experiment protocol](EXPERIMENT_PROTOCOL.md).
