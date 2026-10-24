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

The **code foundation for Task 1 is merged into `main`** (pull request #1). The loader and sampler should be used and verified rather than rewritten. The remaining Task 1 work is a real-data run and its recorded findings.

| Work | Status now | Continue with |
| --- | --- | --- |
| Load a named split | **Implemented:** `load_surface_code_dataset()` reads `configs/data.yaml`, requests `train` by default, returns the selected split and config, and requires explicit opt-in for `test`. | Run it with the real Hugging Face dataset and confirm the returned split and schema. |
| Reproducible sample | **Implemented:** `sample_split()` selects distinct rows with an explicit size and seed, and rejects invalid or oversized requests. Five offline data tests pass. | Reproduce the 50,000-row training sample in the project environment. |
| Sample settings | **Set:** training size 50,000, validation size 10,000, sampling seed 42; test sample size is unset. The source split seed is recorded separately. | Pin the dataset revision before reporting reproducible results. |
| Dataset inspection | **Pending:** no actual split-size, column-value, missingness, or shape report has been saved yet. | Inspect real training rows and update `docs/DATA_DICTIONARY.md`. |
| Decoder outcomes and routing target | **Starter functions and a manual test already exist** in `src/router/decoders.py`, `src/router/labels.py`, and `tests/test_routing.py`; this branch has not produced sample metrics. | Verify field encoding, run the functions on the sampled rows, and check the counts and rates. |
| Colab presentation | **Pending:** no Week 1 notebook or findings note exists. | Build `notebooks/00_dataset_and_feasibility.ipynb` after the data and metric checks. |

The offline tests check function behavior without downloading data. They do **not** confirm the dataset's actual field values or complete Task 1's inspection deliverable.

## Leakage boundary for this task

- The dataset's published train, validation, and test splits are [stratified within each experiment](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes). Use them as provided. Results from these splits concern new shots from the same experiments; a claim about unseen experiments needs a separate group-held-out study.
- `label`, `pymatching`, and `correlated_matching` are present in loaded rows because they are needed for the routing target and outcome evaluation. They must never become router input features. Feature extraction is not implemented in Week 1, so this boundary must be checked again when feature code is written.
- Training samples come from `train`. The loader and sampler require an explicit `allow_test=True` for test access; this is reserved for final evaluation after all choices are frozen.

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

**Already done in code:**

- `load_surface_code_dataset()` reads the config, requests one named split from Hugging Face, defaults to `train`, and returns a split mapping plus the config. It has not yet been run against the real dataset.
- `sample_split()` selects distinct rows using an explicit size and seed. It has passed offline reproducibility and input-check tests. `configs/data.yaml` sets the Week 1 training sample to 50,000 rows and the sampling seed to 42.
- Both functions require `allow_test=True` before accessing `test`; that option is reserved for final evaluation. The loader's `split` request controls its returned data, while actual download behavior depends on the dataset builder.

**Still to do with real data:**

- Run the existing loader for `train`, then call `sample_split(splits, sample_size=config["development"]["train_size"], seed=config["sampling_seed"])`. Confirm that repeated calls return the same 50,000 rows in the same order.
- Confirm the published train, validation, and test split names and sizes from dataset metadata without opening test rows. Pin the dataset revision used for the run.
- Inspect column names, a few compact rows, missing values, and observed values of `label`, `pymatching`, `correlated_matching`, `basis`, `distance`, and `rounds`.
- Check `syndrome`, `round_widths`, `time_steps`, and `width` for shape consistency without printing many full syndrome arrays. Record confirmed interpretations and unresolved questions in [the data dictionary](DATA_DICTIONARY.md).

The full dataset, Hugging Face cache, and shot-level output files must stay out of Git.

**Deliver:** the functions and configured sample settings are already present. Task 1 is complete only after the real dataset run records split sizes, columns, relevant values, missing or unusual values, the pinned revision, and evidence that the sample is reproducible.

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

**What to do:** create a notebook that presents and calls the reusable functions in `src/router/`. The notebook does not duplicate their calculations.

The notebook should run in this order:

1. **Introduction:** dataset source, question, and scope; explain that the two decoder predictions are stored in the dataset.
2. **Setup:** clone the repository in a fresh Colab runtime and run `%pip install -e .` from the repository root.
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
- [ ] A real 50,000-row training sample reproduces in the same order with the recorded seed and dataset revision.
- [ ] Split sizes, columns, key values, missingness, and open data questions are recorded.
- [ ] The four outcome counts sum to `N`, and all five rates use the correct denominator.
- [ ] `y_router` is 1 only for PM wrong + CM correct; manual tests and `make test` pass.
- [ ] The Colab notebook runs top to bottom and matches the verified functions.
- [ ] A compact findings note records settings and results without shot-level data.
- [ ] No test rows informed development, and no true label or stored decoder prediction is used as a router input feature.

For background and constraints, see the [project guide](PROJECT_GUIDE.md), [data dictionary](DATA_DICTIONARY.md), and [experiment protocol](EXPERIMENT_PROTOCOL.md).
