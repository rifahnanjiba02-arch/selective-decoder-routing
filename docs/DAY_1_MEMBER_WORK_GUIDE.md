# Day 1 member work guide

This guide turns the Day 1 assignment in `cse491_project.md` into concrete work and handoffs. **Member 4 owns the Colab notebook that brings the other three members' work together.** Day 1 is a feasibility check, not model training.

## Goal and shared rules

For a small, reproducible sample of the **training** split, inspect the dataset and answer: how often does each stored decoder fail, when could correlated matching rescue PyMatching, and what is the best possible error rate if we could always pick the correct decoder?

- Work from the published train/validation/test splits. Sample about **50,000–100,000 training rows** for Day 1; record the sample size and seed. Read split metadata to confirm sizes, but keep test rows untouched.
- Use `label`, `pymatching`, and `correlated_matching` only for outcome labels and evaluation. Never use them as router input features.
- Confirm that the three fields have the expected binary values and check missing values before calculating rates.
- Put reusable calculations in `src/router/`; use the notebook to run and explain them. Do not commit the full dataset, Hugging Face cache, or large generated files.
- Record what was actually checked. If a field, split size, or decoder meaning is uncertain, mark it as unverified instead of guessing.

Start with [the project guide](PROJECT_GUIDE.md), [data dictionary](DATA_DICTIONARY.md), and [experiment protocol](EXPERIMENT_PROTOCOL.md). The repository already has starter implementations in `src/router/decoders.py` and `src/router/labels.py`, plus a small test in `tests/test_routing.py`; inspect and verify those before changing them.

## Work order and handoffs

| Member | Owns | Gives the next member |
| --- | --- | --- |
| **1 — Dataset** | `src/router/data.py`, `configs/data.yaml` | Working loader and deterministic sampler; actual split sizes, columns, value checks, and any data problems. |
| **2 — Decoder outcomes** | `src/router/decoders.py` | Four per-shot masks and a summary of counts and rates from the sampled rows. |
| **3 — Target and checks** | `src/router/labels.py`, tests | Verified rescue target and hand-checked tests for every outcome and metric. |
| **4 — Colab integration (you)** | `notebooks/00_dataset_and_feasibility.ipynb` | A runnable, clearly explained Day 1 notebook and a short findings record in `results/feasibility/`. |

Members 2 and 3 can work with small manual arrays while Member 1 sets up the loader. Member 4 can build the notebook structure immediately, then run its data and metric cells after the interfaces are ready.

## Member 1 — Load and inspect the data

1. Check the dataset card and use Hugging Face `load_dataset` to identify the published split names and sizes. Avoid loading all rows into a Pandas DataFrame just to inspect them.
2. Implement `load_surface_code_dataset()` so the dataset can be loaded by split. Implement `sample_split()` so the same split, sample size, and seed produce the same rows in the same order. Agree on and document the return types and arguments so Member 4 can call them directly.
3. Set a Day 1 train sample size and sampling seed in `configs/data.yaml`. Keep cache files in the configured ignored cache directory. Do not select a test sample for development.
4. On the training sample, report column names, sample row shapes/types, missing values, and observed values for `label`, `pymatching`, `correlated_matching`, `basis`, `distance`, and `rounds`. Also inspect `syndrome`, `round_widths`, `time_steps`, and `width` enough to catch inconsistent shapes. Avoid printing full syndrome arrays for many rows.
5. Add confirmed field interpretations or unresolved questions to `docs/DATA_DICTIONARY.md`, and send Member 4 the exact loading and sampling calls plus the sample seed and size.

**Done when:** Member 4 can run the loader and sampler in a fresh Colab session and reproduce the same training rows, and the schema/split summary is written down.

## Member 2 — Count decoder outcomes

1. Check the existing `outcome_masks(label, pymatching, correlated_matching)` and `summarize_outcomes(...)` in `src/router/decoders.py`. Verify equal-length, one-dimensional input and the dataset's actual value encoding before using the results.
2. For each shot, compare both stored predictions with the true `label` and assign exactly one group:

   | Group | Condition | Meaning |
   | --- | --- | --- |
   | `both_correct` | PM correct, CM correct | PyMatching was sufficient. |
   | `rescue` | PM wrong, CM correct | Switching helps. |
   | `harmful_switch` | PM correct, CM wrong | Switching hurts. |
   | `both_wrong` | PM wrong, CM wrong | Neither stored prediction helps. |

3. Report the four **counts** and confirm that they sum to the sample size. With `N` shots, calculate:

   ```text
   pymatching_ler         = (rescue + both_wrong) / N
   correlated_matching_ler = (harmful_switch + both_wrong) / N
   rescue_opportunity     = rescue / N
   decoder_disagreement   = count(PM prediction != CM prediction) / N
   oracle_ler             = both_wrong / N
   ```

4. Give Member 3 the function behavior and a sample result to check; give Member 4 the summary output. Name the denominator and separate counts from rates in any table.

**Done when:** all nine requested outputs are available, the groups are exhaustive and exclusive, and `rescue_opportunity` is clearly different from `oracle_ler`.

## Member 3 — Verify the routing target and metrics

1. Check the existing `route_to_correlated(...)` in `src/router/labels.py`. It must return true/1 **only** when PM is wrong **and** CM is correct. Prediction disagreement alone is not the target.
2. Create or extend a tiny manual test set containing all four outcome groups. State the expected group for each row before running code.
3. Assert the target is positive only on rescue rows. Assert the four masks do not overlap, cover every row, and produce the expected counts and all summary rates. Include a case where switching harms and a case where both decoders are wrong.
4. Run `make test` from the repository root and share the result with Members 2 and 4. If the dataset uses unexpected values, coordinate validation changes with Members 1 and 2 before treating metrics as trustworthy.

**Done when:** the manual expected values match the functions, including the distinction between rescue opportunity and oracle LER.

## Member 4 — Build the Day 1 Colab notebook

Create `notebooks/00_dataset_and_feasibility.ipynb`. Your job is to make the work reproducible and understandable; import the other members' functions instead of copying their logic into notebook cells.

1. **Add the notebook introduction.** State the question, dataset source, Day 1 scope, and that stored decoder predictions are being compared, not recomputed. Link the repository guides.
2. **Add Colab setup.** In a fresh Colab runtime, clone the repository, enter its directory, and install the package. For example:

   ```python
   !git clone https://github.com/rifahnanjiba02-arch/selective-decoder-routing.git
   %cd selective-decoder-routing
   %pip install -e .
   ```

   If you opened the notebook from an already cloned checkout, skip the clone cell and run the install from the repository root. Re-run `%pip install -e .` after pulling teammates' changes if the runtime needs to pick them up.
3. **Record run settings.** Display the dataset name/revision if known, split, sample size, sampling seed, and package commit or version. Import `load_surface_code_dataset` and `sample_split` using the agreed Member 1 interface. Use the **train** sample only for Day 1 calculations.
4. **Inspect the data.** Show the published split names and sizes, column list, a few compact example rows, relevant value sets and missing counts. Summarize long `syndrome` arrays by length rather than dumping them. Note any unresolved schema interpretation.
5. **Compute and explain outcomes.** Convert the sampled `label`, `pymatching`, and `correlated_matching` columns to aligned arrays. Call `outcome_masks()`, `summarize_outcomes()`, and `route_to_correlated()`. Show a table with the four outcome counts and a second table with the five rates. Explain in plain words why rescue is the positive target and why oracle LER is `both_wrong / N`.

   Once Member 1's sampler returns a column-indexable sample, the core calculation can look like this:

   ```python
   import numpy as np
   from router.decoders import outcome_masks, summarize_outcomes
   from router.labels import route_to_correlated

   y = np.asarray(sample["label"])
   pm = np.asarray(sample["pymatching"])
   cm = np.asarray(sample["correlated_matching"])
   masks = outcome_masks(y, pm, cm)
   summary = summarize_outcomes(y, pm, cm)
   target = route_to_correlated(y, pm, cm)

   assert sum(summary[name] for name in masks) == summary["shots"]
   assert np.array_equal(target, masks["rescue"])
   summary
   ```

   Adapt the column extraction if Member 1 chooses a different sampler return type.
6. **Add notebook checks.** Assert that group counts sum to `N`, masks have length `N`, and the target equals the rescue mask. Check that all reported rates are between 0 and 1. These checks catch wiring mistakes; Member 3's manual tests establish the expected math.
7. **Write a short interpretation.** Say whether rescue opportunities exist in this sample, how often a switch would hurt, and how far the oracle is from always-PyMatching. Treat the sample as an initial feasibility result, not a claim about a trained router or decoder runtime savings.
8. **Save a compact findings note** under `results/feasibility/` with the run settings, split/schema findings, counts/rates, and limitations. Do not save or commit shot-level rows.

**Done when:** the notebook runs from top to bottom in a fresh Colab runtime after the three code handoffs, its numbers match the verified functions, and another team member can repeat the sample using the recorded settings.

## Final Day 1 check

- [ ] Member 1: loader and reproducible train sampler work; schema, split sizes, and missing/odd values are recorded.
- [ ] Member 2: outcome groups sum to `N`; all requested counts and rates are reported.
- [ ] Member 3: rescue target and metrics pass hand-checked tests; `make test` passes.
- [ ] Member 4: notebook runs top to bottom, explains results, records settings, and commits the notebook plus a compact findings note without shot-level data.
- [ ] Everyone: no test rows informed Day 1 decisions, and no label or decoder prediction is used as a router input feature.
