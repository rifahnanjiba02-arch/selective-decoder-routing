# Data dictionary

Source: [Bauxitiego/surface-code-syndromes dataset card](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes), which reformats [Google Quantum AI experimental data](https://zenodo.org/records/6804040). Confirm types, nulls, and sample rows during Day 1 before writing transforms.

| Field | Dataset-card description | Day 1 check |
| --- | --- | --- |
| `syndrome` | Flattened binary `[time_steps, width]` sequence, right-padded with zeros. | Verify length and value domain per sampled row. |
| `label` | Ground-truth logical-observable flip. | Confirm binary encoding and missingness. Never use as router input. |
| `pymatching` | Stored minimum-weight perfect-matching prediction. | Confirm binary encoding and compare to `label` only for targets/evaluation. |
| `correlated_matching` | Stored correlated-matching prediction. | Confirm binary encoding and compare to `label` only for targets/evaluation. |
| `distance` | Surface-code distance. | Check observed values and groups. |
| `rounds` | Number of syndrome-extraction rounds. | Check relation to `time_steps`. |
| `basis` | Memory basis, `X` or `Z`. | Check observed values. |
| `round_widths` | Count of real detector positions at each time step. | Build a mask so padded zeros are not counted as observations. |
| `time_steps` | First reshaped-syndrome dimension; dataset card says `rounds + 1`. | Verify against `round_widths` length. |
| `width` | Maximum round width, second reshaped-syndrome dimension. | Verify `len(syndrome) == time_steps * width`. |
| `experiment` | Source experiment directory name. | Check grouping and whether it should be used as an input feature. |
| `tensor_network` | Stored tensor-network prediction, null where not run. | Outside current two-decoder routing scope. |

**Unresolved interpretation:** TODO: verify from dataset documentation and original release exactly what costs or runtimes distinguish the two decoders in this dataset. Until measured, "usage" is a selection fraction, not a measured time or energy saving. TODO: verify whether experiment identifier or other metadata would create an unfair shortcut for the intended generalization claim.

## Initial Week 1 Colab observation (24 October 2026)

The saved output of [the feasibility notebook](../notebooks/00_dataset_and_feasibility.ipynb) shows a successful train-split load and a 50,000-row sample with seed `42`. It records dataset revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3` in a temporary Colab config. The repository's `configs/data.yaml` still has `dataset_revision: null`.

The loaded train split had **4,550,000 rows** and these 12 columns: `experiment`, `basis`, `distance`, `rounds`, `time_steps`, `width`, `round_widths`, `syndrome`, `label`, `pymatching`, `correlated_matching`, and `tensor_network`. The [dataset card](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes) lists **650,000 validation** and **1,300,000 test** rows; the notebook did not select rows from either split for analysis. Hugging Face did download/prepare their files during the train load, as its saved output shows.

The notebook displays five compact sample rows and repeats the sample with matching `label`, `pymatching`, and `correlated_matching` columns. Full observed value sets, missing counts, and syndrome/`round_widths` shape checks are **not recorded yet**. The preview is not evidence that every row has valid binary labels or decoder predictions.
