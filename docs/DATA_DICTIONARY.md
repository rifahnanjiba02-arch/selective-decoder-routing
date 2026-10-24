# Data dictionary

Source: [Bauxitiego/surface-code-syndromes dataset card](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes), which reformats [Google Quantum AI experimental data](https://zenodo.org/records/6804040). The field roles below come from the dataset card; measured types, missing counts, and sampled-row checks are recorded in the Task 1 findings.

| Field | Dataset-card description | Project use |
| --- | --- | --- |
| `syndrome` | Flattened binary `[time_steps, width]` sequence, right-padded with zeros. | Candidate router input; padded positions need a mask. |
| `label` | Ground-truth logical-observable flip. | Target/evaluation only; never a router input. |
| `pymatching` | Stored minimum-weight perfect-matching prediction. | Compare to `label` for targets/evaluation only. |
| `correlated_matching` | Stored correlated-matching prediction. | Compare to `label` for targets/evaluation only. |
| `distance` | Surface-code distance. | Grouping variable and possible input, subject to the leakage boundary. |
| `rounds` | Number of syndrome-extraction rounds. | Relates to `time_steps` and possible input grouping. |
| `basis` | Memory basis, `X` or `Z`. | Grouping variable and possible input. |
| `round_widths` | Count of real detector positions at each time step. | Defines a mask so padded zeros are not counted as observations. |
| `time_steps` | First reshaped-syndrome dimension; dataset card says `rounds + 1`. | Used to reshape and mask `syndrome`. |
| `width` | Maximum round width, second reshaped-syndrome dimension. | Used to reshape and mask `syndrome`. |
| `experiment` | Source experiment directory name. | Potential grouping metadata; use as an input needs separate review. |
| `tensor_network` | Stored tensor-network prediction, null where not run. | Outside current two-decoder routing scope. |

**Unresolved interpretation:** TODO: verify from dataset documentation and original release exactly what costs or runtimes distinguish the two decoders in this dataset. Until measured, "usage" is a selection fraction, not a measured time or energy saving. TODO: verify whether experiment identifier or other metadata would create an unfair shortcut for the intended generalization claim.

## Week 1 Task 1 findings (24 October 2026)

The [feasibility notebook](../notebooks/00_dataset_and_feasibility.ipynb) has saved Colab outputs for these checks on dataset revision `9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3`, with **50,000 train rows** and sampling seed `42`. Repeating the selection matched **all fields of all 50,000 sampled rows in the same order**, checked in batches. The repository's `configs/data.yaml` still has `dataset_revision: null`; the notebook used a temporary pinned config.

The loaded train split contained **4,550,000 rows**. The [dataset card](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes) lists **650,000 validation** and **1,300,000 test** rows. Only train rows were selected for analysis. The Hugging Face loader also downloaded/prepared validation and test files internally, as the notebook's saved output shows.

| Field | Loaded dataset type | Missing rows in sample |
| --- | --- | ---: |
| `experiment` | string | 0 |
| `basis` | string | 0 |
| `distance` | int8 | 0 |
| `rounds` | int16 | 0 |
| `time_steps` | int8 | 0 |
| `width` | int16 | 0 |
| `round_widths` | list of uint8 | 0 |
| `syndrome` | list of uint8 | 0 |
| `label` | uint8 | 0 |
| `pymatching` | uint8 | 0 |
| `correlated_matching` | uint8 | 0 |
| `tensor_network` | uint8, nullable | 50,000 |

Observed values in the sample were `label`, `pymatching`, and `correlated_matching`: `{0, 1}` each; `basis`: `{X, Z}`; `distance`: `{3, 5}`; and `rounds`: every odd integer from `1` through `25`. These six fields had **zero** missing rows and **zero** values outside the expected sets. `tensor_network` was null in every sampled row; it is outside this two-decoder analysis.

Across all 50,000 sampled rows, the following checks each had **zero failing rows**: positive integer dimensions, `time_steps == rounds + 1`, `len(syndrome) == time_steps * width`, `len(round_widths) == time_steps`, each real width within `0..width`, binary syndrome values, and zero values in padded syndrome positions. The run checked **514,200 padded positions**, with **zero nonzero values**. These findings describe the sampled training rows; they do not establish that every row in the full dataset is valid.
