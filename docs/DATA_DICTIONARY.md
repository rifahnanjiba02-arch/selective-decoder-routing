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
