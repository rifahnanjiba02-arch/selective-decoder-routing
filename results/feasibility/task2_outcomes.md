# Task 2 - decoder outcomes (50,000-row train sample)

Settings: dataset Bauxitiego/surface-code-syndromes, revision
9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3, split train, N = 50,000, seed 42.
The run used a temporary copy of `configs/data.yaml` with that revision pinned.
It loaded the train split with `load_surface_code_dataset()` and selected rows
with the repository's `sample_split()`. Counts and rates came from
`outcome_masks()` and `summarize_outcomes()`; the four counts sum to N.
The notebook's saved Colab decoder tables and checks reproduce these results.

| Outcome | Count | Share of N |
| --- | ---: | ---: |
| both_correct | 32055 | 0.64110 |
| rescue | 4625 | 0.09250 |
| harmful_switch | 3728 | 0.07456 |
| both_wrong | 9592 | 0.19184 |

| Rate | Value |
| --- | ---: |
| pymatching_ler | 0.28434 |
| correlated_matching_ler | 0.26640 |
| rescue_opportunity | 0.09250 |
| decoder_disagreement | 0.16706 |
| oracle_ler | 0.19184 |

Limitations: results cover the sampled train rows only. Oracle LER is the lower
bound on LER achievable by selecting between these stored predictions; it is
not a measured trained-router result. Schema findings are in the
[data dictionary](../../docs/DATA_DICTIONARY.md).
