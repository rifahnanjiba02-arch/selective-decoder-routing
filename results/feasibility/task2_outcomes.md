# Task 2 - decoder outcomes (50,000-row train sample)

Settings: dataset Bauxitiego/surface-code-syndromes, revision
9c51b27ffc51ed7ff3ef3a72167ebf4862d1f2f3, split train, N = 50,000, seed 42.
Note: sampled in Colab with numpy rng.choice; re-check with repo sample_split().

| Outcome | Count | Share of N |
| --- | ---: | ---: |
| both_correct | 31807 | 0.63614 |
| rescue | 4790 | 0.09580 |
| harmful_switch | 3897 | 0.07794 |
| both_wrong | 9506 | 0.19012 |

| Rate | Value |
| --- | ---: |
| pymatching_ler | 0.28592 |
| correlated_matching_ler | 0.26806 |
| rescue_opportunity | 0.09580 |
| decoder_disagreement | 0.17374 |
| oracle_ler | 0.19012 |

Limitations: results cover the sampled train rows only; oracle LER is an upper
bound on what a perfect selector could reach, not what a trained router will get.
