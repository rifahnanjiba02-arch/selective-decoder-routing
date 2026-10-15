# Selective Decoder Routing for Surface Code Data

This project studies when to use a stronger decoder for a surface code measurement. A router will choose between PyMatching (PM) and correlated matching (CM) for each shot, aiming to reduce the logical error rate while limiting how often CM is used.

## Data

The planned data source is [Bauxitiego/surface-code-syndromes](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes). It provides syndrome measurements, the true logical label, stored PM and CM predictions, and experiment metadata. The dataset is not included in this repository.

## Study plan

1. Measure PM and CM logical error rates and the **oracle bound**: the fraction of shots where PM is wrong and CM is right.
2. Build syndrome and metadata features. Compare logistic regression, random forest, and XGBoost with a variational quantum classifier on the same small feature set. Also evaluate classical models on a larger feature set.
3. Use experiment-grouped splits to avoid leakage. Evaluate on held-out shots using precision-recall AUC and curves of logical error rate versus CM usage.
4. Report always-PM, always-CM, random routing, and oracle routing as reference points, with results across multiple seeds and confidence intervals.

## Current status

This repository currently contains the planned folder and file structure only. The configuration, scripts, models, Makefile targets, and app are empty placeholders; no experiment has been run and there are no results to report yet.

The first implementation milestone is to inspect the dataset fields and calculate the oracle bound before training a router.
