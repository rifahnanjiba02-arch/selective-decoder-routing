# Experiment protocol

The objective is low system logical error rate (LER) with less unnecessary use of correlated matching. Select routing thresholds on validation data and report LER versus correlated-matching usage on the final test set. `PM wrong AND CM correct` is the positive rescue target; `PM wrong AND CM wrong` is the two-decoder oracle failure event.

## Non-negotiable rules

1. Never use the true logical `label` as a router input.
2. Never use `pymatching` or `correlated_matching` predictions as router input features.
3. Decoder predictions and the true label may be used only to create routing targets and evaluate the final system.
4. Train models on the training split only.
5. Use validation data for hyperparameters, model selection, and routing-threshold selection.
6. Keep the test split untouched until final evaluation. Do not use a test development sample to guide design.
7. Evaluate all routing methods on the same test samples.
8. Save random seeds, dataset revision, sampling choices, feature definitions, model settings, and the selected threshold for each run.
9. Start with development subsets of train and validation before scaling to millions of samples.
10. Do not upload the complete dataset, cached Hugging Face files, generated feature tables, or large model files to GitHub.

## Study sequence

1. Inspect the [dataset card](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes), schema, split sizes, missing values, and `round_widths` mask. Record any field interpretation that needs confirmation in [the data dictionary](DATA_DICTIONARY.md).
2. On a small training or validation subset, calculate the four decoder-outcome groups and check that their counts sum to the sample size. Report PyMatching LER, correlated-matching LER, rescue opportunity, and oracle LER with correct denominators.
3. Build features only from syndrome and permitted experiment metadata. Padded syndrome entries are structurally empty; use `round_widths` when computing activity and density.
4. Fit a Dummy baseline and candidate routers on train; compare on validation. Select thresholds with the system LER and correlated-matching usage tradeoff, including harmful switches and rescue capture.
5. Freeze all choices, then run a single final comparison on the same test shots: always PyMatching, always correlated matching, selective routing, and oracle routing. Save the exact settings and report uncertainty where appropriate.

## Open scientific choices before implementation

- Decide whether the main claim concerns new shots from the dataset's existing experiments (use the provided within-experiment splits) or generalization to unseen experiments (requires a separate group-held-out protocol). Do not silently mix these claims.
- Decide the acceptable correlated-matching usage budget or threshold selection rule before opening the test split.
- Define whether the main comparison pools shots or reports results by basis, distance, rounds, and experiment; account for different group sizes.
- Validate that stored decoder predictions and labels have the expected binary semantics before computing metrics.
