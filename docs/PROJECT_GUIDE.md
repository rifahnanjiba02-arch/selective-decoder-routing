# Project guide

A **syndrome** is a pattern of detection events from repeated surface-code checks. Each row in the [dataset](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes) represents one experimental shot. The syndrome is the evidence a decoder uses to infer whether the logical observable flipped. The stored sequence is padded to a common width; `round_widths` identifies which positions are real measurements.

**PyMatching** is a minimum-weight perfect-matching decoder. In this project, its stored per-shot prediction is the fast/default choice. **Correlated matching** is another decoder whose stored prediction can differ from PyMatching's. We study whether selecting it on some shots improves the final logical prediction while limiting its use. We do not rerun either decoder in the initial feasibility work.

The **router** learns from permitted syndrome and experiment features to decide which stored decoder prediction to use. It does **not** decode the syndrome itself, change the logical label, or receive the true label or either decoder prediction as an input feature. Logistic Regression, Random Forest, XGBoost, and VQC are all candidates for this one decision.

When PyMatching is wrong and correlated matching is correct, choosing correlated matching **rescues** that shot. That is the positive routing target. Switching when PyMatching is correct and correlated matching is wrong harms the shot, so routing quality must be judged by the final system LER as well as by target-class metrics.

An **oracle router** knows which of the two predictions is correct for every shot and chooses correctly whenever either decoder can. Its LER is therefore the fraction where **both decoders are wrong**. It is a theoretical reference, not a deployable model. The rescue-opportunity fraction is a different quantity.

Using correlated matching only where it helps is **decoder-use efficiency**: fewer calls to the stronger decoder for a given logical-error outcome. Our planned result is an LER-versus-correlated-matching-usage curve, not a claim that one model family always wins.
