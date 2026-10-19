"""Small, decoder-agnostic feasibility calculations on stored predictions.

These functions compare predictions to logical labels. They are for target
construction and evaluation, never feature extraction.
"""

import numpy as np


def outcome_masks(label, pymatching, correlated_matching):
    """Return boolean masks for the four mutually exclusive decoder outcomes."""
    y = np.asarray(label)
    pm = np.asarray(pymatching)
    cm = np.asarray(correlated_matching)
    if y.ndim != 1 or pm.shape != y.shape or cm.shape != y.shape:
        raise ValueError("label and decoder predictions must be equal-length 1D arrays")
    if y.size == 0:
        raise ValueError("at least one shot is required")

    pm_correct = pm == y
    cm_correct = cm == y
    return {
        "both_correct": pm_correct & cm_correct,
        "rescue": ~pm_correct & cm_correct,
        "harmful_switch": pm_correct & ~cm_correct,
        "both_wrong": ~pm_correct & ~cm_correct,
    }


def summarize_outcomes(label, pymatching, correlated_matching):
    """Return four counts, decoder LERs, rescue rate, disagreement, and oracle LER."""
    masks = outcome_masks(label, pymatching, correlated_matching)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    total = sum(counts.values())
    return {
        "shots": total,
        **counts,
        "pymatching_ler": (counts["rescue"] + counts["both_wrong"]) / total,
        "correlated_matching_ler": (counts["harmful_switch"] + counts["both_wrong"]) / total,
        "rescue_opportunity": counts["rescue"] / total,
        "decoder_disagreement": float(np.mean(np.asarray(pymatching) != np.asarray(correlated_matching))),
        "oracle_ler": counts["both_wrong"] / total,
    }
