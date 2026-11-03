"""Decoder outcome calculations (Week 1, Task 2)."""
import numpy as np


def _as_binary(x, name):
    """Convert to a 1-D bool array; only 0/1 (or False/True) allowed."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return arr.astype(bool)


def outcome_masks(label, pymatching, correlated_matching):
    """Return four boolean masks. Each row is in exactly one mask."""
    y = _as_binary(label, "label")
    p = _as_binary(pymatching, "pymatching")
    c = _as_binary(correlated_matching, "correlated_matching")
    if not (len(y) == len(p) == len(c)):
        raise ValueError("label, pymatching, correlated_matching must have equal length")
    if y.size == 0:
        raise ValueError("at least one shot is required")

    pm_ok = p == y   # PM correct
    cm_ok = c == y   # CM correct
    return {
        "both_correct": pm_ok & cm_ok,
        "rescue": ~pm_ok & cm_ok,
        "harmful_switch": pm_ok & ~cm_ok,
        "both_wrong": ~pm_ok & ~cm_ok,
    }


def summarize_outcomes(label, pymatching, correlated_matching):
    """Return shots, four counts and five rates (denominator = N)."""
    masks = outcome_masks(label, pymatching, correlated_matching)
    n = len(masks["both_correct"])
    counts = {k: int(v.sum()) for k, v in masks.items()}
    assert sum(counts.values()) == n  # masks cover every row, no overlap

    p = np.asarray(pymatching).astype(bool)
    c = np.asarray(correlated_matching).astype(bool)
    return {
        "shots": n,
        **counts,
        "pymatching_ler": (counts["rescue"] + counts["both_wrong"]) / n,
        "correlated_matching_ler": (counts["harmful_switch"] + counts["both_wrong"]) / n,
        "rescue_opportunity": counts["rescue"] / n,
        "decoder_disagreement": float((p != c).sum()) / n,
        "oracle_ler": counts["both_wrong"] / n,
    }
