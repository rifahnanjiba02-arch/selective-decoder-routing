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


def outcome_masks(true_label, pm, cm):
    """Return four boolean masks. Each row is in exactly one mask."""
    y = _as_binary(true_label, "label")
    p = _as_binary(pm, "pymatching")
    c = _as_binary(cm, "correlated_matching")
    if not (len(y) == len(p) == len(c)):
        raise ValueError("label, pymatching, correlated_matching must have equal length")

    pm_ok = p == y   # PM correct
    cm_ok = c == y   # CM correct
    return {
        "both_correct": pm_ok & cm_ok,
        "rescue": ~pm_ok & cm_ok,
        "harmful_switch": pm_ok & ~cm_ok,
        "both_wrong": ~pm_ok & ~cm_ok,
    }


def summarize_outcomes(true_label, pm, cm):
    """Return shots, four counts and five rates (denominator = N)."""
    masks = outcome_masks(true_label, pm, cm)
    n = len(masks["both_correct"])
    if n == 0:
        raise ValueError("empty input")
    counts = {k: int(v.sum()) for k, v in masks.items()}
    assert sum(counts.values()) == n  # masks cover every row, no overlap

    p = np.asarray(pm).astype(bool)
    c = np.asarray(cm).astype(bool)
    return {
        "shots": n,
        **counts,
        "pymatching_ler": (counts["rescue"] + counts["both_wrong"]) / n,
        "correlated_matching_ler": (counts["harmful_switch"] + counts["both_wrong"]) / n,
        "rescue_opportunity": counts["rescue"] / n,
        "decoder_disagreement": float((p != c).sum()) / n,
        "oracle_ler": counts["both_wrong"] / n,
    }
