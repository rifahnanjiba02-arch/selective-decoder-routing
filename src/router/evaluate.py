"""Validation metrics for rescue prediction and final decoder routing.

The 0.5 threshold used in notebook 02 is provisional. Threshold selection and
the untouched test split belong to later work.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .decoders import outcome_masks
from .routing import choose_correlated, routed_prediction


def rescue_metrics(y_true, rescue_probability, threshold=0.5):
    """Score the rescue class on one validation sample.

    Average precision uses the continuous scores. Precision, recall, and F1
    use the stated threshold; undefined values are reported as zero.
    """
    y = np.asarray(y_true)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be a nonempty binary 1D array")
    predicted = choose_correlated(rescue_probability, threshold)
    if predicted.shape != y.shape:
        raise ValueError("scores and y_true must have the same length")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y.astype(int), predicted.astype(int), average="binary", zero_division=0
    )
    return {
        "average_precision": float(average_precision_score(y, rescue_probability)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def system_metrics(label, pymatching, correlated_matching, use_correlated):
    """Evaluate selected stored predictions on the same ordered rows.

    A rescue is captured when correlated matching is selected on a rescue row.
    A harmful switch selects it where PyMatching alone was correct.
    """
    masks = outcome_masks(label, pymatching, correlated_matching)
    decision = np.asarray(use_correlated)
    n = len(masks["rescue"])
    if decision.shape != (n,) or not np.isin(decision, (False, True)).all():
        raise ValueError("use_correlated must be a binary 1D decision for every row")
    decision = decision.astype(bool)
    final_prediction = routed_prediction(decision, pymatching, correlated_matching)
    return {
        "logical_error_rate": float(np.mean(final_prediction != np.asarray(label))),
        "correlated_usage": float(np.mean(decision)),
        "rescues_captured": int(np.count_nonzero(decision & masks["rescue"])),
        "harmful_switches": int(np.count_nonzero(decision & masks["harmful_switch"])),
    }
