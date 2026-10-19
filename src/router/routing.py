"""Threshold decisions and final prediction selection for candidate routers."""

import numpy as np


def choose_correlated(rescue_probability, threshold):
    """Choose correlated matching where predicted rescue probability meets threshold."""
    probability = np.asarray(rescue_probability)
    if probability.ndim != 1 or not np.all(np.isfinite(probability)):
        raise ValueError("probabilities must be a finite 1D array")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie between 0 and 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie between 0 and 1")
    return probability >= threshold


def routed_prediction(use_correlated, pymatching, correlated_matching):
    """Select one stored decoder prediction for each shot."""
    decision = np.asarray(use_correlated)
    pm = np.asarray(pymatching)
    cm = np.asarray(correlated_matching)
    if decision.ndim != 1 or decision.shape != pm.shape or pm.shape != cm.shape:
        raise ValueError("decision and decoder predictions must be equal-length 1D arrays")
    return np.where(decision, cm, pm)
