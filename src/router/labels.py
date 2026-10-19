"""Routing labels derived from decoder outcomes, not from router features."""

from .decoders import outcome_masks


def route_to_correlated(label, pymatching, correlated_matching):
    """Positive only when PyMatching is wrong and correlated matching correct."""
    return outcome_masks(label, pymatching, correlated_matching)["rescue"]
