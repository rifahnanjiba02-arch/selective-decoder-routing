import unittest

import numpy as np

from router.decoders import outcome_masks, summarize_outcomes
from router.labels import route_to_correlated
from router.routing import choose_correlated, routed_prediction


class RoutingTests(unittest.TestCase):
    def test_four_outcomes_and_oracle_ler_are_distinct_from_rescue_rate(self):
        label = np.zeros(6, dtype=int)
        pm = np.array([0, 1, 0, 1, 1, 0])
        cm = np.array([0, 0, 1, 1, 1, 0])

        masks = outcome_masks(label, pm, cm)
        self.assertEqual({name: mask.tolist() for name, mask in masks.items()}, {
            "both_correct": [True, False, False, False, False, True],
            "rescue": [False, True, False, False, False, False],
            "harmful_switch": [False, False, True, False, False, False],
            "both_wrong": [False, False, False, True, True, False],
        })
        self.assertEqual(route_to_correlated(label, pm, cm).tolist(), masks["rescue"].tolist())

        summary = summarize_outcomes(label, pm, cm)
        self.assertEqual(summary["shots"], 6)
        self.assertEqual(summary["pymatching_ler"], 0.5)
        self.assertEqual(summary["correlated_matching_ler"], 0.5)
        self.assertEqual(summary["rescue_opportunity"], 1 / 6)
        self.assertEqual(summary["oracle_ler"], 2 / 6)
        self.assertEqual(summary["decoder_disagreement"], 2 / 6)

    def test_threshold_and_selected_predictions(self):
        use_cm = choose_correlated([0.2, 0.8, 0.5], threshold=0.5)
        self.assertEqual(use_cm.tolist(), [False, True, True])
        self.assertEqual(routed_prediction(use_cm, [0, 1, 0], [1, 0, 1]).tolist(), [0, 0, 1])

    def test_reject_invalid_probability_and_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            choose_correlated([0.2, 1.2], threshold=0.5)
        with self.assertRaises(ValueError):
            outcome_masks([0, 1], [0], [0, 1])
