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

    def test_decoder_api_rejects_empty_shots_and_accepts_keyword_inputs(self):
        with self.assertRaises(ValueError):
            outcome_masks([], [], [])
        with self.assertRaises(ValueError):
            summarize_outcomes([], [], [])

        inputs = {"label": [0, 1], "pymatching": [0, 0], "correlated_matching": [0, 1]}
        self.assertEqual(outcome_masks(**inputs)["rescue"].tolist(), [False, True])
        self.assertEqual(summarize_outcomes(**inputs)["rescue"], 1)

    def test_route_target_values_for_six_row_example(self):
        label = np.zeros(6, dtype=int)
        pm = np.array([0, 1, 0, 1, 1, 0])
        cm = np.array([0, 0, 1, 1, 1, 0])
        self.assertEqual(route_to_correlated(label, pm, cm).astype(int).tolist(), [0, 1, 0, 0, 0, 0])

    def test_route_target_each_outcome_single_row(self):
        # (label, pm, cm) -> expected y_router
        cases = {
            "both_correct": ((0, 0, 0), 0),
            "rescue": ((0, 1, 0), 1),
            "harmful_switch": ((0, 0, 1), 0),
            "both_wrong": ((0, 1, 1), 0),
        }
        for name, ((y, p, c), expected) in cases.items():
            with self.subTest(outcome=name):
                self.assertEqual(int(route_to_correlated([y], [p], [c])[0]), expected)
                self.assertTrue(outcome_masks([y], [p], [c])[name][0])

    def test_route_target_with_label_one(self):
        # label = 1 flips which prediction counts as correct
        label = [1, 1, 1, 1]
        pm = [1, 0, 1, 0]
        cm = [1, 1, 0, 0]
        self.assertEqual(route_to_correlated(label, pm, cm).astype(int).tolist(), [0, 1, 0, 0])
        masks = outcome_masks(label, pm, cm)
        self.assertEqual(masks["both_correct"].tolist(), [True, False, False, False])
        self.assertEqual(masks["harmful_switch"].tolist(), [False, False, True, False])
        self.assertEqual(masks["both_wrong"].tolist(), [False, False, False, True])

    def test_disagreement_alone_is_not_positive_target(self):
        # harmful switch: PM and CM disagree, but target must stay 0
        self.assertEqual(route_to_correlated([0], [0], [1]).astype(int).tolist(), [0])

    def test_masks_cover_every_row_without_overlap(self):
        label = [0, 1, 0, 1, 0, 1, 1, 0]
        pm = [0, 0, 1, 1, 1, 0, 1, 0]
        cm = [1, 1, 0, 0, 1, 0, 1, 0]
        masks = outcome_masks(label, pm, cm)
        total = sum(m.astype(int) for m in masks.values())
        self.assertEqual(total.tolist(), [1] * len(label))

    def test_route_rejects_unequal_lengths(self):
        with self.assertRaises(ValueError):
            route_to_correlated([0, 1], [0, 1], [0])
