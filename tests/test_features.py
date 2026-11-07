"""Hand-checked tests for src/router/features.py (Week 2 Task 1)."""

import copy
import os
import unittest

import numpy as np

from router.features import (
    ACTIVITY_COLUMNS,
    FEATURE_COLUMNS,
    MAX_TIME_STEPS,
    SOURCE_FIELDS,
    extract_features,
    row_features,
)

FORBIDDEN = {
    "label", "pymatching", "correlated_matching", "decoder_correct",
    "y_router", "experiment",
}


def make_row(**changes):
    # 3 time steps, width 4, real widths 2, 3, 1. Grid:
    #   t0: 1 0 | 1 1   (last two are padding)
    #   t1: 1 1 0 | 1   (last one is padding)
    #   t2: 0 | 1 1 1   (last three are padding)
    row = {
        "syndrome": [1, 0, 1, 1,  1, 1, 0, 1,  0, 1, 1, 1],
        "round_widths": [2, 3, 1],
        "time_steps": 3,
        "width": 4,
        "distance": 3,
        "rounds": 2,
        "basis": "X",
        "label": 1,
        "pymatching": 0,
        "correlated_matching": 1,
        "experiment": "exp_a",
    }
    row.update(changes)
    return row


def as_columns(*rows):
    return {key: [r[key] for r in rows] for key in rows[0]}


def one(row):
    return {k: v[0] for k, v in extract_features(as_columns(row)).items()}


class FeatureTests(unittest.TestCase):
    def test_padding_is_ignored(self):
        f = one(make_row())
        # real events: t0 -> 1, t1 -> 2, t2 -> 0. Padded 1s must not count.
        self.assertEqual(f["detection_event_count"], 3)
        self.assertAlmostEqual(f["event_density"], 3 / 6)
        self.assertEqual([f[c] for c in ACTIVITY_COLUMNS[:4]], [1, 2, 0, 0])
        self.assertEqual(f["active_rounds"], 2)
        self.assertEqual(f["maximum_round_activity"], 2)
        self.assertEqual((f["distance"], f["rounds"], f["basis"]), (3, 2, "X"))

    def test_all_zero_syndrome(self):
        f = one(make_row(syndrome=[0] * 12))
        self.assertEqual(f["detection_event_count"], 0)
        self.assertEqual(f["event_density"], 0.0)
        self.assertEqual(f["active_rounds"], 0)
        self.assertEqual(f["maximum_round_activity"], 0)
        self.assertTrue(all(f[c] == 0 for c in ACTIVITY_COLUMNS))

    def test_zero_real_positions_gives_zero_density(self):
        f = one(make_row(round_widths=[0, 0, 0]))
        self.assertEqual(f["detection_event_count"], 0)
        self.assertEqual(f["event_density"], 0.0)

    def test_rows_with_different_widths_and_time_steps(self):
        a = make_row()
        # 2 time steps, width 3, real widths 3 and 2: t0 = 0 1 0, t1 = 1 0 | 1 (last is padding)
        b = make_row(syndrome=[0, 1, 0, 1, 0, 1], round_widths=[3, 2],
                     time_steps=2, width=3, rounds=1, basis="Z", distance=5)
        out = extract_features(as_columns(a, b))
        self.assertEqual(out["detection_event_count"].tolist(), [3, 2])
        self.assertEqual(out["event_density"].tolist(), [3 / 6, 2 / 5])
        self.assertEqual(out["activity_round_00"].tolist(), [1, 1])
        self.assertEqual(out["activity_round_01"].tolist(), [2, 1])
        self.assertEqual(out["active_rounds"].tolist(), [2, 2])
        self.assertEqual(out["maximum_round_activity"].tolist(), [2, 1])
        self.assertEqual(out["basis"].tolist(), ["X", "Z"])
        self.assertEqual(out["distance"].tolist(), [3, 5])

    def test_final_time_step_counts_as_active(self):
        f = one(make_row(syndrome=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
        self.assertEqual(f["active_rounds"], 1)
        self.assertEqual(f["activity_round_02"], 1)

    def test_exact_column_order_and_no_forbidden_fields(self):
        out = extract_features(as_columns(make_row()))
        self.assertEqual(tuple(out), FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 2 + MAX_TIME_STEPS + 5)
        self.assertEqual(FEATURE_COLUMNS[:3], ("detection_event_count", "event_density",
                                               "activity_round_00"))
        self.assertEqual(FEATURE_COLUMNS[-5:], ("active_rounds", "maximum_round_activity",
                                                "distance", "rounds", "basis"))
        self.assertFalse(FORBIDDEN & set(FEATURE_COLUMNS))
        self.assertFalse(FORBIDDEN & set(SOURCE_FIELDS))

    def test_forbidden_fields_do_not_change_x(self):
        base = extract_features(as_columns(make_row()))
        changed = extract_features(as_columns(make_row(
            label=0, pymatching=1, correlated_matching=0, experiment="exp_z",
            y_router=1, decoder_correct=0)))
        for name in FEATURE_COLUMNS:
            self.assertEqual(base[name].tolist(), changed[name].tolist(), name)

    def test_forbidden_fields_can_be_absent(self):
        row = {k: v for k, v in make_row().items() if k in SOURCE_FIELDS}
        self.assertEqual(one(row)["detection_event_count"], 3)

    def test_short_rows_are_padded_and_long_rows_rejected(self):
        f = one(make_row())
        self.assertEqual(len(ACTIVITY_COLUMNS), 26)
        self.assertEqual(f["activity_round_25"], 0)
        too_long = make_row(syndrome=[0] * 27, round_widths=[1] * 27,
                            time_steps=27, width=1, rounds=26)
        with self.assertRaisesRegex(ValueError, "longer than"):
            extract_features(as_columns(too_long))

    def test_invalid_inputs_raise(self):
        bad = {
            "wrong syndrome length": make_row(syndrome=[0] * 11),
            "wrong round_widths length": make_row(round_widths=[2, 3]),
            "width above max": make_row(round_widths=[2, 5, 1]),
            "negative width": make_row(round_widths=[2, -1, 1]),
            "non-binary syndrome": make_row(syndrome=[0, 2] + [0] * 10),
            "zero time_steps": make_row(time_steps=0),
            "float time_steps": make_row(time_steps=3.5),
            "bad basis": make_row(basis="Y"),
        }
        for name, row in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                extract_features(as_columns(row))

    def test_missing_field_and_ragged_columns_raise(self):
        cols = as_columns(make_row())
        del cols["basis"]
        with self.assertRaisesRegex(ValueError, "missing required field"):
            extract_features(cols)
        cols = as_columns(make_row(), make_row())
        cols["distance"] = [3]
        with self.assertRaisesRegex(ValueError, "expected 2"):
            extract_features(cols)

    def test_deterministic_and_input_not_modified(self):
        cols = as_columns(make_row(), make_row(basis="Z"))
        before = copy.deepcopy(cols)
        first = extract_features(cols)
        second = extract_features(cols)
        self.assertEqual(cols, before)
        for name in FEATURE_COLUMNS:
            self.assertEqual(first[name].tolist(), second[name].tolist())

    def test_numeric_columns_are_finite(self):
        out = extract_features(as_columns(make_row(), make_row(syndrome=[0] * 12)))
        for name in FEATURE_COLUMNS:
            if name != "basis":
                self.assertTrue(np.isfinite(out[name]).all(), name)

    def test_config_matches_code(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not installed")
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "features.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["max_time_steps"], MAX_TIME_STEPS)
        self.assertEqual(tuple(cfg["source_fields"]), SOURCE_FIELDS)


if __name__ == "__main__":
    unittest.main()
