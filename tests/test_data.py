"""Checks for bounded, reproducible dataset access without network calls."""

import sys
import types
import unittest
from unittest.mock import Mock, mock_open, patch

from router.data import load_surface_code_dataset, sample_split


class FakeSplit:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def select(self, indices):
        return [self.values[index] for index in indices]


class DataTests(unittest.TestCase):
    def test_loader_requests_only_train_by_default(self):
        config = {
            "dataset_name": "example/surface-code",
            "dataset_revision": "abc123",
            "paths": {"cache_dir": ".cache/huggingface"},
        }
        loaded_split = FakeSplit(list(range(10)))
        yaml_module = types.ModuleType("yaml")
        yaml_module.safe_load = Mock(return_value=config)
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = Mock(return_value=loaded_split)

        with patch.dict(sys.modules, {"yaml": yaml_module, "datasets": datasets_module}):
            with patch("builtins.open", mock_open(read_data="config content")):
                splits, returned_config = load_surface_code_dataset("config.yaml")

        self.assertEqual(list(splits), ["train"])
        self.assertIs(splits["train"], loaded_split)
        self.assertIs(returned_config, config)
        datasets_module.load_dataset.assert_called_once_with(
            "example/surface-code",
            split="train",
            cache_dir=".cache/huggingface",
            revision="abc123",
        )

    def test_loader_can_request_test_only_with_explicit_opt_in(self):
        yaml_module = types.ModuleType("yaml")
        yaml_module.safe_load = Mock(return_value={"dataset_name": "example/surface-code"})
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = Mock(return_value=FakeSplit(list(range(5))))

        with patch.dict(sys.modules, {"yaml": yaml_module, "datasets": datasets_module}):
            with patch("builtins.open", mock_open(read_data="config content")):
                splits, _ = load_surface_code_dataset(
                    "config.yaml", split_name="test", allow_test=True
                )

        self.assertEqual(list(splits), ["test"])
        datasets_module.load_dataset.assert_called_once_with(
            "example/surface-code", split="test", cache_dir=".cache/huggingface"
        )

    def test_sample_is_reproducible_bounded_and_without_replacement(self):
        values = list(range(100))
        splits = {"train": FakeSplit(values)}

        first = sample_split(splits, sample_size=12, seed=42)
        again = sample_split(splits, sample_size=12, seed=42)
        other_seed = sample_split(splits, sample_size=12, seed=43)

        self.assertEqual(first, again)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertEqual(splits["train"].values, values)

    def test_rejects_missing_split_and_invalid_sample_size(self):
        splits = {"train": FakeSplit(list(range(5)))}

        with self.assertRaisesRegex(ValueError, "not found"):
            sample_split(splits, split_name="validation", sample_size=1, seed=42)
        for size in (None, 0, -1, True, 6):
            with self.subTest(size=size), self.assertRaises(ValueError):
                sample_split(splits, sample_size=size, seed=42)
        with self.assertRaisesRegex(ValueError, "seed"):
            sample_split(splits, sample_size=1)

    def test_test_split_requires_explicit_final_evaluation_opt_in(self):
        test_split = FakeSplit(list(range(5)))
        with self.assertRaisesRegex(ValueError, "reserved for final evaluation"):
            load_surface_code_dataset(split_name="test")
        with self.assertRaisesRegex(ValueError, "reserved for final evaluation"):
            load_surface_code_dataset(split_name="test", allow_test="yes")
        with self.assertRaisesRegex(ValueError, "reserved for final evaluation"):
            sample_split({"test": test_split}, split_name="test", sample_size=2, seed=42)

        sample = sample_split(
            {"test": test_split}, split_name="test", sample_size=2, seed=42, allow_test=True
        )
        self.assertEqual(len(sample), 2)
