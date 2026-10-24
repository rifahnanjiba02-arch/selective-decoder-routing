"""Load one published split and draw a reproducible development sample.

The default is the training split. Request the test split only for final
evaluation, after all model and threshold choices are frozen.
"""

from random import Random


def _check_split_access(split_name, allow_test):
    if not isinstance(split_name, str) or split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    if split_name == "test" and allow_test is not True:
        raise ValueError("test split is reserved for final evaluation; pass allow_test=True")


def load_surface_code_dataset(
    config_path="configs/data.yaml", split_name="train", *, allow_test=False
):
    """Request one named split and return ``({split_name: Dataset}, config)``.

    The caller can pass the returned mapping directly to :func:`sample_split`.
    The config path is resolved from the current working directory.
    """
    _check_split_access(split_name, allow_test)

    import yaml
    from datasets import load_dataset

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    dataset_name = config["dataset_name"]
    cache_dir = config.get("paths", {}).get("cache_dir", ".cache/huggingface")
    load_options = {"split": split_name, "cache_dir": cache_dir}
    if config.get("dataset_revision"):
        load_options["revision"] = config["dataset_revision"]

    dataset = load_dataset(dataset_name, **load_options)
    return {split_name: dataset}, config


def sample_split(dataset, split_name="train", sample_size=None, seed=None, *, allow_test=False):
    """Select ``sample_size`` distinct rows in seeded random order.

    Only the requested number of row indices is generated; the full split is
    not shuffled. Pass the size and seed recorded in ``configs/data.yaml``.
    """
    _check_split_access(split_name, allow_test)
    if split_name not in dataset:
        raise ValueError(
            f"Split '{split_name}' not found. Available splits: {list(dataset.keys())}"
        )
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    split_data = dataset[split_name]
    if sample_size > len(split_data):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds '{split_name}' split size ({len(split_data)})"
        )

    indices = Random(seed).sample(range(len(split_data)), sample_size)
    return split_data.select(indices)
