"""Dataset access, split selection, and development sampling (planned).

The source is Bauxitiego/surface-code-syndromes on Hugging Face. Use its
published train/validation/test splits; do not read test while designing the
router. Implement bounded development sampling before any full-data pipeline.
Keep downloaded data and caches outside Git.
"""
import yaml
from datasets import load_dataset


def load_surface_code_dataset(config_path="configs/data.yaml"):
    """Loads dataset configuration and downloads/loads the Hugging Face dataset."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    cache_dir = config.get("paths", {}).get("cache_dir", ".cache/huggingface")

    print(f"Loading Hugging Face dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    return dataset, config


def sample_split(dataset, split_name="train", sample_size=50000, seed=42):
    """Extracts a reproducible sample from the specified dataset split."""
    if split_name not in dataset:
        raise ValueError(
            f"Split '{split_name}' not found. Available splits: {list(dataset.keys())}"
        )

    split_data = dataset[split_name]

    if sample_size is None or sample_size >= len(split_data):
        actual_size = len(split_data)
    else:
        actual_size = sample_size

    print(
        f"Sampling {actual_size} rows from '{split_name}' split (seed={seed})..."
    )
    sampled = split_data.shuffle(seed=seed).select(range(actual_size))

    return sampled