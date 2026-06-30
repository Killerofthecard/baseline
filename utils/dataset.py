"""Shared GAIA dataset loader."""

import os


def load_gaia(
    config: str = "2023_level1",
    split: str = "validation",
    return_data_dir: bool = False,
):
    """Load the GAIA dataset split.

    Args:
        config: Dataset config name, e.g. "2023_level1" or "2023_all".
        split: Either "validation" (has ground-truth answers) or "test".
        return_data_dir: If True, also return the downloaded dataset directory.

    Returns:
        A HuggingFace datasets object (and optionally the data directory path).
    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
    dataset = load_dataset(data_dir, config, split=split)
    if return_data_dir:
        return dataset, data_dir
    return dataset


def attachment_path(data_dir: str, example: dict) -> str | None:
    """Return the absolute path to an example's attachment, if any."""
    rel_path = example.get("file_path") or example.get("file_name")
    if rel_path:
        return os.path.join(data_dir, rel_path)
    return None
