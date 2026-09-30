"""Offline LeRobot dataset helpers for RLT Stage1 (no HuggingFace Hub)."""

from __future__ import annotations

import os
from pathlib import Path


def force_hf_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_local_lerobot_dataset(
    dataset_root: str | Path,
    *,
    episodes: list[int] | None = None,
    download_videos: bool = False,
    repo_id: str | None = None,
    delta_timestamps: dict | None = None,
):
    """Load a local cleaned/ LeRobot v2.1 tree without Hub access.

    ``dataset_root`` must be the dataset directory itself
    (contains ``meta/``, ``data/``, ``videos/``), e.g.
    ``datasets/cleaned/rm65_stacking_blocks``.
    """
    force_hf_offline()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root).resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root (missing meta/info.json): {root}")

    # Fake local id — must never be fetched from Hub; files already under root.
    rid = repo_id or f"local/{root.name}"
    return LeRobotDataset(
        rid,
        root=root,
        episodes=episodes,
        download_videos=download_videos,
        delta_timestamps=delta_timestamps,
    )
