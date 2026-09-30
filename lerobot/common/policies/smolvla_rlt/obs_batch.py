"""Build SmolVLA batches from RobotObservation (S2-3 observation alignment)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def image_to_chw_float(img: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Accept HWC/CHW uint8 or float → (C,H,W) float32 in [0,1] on device."""
    if isinstance(img, torch.Tensor):
        t = img.detach()
    else:
        t = torch.as_tensor(img)
    if t.ndim != 3:
        raise ValueError(f"expected 3D image, got {tuple(t.shape)}")
    if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    t = t.to(device=device, dtype=torch.float32)
    if float(t.max()) > 1.5:
        t = t / 255.0
    return t


def build_vla_batch(
    *,
    state: np.ndarray,
    images: dict[str, np.ndarray],
    task: str,
    device: torch.device,
    expected_image_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Map adapter observation → keys SmolVLA ``prepare_*`` expects.

    Image keys become ``observation.images.<name>``.
    If ``expected_image_keys`` is set (from policy config), only those names
    are filled; missing ones are skipped so ``empty_cameras`` can pad.
    """
    state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device).unsqueeze(0)
    batch: dict[str, Any] = {
        "observation.state": state_t,
        "task": task,
    }
    if expected_image_keys is None:
        for name, img in images.items():
            t = image_to_chw_float(img, device)
            batch[f"observation.images.{name}"] = t.unsqueeze(0)
        return batch

    # Policy configs use camera1/camera2. USB observations use front/mono.
    from lerobot.common.policies.smolvla_rlt.robot_adapter import CAMERA_KEY_ALIASES

    for key in expected_image_keys:
        short = key.split(".")[-1]
        source = short if short in images else None
        if source is None and key in images:
            source = key
        if source is None:
            for alias in CAMERA_KEY_ALIASES.get(short, ()):
                if alias in images:
                    source = alias
                    break
        if source is None:
            continue
        t = image_to_chw_float(images[source], device)
        batch[key if key.startswith("observation.images.") else f"observation.images.{short}"] = t.unsqueeze(0)
    return batch


def describe_batch_images(batch: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    out = {}
    for k, v in batch.items():
        if k.startswith("observation.images.") and isinstance(v, torch.Tensor):
            out[k] = tuple(v.shape)
    return out
