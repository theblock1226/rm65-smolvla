"""In-process SmolVLA actor for RM65 DAgger collect."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_chw_float01(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 / CHW float → CHW float32 in [0, 1]."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        t = torch.from_numpy(arr.astype(np.float32))
        if float(t.max()) > 1.5:
            t = t / 255.0
        return t.clamp(0, 1)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)
        if float(t.max()) > 1.5:
            t = t / 255.0
        return t.clamp(0, 1)
    raise ValueError(f"unexpected image shape {arr.shape}")


class SmolVLAActor:
    """Thin wrapper: obs → 7-D action via ``select_action``."""

    def __init__(
        self,
        ckpt_dir: str | Path,
        *,
        device: str = "cuda",
        task: str = "",
        state_dim: int | None = None,
    ):
        from lerobot.scripts.server.rm65_ckpt import load_smolvla_for_server

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.task = task

        self.policy = load_smolvla_for_server(str(self.ckpt_dir), str(self.device))
        want = int(self.policy.config.robot_state_feature.shape[0])
        if state_dim is not None:
            want = int(state_dim)
        else:
            env_dim = os.environ.get("RM65_POLICY_STATE_DIM", "").strip()
            if env_dim and int(env_dim) > want:
                want = int(env_dim)
            elif env_dim and int(env_dim) < want:
                logger.warning(
                    "RM65_POLICY_STATE_DIM=%s is below checkpoint state %s; using the checkpoint",
                    env_dim,
                    want,
                )
        self.state_dim = want
        self.policy.reset()
        logger.info(
            "SmolVLAActor loaded %s device=%s state_dim=%d task=%r",
            self.ckpt_dir,
            self.device,
            self.state_dim,
            self.task,
        )

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state_in = state[: self.state_dim]

        batch: dict = {
            "task": self.task,
            "observation.state": torch.from_numpy(state_in).unsqueeze(0).to(self.device),
        }
        alias = {
            "camera1": ("camera1", "front"),
            "camera2": ("camera2", "astra"),
            "camera3": ("camera3",),
            "empty_camera_0": ("empty_camera_0",),
        }
        for feat_key, ft in self.policy.config.input_features.items():
            if not feat_key.startswith("observation.images."):
                continue
            cam = feat_key.split("observation.images.", 1)[-1]
            names = alias.get(cam, (cam,))
            img = None
            for n in names:
                if n in images and images[n] is not None:
                    img = images[n]
                    break
            if img is None:
                c, h, w = ft.shape
                batch[feat_key] = torch.zeros(1, c, h, w, device=self.device)
            else:
                t = _to_chw_float01(img)
                batch[feat_key] = t.unsqueeze(0).to(self.device)

        action = self.policy.select_action(batch)
        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            grip = float(state[6]) if state.shape[0] >= 7 else 1.0
            action = np.concatenate([action, [grip]])
        return action[:7].astype(np.float32)


class MockPolicy:
    """Deterministic-ish joint targets for dry-run / wiring tests."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, state: np.ndarray, images: dict[str, np.ndarray] | None = None) -> np.ndarray:
        del images
        self._t += 1
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] < 7:
            state = np.pad(state, (0, 7 - state.shape[0]))
        drift = np.sin(np.arange(7) + self._t * 0.1).astype(np.float32) * 0.05
        return (state[:7] + drift + self.rng.normal(0, 0.01, size=7).astype(np.float32)).astype(
            np.float32
        )
