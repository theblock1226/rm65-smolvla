"""RM65 DAgger-lite: HG-DAgger style collect loop for local fork.

Path B — not the official ``lerobot-rollout --strategy.type=dagger`` stack.
Reuses the same hold-to-intervene UX as RLT Stage2 for consistency.

Phases (clutch-style, gamepad hold)::

    AUTONOMOUS  --(hold)-->  CORRECTING  --(release)-->  AUTONOMOUS

Recording modes:
  - corrections_only: only human frames become episodes (default, classic DAgger)
  - record_all: autonomous + correction frames; correction tagged intervention=True
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot.common.policies.smolvla_rlt.robot_adapter import RM65_JOINT_KEYS

logger = logging.getLogger(__name__)


class DAggerPhase(enum.Enum):
    AUTONOMOUS = "autonomous"
    CORRECTING = "correcting"


@dataclass
class DAggerFrame:
    state: np.ndarray  # (7,)
    action: np.ndarray  # (7,) executed / expert
    policy_action: np.ndarray  # (7,) what policy wanted (for analysis)
    intervention: bool
    images: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = "VLA"  # VLA | HUMAN


@dataclass
class DAggerEpisode:
    frames: list[DAggerFrame] = field(default_factory=list)
    task: str = ""

    @property
    def n_intervention(self) -> int:
        return sum(1 for f in self.frames if f.intervention)

    @property
    def n_autonomous(self) -> int:
        return sum(1 for f in self.frames if not f.intervention)


class DAggerLiteController:
    """Hold-button state machine + episode buffering."""

    def __init__(self, *, corrections_only: bool = True, task: str = ""):
        self.corrections_only = corrections_only
        self.task = task
        self.phase = DAggerPhase.AUTONOMOUS
        self._episode = DAggerEpisode(task=task)
        self.completed_episodes: list[DAggerEpisode] = []

    def reset_episode(self) -> None:
        self.phase = DAggerPhase.AUTONOMOUS
        self._episode = DAggerEpisode(task=self.task)

    def update_hold(self, holding: bool) -> DAggerPhase:
        prev = self.phase
        self.phase = DAggerPhase.CORRECTING if holding else DAggerPhase.AUTONOMOUS
        if prev != self.phase:
            logger.info("  DAgger phase: %s -> %s", prev.value, self.phase.value)
        return self.phase

    def on_step(
        self,
        *,
        state: np.ndarray,
        policy_action: np.ndarray,
        human_action: np.ndarray | None,
        images: dict[str, np.ndarray],
        holding: bool,
    ) -> tuple[np.ndarray, str]:
        """Return (action_to_execute, source). Optionally buffer a frame."""
        self.update_hold(holding)
        if self.phase is DAggerPhase.CORRECTING:
            assert human_action is not None
            action = np.asarray(human_action, dtype=np.float32)
            source = "HUMAN"
            intervention = True
        else:
            action = np.asarray(policy_action, dtype=np.float32)
            source = "VLA"
            intervention = False

        should_record = intervention or (not self.corrections_only)
        if should_record:
            self._episode.frames.append(
                DAggerFrame(
                    state=np.asarray(state, dtype=np.float32).copy(),
                    action=action.copy(),
                    policy_action=np.asarray(policy_action, dtype=np.float32).copy(),
                    intervention=intervention,
                    images={k: v.copy() for k, v in images.items()},
                    source=source,
                )
            )
        return action, source

    def end_episode(self) -> DAggerEpisode | None:
        """Finalize episode if it has frames; return it or None."""
        ep = self._episode
        self.reset_episode()
        if not ep.frames:
            return None
        # corrections_only: drop empty (already filtered); require ≥1 intervention
        if self.corrections_only and ep.n_intervention == 0:
            logger.info("  skip episode (no corrections)")
            return None
        self.completed_episodes.append(ep)
        return ep


def save_episode_bundle(ep: DAggerEpisode, out_dir: Path, episode_index: int) -> Path:
    """Save one episode as NumPy+JSON (+ optional PNG images).

    Real LeRobot parquet/video export is done by ``scripts/dagger_rm65_to_lerobot.py``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = out_dir / f"episode_{episode_index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    states = np.stack([f.state for f in ep.frames], axis=0)
    actions = np.stack([f.action for f in ep.frames], axis=0)
    policy_actions = np.stack([f.policy_action for f in ep.frames], axis=0)
    intervention = np.asarray([f.intervention for f in ep.frames], dtype=np.bool_)
    sources = [f.source for f in ep.frames]

    np.savez_compressed(
        ep_dir / "trajectory.npz",
        states=states,
        actions=actions,
        policy_actions=policy_actions,
        intervention=intervention,
        joint_keys=np.asarray(RM65_JOINT_KEYS),
    )

    cam_names: list[str] = []
    if ep.frames and ep.frames[0].images:
        cam_names = sorted(ep.frames[0].images.keys())
        for cam in cam_names:
            cam_dir = ep_dir / "images" / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(ep.frames):
                img = fr.images[cam]
                # Accept CHW float[0,1] or HWC uint8
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    arr = np.transpose(img, (1, 2, 0))
                else:
                    arr = img
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                # Use numpy-only PPM to avoid PIL dependency in collector
                _write_ppm(cam_dir / f"frame_{i:06d}.ppm", arr)

    meta = {
        "task": ep.task,
        "n_frames": len(ep.frames),
        "n_intervention": ep.n_intervention,
        "n_autonomous": ep.n_autonomous,
        "sources": sources,
        "joint_keys": RM65_JOINT_KEYS,
        "cameras": cam_names,
        "image_ext": "ppm" if cam_names else None,
        "format": "dagger_rm65_episode_v1",
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return ep_dir


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    """Write HWC uint8 RGB as binary PPM (P6)."""
    h, w, c = rgb.shape
    assert c == 3
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    path.write_bytes(header + np.ascontiguousarray(rgb).tobytes())


def write_session_report(out_dir: Path, episodes: list[DAggerEpisode], extra: dict | None = None) -> dict:
    report = {
        "n_episodes": len(episodes),
        "n_frames": sum(len(e.frames) for e in episodes),
        "n_intervention_frames": sum(e.n_intervention for e in episodes),
        "n_autonomous_frames": sum(e.n_autonomous for e in episodes),
        "tasks": list({e.task for e in episodes}),
        **(extra or {}),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
