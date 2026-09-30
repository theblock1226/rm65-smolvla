"""Software safety gate for RM65 joint commands (deadman + rate limit + watchdog).

Does **not** replace a hardware E-Stop. Use before ``SendRobot.send_action``.

Typical bring-up::

    gate = SafeActionGate(SafetyConfig(speed_scale=0.3, require_deadman=True))
    cmd = gate.filter(desired, current_state=state, deadman_held=held)
    if cmd is not None:
        robot.send_action(cmd)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Defaults are conservative for first real-robot sessions."""

    enabled: bool = True
    # Hold-to-arm: no motion unless deadman is held (when require_deadman=True)
    require_deadman: bool = True
    deadman_button: int = field(
        default_factory=lambda: int(os.environ.get("RLT_DEADMAN_BUTTON", "5"))
    )
    # Max |Δq| per control tick (joints in deg-like units; gripper in [0,1] scale)
    max_delta_joints: float = 2.0  # ~40 deg/s at 20 Hz
    max_delta_gripper: float = 0.08
    # Global shrink of commanded step after clipping (0~1)
    speed_scale: float = 0.3
    # If filter() not called within this window while armed → disarm
    watchdog_timeout_s: float = 0.25
    action_dim: int = 7

    def __post_init__(self) -> None:
        if self.speed_scale <= 0 or self.speed_scale > 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {self.speed_scale}")
        if self.max_delta_joints <= 0 or self.max_delta_gripper <= 0:
            raise ValueError("max_delta_* must be positive")
        if self.watchdog_timeout_s <= 0:
            raise ValueError("watchdog_timeout_s must be positive")


@dataclass
class SafetyStatus:
    armed: bool
    deadman: bool
    blocked_reason: str = ""
    clipped: bool = False
    watchdog_tripped: bool = False
    last_command: np.ndarray | None = None


class SafeActionGate:
    """Gate + per-tick joint rate limit between policy and robot."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._force_deadman: bool | None = None  # test override
        self._last_filter_t = time.perf_counter()
        self._watchdog_tripped = False
        self._blocked_count = 0
        self._clip_count = 0
        self._sent_count = 0
        self.last_status = SafetyStatus(armed=False, deadman=False)

    def set_force_deadman(self, value: bool | None) -> None:
        """Mock/test: override hardware deadman (None = use poll result)."""
        self._force_deadman = value

    def reset(self) -> None:
        self._watchdog_tripped = False
        self._last_filter_t = time.perf_counter()
        self.last_status = SafetyStatus(armed=False, deadman=False)

    def _max_delta_vec(self) -> np.ndarray:
        d = np.full(self.config.action_dim, self.config.max_delta_joints, dtype=np.float32)
        if self.config.action_dim >= 7:
            d[6] = self.config.max_delta_gripper
        return d

    def rate_limit(
        self, desired: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Clip step toward ``desired`` from ``current``; apply speed_scale."""
        desired = np.asarray(desired, dtype=np.float32).reshape(-1)
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        if desired.shape[0] != self.config.action_dim:
            raise ValueError(f"desired dim {desired.shape} != {self.config.action_dim}")
        delta = desired - current
        max_d = self._max_delta_vec()
        clipped_delta = np.clip(delta, -max_d, max_d)
        clipped = not np.allclose(clipped_delta, delta, rtol=0.0, atol=1e-6)
        step = clipped_delta * float(self.config.speed_scale)
        return (current + step).astype(np.float32), clipped

    def filter(
        self,
        desired: np.ndarray,
        *,
        current_state: np.ndarray,
        deadman_held: bool,
    ) -> np.ndarray | None:
        """Return command to send, or None if motion is blocked.

        When blocked, caller must **not** send a new target (hold last pose on robot).
        """
        now = time.perf_counter()
        cfg = self.config
        if not cfg.enabled:
            self.last_status = SafetyStatus(
                armed=True, deadman=True, last_command=np.asarray(desired, dtype=np.float32)
            )
            self._sent_count += 1
            return np.asarray(desired, dtype=np.float32).reshape(cfg.action_dim)

        # Watchdog: long gap while we expected streaming → trip until reset
        dt = now - self._last_filter_t
        self._last_filter_t = now
        watchdog = False
        if dt > cfg.watchdog_timeout_s and self._sent_count > 0:
            watchdog = True
            if not self._watchdog_tripped:
                logger.warning(
                    "Safety watchdog tripped (dt=%.3fs > %.3fs) — blocking until reset()",
                    dt,
                    cfg.watchdog_timeout_s,
                )
            self._watchdog_tripped = True

        deadman = bool(deadman_held) if self._force_deadman is None else bool(self._force_deadman)

        if self._watchdog_tripped:
            self._blocked_count += 1
            self.last_status = SafetyStatus(
                armed=False,
                deadman=deadman,
                blocked_reason="watchdog",
                watchdog_tripped=True,
            )
            return None

        if cfg.require_deadman and not deadman:
            self._blocked_count += 1
            self.last_status = SafetyStatus(
                armed=False, deadman=False, blocked_reason="deadman_released"
            )
            return None

        cmd, clipped = self.rate_limit(desired, current_state)
        if clipped:
            self._clip_count += 1
        self._sent_count += 1
        self.last_status = SafetyStatus(
            armed=True, deadman=deadman, clipped=clipped, last_command=cmd.copy()
        )
        return cmd

    def stats(self) -> dict:
        return {
            "sent": self._sent_count,
            "blocked": self._blocked_count,
            "clipped": self._clip_count,
            "watchdog_tripped": self._watchdog_tripped,
            "config": {
                "enabled": self.config.enabled,
                "require_deadman": self.config.require_deadman,
                "deadman_button": self.config.deadman_button,
                "max_delta_joints": self.config.max_delta_joints,
                "max_delta_gripper": self.config.max_delta_gripper,
                "speed_scale": self.config.speed_scale,
                "watchdog_timeout_s": self.config.watchdog_timeout_s,
            },
        }
