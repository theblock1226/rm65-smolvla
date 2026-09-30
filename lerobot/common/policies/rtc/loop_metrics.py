"""Inference / control-loop metrics for RTC + async client tuning."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InferenceLoopReport:
    """Aggregated stats + suggested RTC knobs (heuristic)."""

    n_control_ticks: int = 0
    n_policy_actions: int = 0
    n_chunks: int = 0
    underruns: int = 0
    chunk_latency_ms: list[float] = field(default_factory=list)
    control_dt_ms: list[float] = field(default_factory=list)
    queue_depth: list[int] = field(default_factory=list)
    target_fps: float = 20.0
    interpolation_multiplier: int = 1
    policy_mode: str = "mock"

    def add_chunk_latency(self, seconds: float) -> None:
        self.n_chunks += 1
        self.chunk_latency_ms.append(1000.0 * float(seconds))

    def add_control_tick(self, *, dt_seconds: float, queue_depth: int, underrun: bool) -> None:
        self.n_control_ticks += 1
        self.control_dt_ms.append(1000.0 * float(dt_seconds))
        self.queue_depth.append(int(queue_depth))
        if underrun:
            self.underruns += 1

    def add_policy_action(self) -> None:
        self.n_policy_actions += 1

    @staticmethod
    def _pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        import numpy as np

        return float(np.quantile(np.asarray(vals, dtype=np.float64), q))

    def summary(self) -> dict[str, Any]:
        target_ms = 1000.0 / max(1e-6, self.target_fps)
        chunk_p50 = self._pct(self.chunk_latency_ms, 0.5)
        chunk_p95 = self._pct(self.chunk_latency_ms, 0.95)
        chunk_max = max(self.chunk_latency_ms) if self.chunk_latency_ms else None
        q_mean = float(sum(self.queue_depth) / len(self.queue_depth)) if self.queue_depth else 0.0
        q_min = min(self.queue_depth) if self.queue_depth else 0

        # Heuristic: inference_delay ≈ ceil(p95_chunk / action_dt)
        inferred_delay = None
        if chunk_p95 is not None:
            inferred_delay = max(1, int(round(chunk_p95 / target_ms)))

        # execution_horizon often slightly larger than delay for guidance window
        suggested_horizon = None if inferred_delay is None else inferred_delay + max(2, inferred_delay // 2)

        pass_underrun = self.underruns == 0
        pass_latency = chunk_p95 is None or chunk_p95 < 3.0 * target_ms

        return {
            "n_control_ticks": self.n_control_ticks,
            "n_policy_actions": self.n_policy_actions,
            "n_chunks": self.n_chunks,
            "underruns": self.underruns,
            "target_fps": self.target_fps,
            "target_action_dt_ms": target_ms,
            "interpolation_multiplier": self.interpolation_multiplier,
            "control_fps": self.target_fps * self.interpolation_multiplier,
            "policy_mode": self.policy_mode,
            "chunk_latency_ms": {
                "p50": chunk_p50,
                "p95": chunk_p95,
                "max": chunk_max,
            },
            "queue_depth": {"mean": q_mean, "min": q_min},
            "suggested_rtc": {
                "inference_delay_steps": inferred_delay,
                "execution_horizon_steps": suggested_horizon,
                "note": "Heuristic from chunk p95 / action_dt; validate on real RM65 ckpt",
            },
            "pass_no_underrun": pass_underrun,
            "pass_chunk_p95_lt_3x_dt": pass_latency,
            "pass": pass_underrun and self.n_control_ticks > 0,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"raw": asdict(self), "summary": self.summary()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class Tic:
    """Simple perf counter context."""

    def __init__(self):
        self.t0 = time.perf_counter()

    def dt(self) -> float:
        return time.perf_counter() - self.t0
