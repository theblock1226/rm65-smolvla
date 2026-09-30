#!/usr/bin/env python3
"""Profile VLA chunk latency + ActionQueue underruns (no robot required).

Async-style loop: inference runs in a worker thread while the control loop
keeps consuming the queue. New chunks merge with ``real_delay`` ≈ steps
consumed during inference.

Mock::

    PYTHONPATH=. python3 scripts/profile_inference_loop.py \\
      --policy_mode mock --fps 20 --control_steps 120 --fake_chunk_ms 150

SmolVLA::

    PYTHONPATH=. python3 scripts/profile_inference_loop.py \\
      --policy_mode smolvla --policy_path outputs/test/060000/pretrained_model \\
      --device cuda --control_steps 60
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Inference loop profiler")
    p.add_argument("--policy_mode", choices=["mock", "smolvla"], default="mock")
    p.add_argument("--policy_path", type=str, default="outputs/test/060000/pretrained_model")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--interpolation_multiplier", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--control_steps", type=int, default=120)
    p.add_argument("--request_when_qsize_le", type=int, default=25)
    p.add_argument("--fake_chunk_ms", type=float, default=120.0)
    p.add_argument("--output", type=str, default="outputs/profiling/inference_loop_report.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class MockChunkPolicy:
    def __init__(self, chunk_size: int, action_dim: int, fake_ms: float, seed: int):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.fake_ms = fake_ms
        self.rng = torch.Generator().manual_seed(seed)
        self.t = 0
        self._lock = threading.Lock()

    def predict_action_chunk(self, *_args, **_kwargs) -> torch.Tensor:
        time.sleep(self.fake_ms / 1000.0)
        with self._lock:
            base = torch.linspace(0, 1, self.chunk_size).unsqueeze(-1)
            noise = torch.randn(self.chunk_size, self.action_dim, generator=self.rng) * 0.02
            self.t += 1
            t = self.t
        return (base + noise + 0.01 * t).float()


def load_smolvla(path: str, device: torch.device):
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(path)
    policy.to(device)
    policy.eval()
    return policy


def build_batch(policy, device: torch.device, seed: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    state_dim = policy.config.robot_state_feature.shape[0]
    batch = {
        "observation.state": torch.randn(1, state_dim, generator=g).to(device),
        "task": "profile inference loop",
    }
    img = torch.rand(1, 3, 240, 320, generator=g).to(device)
    for key in policy.config.image_features:
        batch[key] = img.clone()
    return batch


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.interpolation_multiplier < 1:
        raise SystemExit("interpolation_multiplier must be >= 1")

    from lerobot.common.policies.rtc.action_interpolator import ActionInterpolator
    from lerobot.common.policies.rtc.action_queue import ActionQueue
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.common.policies.rtc.loop_metrics import InferenceLoopReport

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    report = InferenceLoopReport(
        target_fps=args.fps,
        interpolation_multiplier=args.interpolation_multiplier,
        policy_mode=args.policy_mode,
    )

    if args.policy_mode == "mock":
        policy = MockChunkPolicy(args.chunk_size, args.action_dim, args.fake_chunk_ms, args.seed)
        batch = None
        action_dim = args.action_dim
        chunk_size = args.chunk_size
    else:
        policy = load_smolvla(args.policy_path, device)
        batch = build_batch(policy, device, args.seed)
        action_dim = int(policy.config.action_feature.shape[0])
        chunk_size = int(policy.config.chunk_size)

    rtc_cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(rtc_cfg)
    interp = ActionInterpolator(multiplier=args.interpolation_multiplier)

    control_fps = args.fps * args.interpolation_multiplier
    control_dt = 1.0 / control_fps

    lock = threading.Lock()
    pending = False
    result: dict | None = None

    def infer_worker(start_index: int):
        nonlocal result, pending
        t0 = time.perf_counter()
        try:
            if args.policy_mode == "mock":
                chunk = policy.predict_action_chunk()
            else:
                noise = torch.randn(1, chunk_size, action_dim, device=device)
                chunk = policy.predict_action_chunk(batch, noise=noise).squeeze(0).detach().cpu()
            if chunk.ndim == 3:
                chunk = chunk.squeeze(0)
            latency = time.perf_counter() - t0
            with lock:
                result = {
                    "chunk": chunk.cpu(),
                    "latency": latency,
                    "start_index": start_index,
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception("inference failed: %s", exc)
            with lock:
                result = {"error": str(exc)}
        finally:
            with lock:
                pending = False

    def start_inference():
        nonlocal pending
        with lock:
            if pending:
                return
            pending = True
            start_index = queue.get_action_index()
        threading.Thread(target=infer_worker, args=(start_index,), daemon=True).start()

    def maybe_merge():
        nonlocal result
        with lock:
            payload = result
            result = None
        if payload is None:
            return
        if "error" in payload:
            return
        chunk = payload["chunk"]
        latency = payload["latency"]
        delay = int(round(latency * args.fps))
        delay = max(0, min(delay, max(0, len(chunk) - 1)))
        report.add_chunk_latency(latency)
        queue.merge(
            chunk,
            chunk,
            real_delay=delay,
            action_index_before_inference=payload["start_index"],
        )
        logger.info(
            "merged T=%d delay=%d latency=%.1fms qsize=%d",
            len(chunk),
            delay,
            latency * 1000,
            queue.qsize(),
        )

    # Prime first chunk synchronously so the loop has actions
    t0 = time.perf_counter()
    if args.policy_mode == "mock":
        chunk0 = policy.predict_action_chunk()
    else:
        noise = torch.randn(1, chunk_size, action_dim, device=device)
        chunk0 = policy.predict_action_chunk(batch, noise=noise).squeeze(0).cpu()
    if chunk0.ndim == 3:
        chunk0 = chunk0.squeeze(0)
    lat0 = time.perf_counter() - t0
    report.add_chunk_latency(lat0)
    queue.merge(chunk0.cpu(), chunk0.cpu(), real_delay=0)

    for step in range(args.control_steps):
        tick_t0 = time.perf_counter()
        underrun = False

        maybe_merge()

        if interp.needs_new_action():
            nxt = queue.get()
            if nxt is None:
                underrun = True
            else:
                interp.add(nxt)
                report.add_policy_action()

        _ = interp.get()
        qsize = queue.qsize()
        report.add_control_tick(
            dt_seconds=time.perf_counter() - tick_t0,
            queue_depth=qsize,
            underrun=underrun,
        )

        with lock:
            is_pending = pending
        if (not is_pending) and qsize <= args.request_when_qsize_le:
            start_inference()

        elapsed = time.perf_counter() - tick_t0
        sleep = control_dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

        if step % 30 == 0:
            logger.info(
                "step %d qsize=%d underruns=%d chunks=%d",
                step,
                qsize,
                report.underruns,
                report.n_chunks,
            )

    # Drain pending
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with lock:
            if not pending and result is None:
                break
        maybe_merge()
        time.sleep(0.01)
    maybe_merge()

    out = Path(args.output)
    report.save(out)
    summary = report.summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
