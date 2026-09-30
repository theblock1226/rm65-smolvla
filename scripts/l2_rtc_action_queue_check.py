#!/usr/bin/env python3
"""L2 simulated closed-loop check: ActionQueue + SmolVLA RTC.

Does **not** need a real robot. Simulates control ticks and a fixed inference
delay (in action steps). Compares RTC vs non-RTC queue behavior.

Success criteria
----------------
1. ActionQueue: after consuming K steps, ``get_left_over()`` length == remaining.
2. RTC merge: queue size after merge == chunk_size - delay.
3. Closed loop (RTC): zero underruns over ``control_steps`` when requesting the
   next chunk early enough (qsize <= delay + margin).
4. Continuity: mean |Δaction| at chunk-handoff steps is lower with RTC than
   without (or RTC underruns == 0 and handoff MAE to planned leftover is small).

Usage (inside rm65_dev2)::

    cd /home/sdznkj/zkx_ws/sendlerobot
    PYTHONPATH=. python3 scripts/l2_rtc_action_queue_check.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import torch


def _build_batch(policy, device: torch.device, seed: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    state_dim = policy.config.robot_state_feature.shape[0]
    img = torch.rand(1, 3, 240, 320, generator=g)
    state = torch.randn(1, state_dim, generator=g)
    batch = {
        "observation.state": state.to(device),
        "task": "Grasp a lego block and put it in the bin.",
    }
    for key in policy.config.image_features:
        batch[key] = img.clone().to(device)
    return batch


def _set_rtc(policy, *, enabled: bool, execution_horizon: int, max_guidance_weight: float = 10.0) -> None:
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.configs.types import RTCAttentionSchedule

    policy.config.rtc_config = RTCConfig(
        enabled=enabled,
        mode="guided",
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
        debug=False,
    )
    policy.init_rtc_processor()


@dataclass
class LoopStats:
    underruns: int = 0
    merges: int = 0
    executed: int = 0
    handoff_deltas: list[float] | None = None
    leftover_maes: list[float] | None = None

    def __post_init__(self):
        if self.handoff_deltas is None:
            self.handoff_deltas = []
        if self.leftover_maes is None:
            self.leftover_maes = []


@torch.no_grad()
def _predict_chunk(policy, batch, noise, *, leftover, delay, horizon):
    kwargs = {}
    if policy.config.rtc_config is not None and policy.config.rtc_config.enabled:
        kwargs = {
            "prev_chunk_left_over": leftover,
            "inference_delay": delay,
            "execution_horizon": horizon,
        }
    actions = policy.predict_action_chunk(batch, noise=noise, **kwargs)
    # (1, T, A) -> (T, A)
    return actions.squeeze(0).cpu()


def _check_queue_unit(delay: int, chunk: int, action_dim: int) -> dict:
    """Pure ActionQueue mechanics (no policy)."""
    from lerobot.common.policies.rtc.action_queue import ActionQueue
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig

    cfg = RTCConfig(enabled=True, execution_horizon=10)
    q = ActionQueue(cfg)
    actions = torch.arange(chunk * action_dim, dtype=torch.float32).reshape(chunk, action_dim)

    q.merge(actions, actions, real_delay=0)
    assert q.qsize() == chunk

    for _ in range(delay):
        assert q.get() is not None
    left = q.get_left_over()
    assert left is not None and left.shape[0] == chunk - delay

    # New chunk arrives after simulated inference of `delay` steps
    new_actions = actions + 1000
    q.merge(new_actions, new_actions, real_delay=delay)
    ok_size = q.qsize() == chunk - delay
    # First queued action should be new_actions[delay]
    first = q.get()
    ok_first = first is not None and torch.allclose(first, new_actions[delay])

    return {
        "leftover_len_ok": left.shape[0] == chunk - delay,
        "merge_size_ok": ok_size,
        "merge_skip_ok": bool(ok_first),
        "pass": bool(left.shape[0] == chunk - delay and ok_size and ok_first),
    }


@torch.no_grad()
def _run_loop(
    policy,
    *,
    device: torch.device,
    rtc: bool,
    delay: int,
    horizon: int,
    control_steps: int,
    seed: int,
    request_margin: int,
) -> LoopStats:
    from lerobot.common.policies.rtc.action_queue import ActionQueue
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig

    # Policy RTC on/off; queue always uses replace+delay semantics (RTC queue mode).
    _set_rtc(policy, enabled=rtc, execution_horizon=horizon)
    q = ActionQueue(RTCConfig(enabled=True, execution_horizon=horizon))
    stats = LoopStats()
    chunk = policy.config.chunk_size
    noise_shape = (1, chunk, policy.config.max_action_dim)

    prev_action = None
    bootstrapped = False

    t = 0
    while t < control_steps:
        need = (not bootstrapped) or (q.qsize() <= delay + request_margin)
        if need:
            if not bootstrapped:
                batch = _build_batch(policy, device, seed=seed)
                noise = torch.randn(noise_shape, device=device)
                actions = _predict_chunk(
                    policy, batch, noise, leftover=None, delay=delay, horizon=horizon
                )
                q.merge(actions, actions, real_delay=0)
                stats.merges += 1
                bootstrapped = True
                continue

            idx_before = q.get_action_index()
            leftover = q.get_left_over()

            # Consume exactly `delay` actions while "inferring"
            for _ in range(delay):
                if t >= control_steps:
                    break
                a = q.get()
                if a is None:
                    stats.underruns += 1
                    a = torch.zeros(policy.config.action_feature.shape[0])
                else:
                    stats.executed += 1
                if prev_action is not None:
                    stats.handoff_deltas.append(float((a - prev_action).abs().mean()))
                prev_action = a
                t += 1

            batch = _build_batch(policy, device, seed=seed + stats.merges)
            noise = torch.randn(noise_shape, device=device)
            actions = _predict_chunk(
                policy,
                batch,
                noise,
                leftover=leftover if rtc else None,
                delay=delay,
                horizon=horizon,
            )
            q.merge(
                actions,
                actions,
                real_delay=delay,
                action_index_before_inference=idx_before,
            )
            stats.merges += 1

            # New head should track leftover[delay] (post-inference continuity)
            if leftover is not None and leftover.shape[0] > delay and q.queue is not None:
                ref = leftover[delay]
                head = q.queue[0]
                stats.leftover_maes.append(float((head - ref).abs().mean()))
            continue

        a = q.get()
        if a is None:
            stats.underruns += 1
            a = torch.zeros(policy.config.action_feature.shape[0])
        else:
            stats.executed += 1
        if prev_action is not None:
            stats.handoff_deltas.append(float((a - prev_action).abs().mean()))
        prev_action = a
        t += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="L2 RTC ActionQueue closed-loop check")
    parser.add_argument(
        "--policy_path",
        type=str,
        default="/home/sdznkj/zkx_ws/sendlerobot/outputs/test/060000/pretrained_model",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference_delay", type=int, default=4)
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument("--control_steps", type=int, default=120)
    parser.add_argument("--request_margin", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    delay = args.inference_delay
    horizon = args.execution_horizon

    # --- Part A: queue unit ---
    unit = _check_queue_unit(delay=delay, chunk=50, action_dim=6)
    print("queue_unit:", json.dumps(unit, ensure_ascii=False))
    if not unit["pass"]:
        print("FAIL: ActionQueue unit checks", file=sys.stderr)
        return 2

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"Loading policy from {args.policy_path} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    # --- Part B: closed loops ---
    stats_rtc = _run_loop(
        policy,
        device=device,
        rtc=True,
        delay=delay,
        horizon=horizon,
        control_steps=args.control_steps,
        seed=args.seed,
        request_margin=args.request_margin,
    )
    stats_nortc = _run_loop(
        policy,
        device=device,
        rtc=False,
        delay=delay,
        horizon=horizon,
        control_steps=args.control_steps,
        seed=args.seed,
        request_margin=args.request_margin,
    )

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    report = {
        "inference_delay": delay,
        "execution_horizon": horizon,
        "control_steps": args.control_steps,
        "rtc": {
            "underruns": stats_rtc.underruns,
            "merges": stats_rtc.merges,
            "executed": stats_rtc.executed,
            "mean_leftover_head_mae": round(_mean(stats_rtc.leftover_maes), 6),
            "mean_step_delta": round(_mean(stats_rtc.handoff_deltas), 6),
            "n_leftover_cmp": len(stats_rtc.leftover_maes),
        },
        "no_rtc": {
            "underruns": stats_nortc.underruns,
            "merges": stats_nortc.merges,
            "executed": stats_nortc.executed,
            "mean_leftover_head_mae": round(_mean(stats_nortc.leftover_maes), 6),
            "mean_step_delta": round(_mean(stats_nortc.handoff_deltas), 6),
            "n_leftover_cmp": len(stats_nortc.leftover_maes),
        },
    }

    # Pass rules:
    # 1) unit already passed
    # 2) RTC loop has zero underruns
    # 3) RTC leftover-head MAE is finite and strictly below no-RTC (when both have samples)
    rtc_mae = _mean(stats_rtc.leftover_maes)
    nortc_mae = _mean(stats_nortc.leftover_maes)
    continuity_ok = (
        len(stats_rtc.leftover_maes) > 0
        and len(stats_nortc.leftover_maes) > 0
        and rtc_mae < nortc_mae
    )
    passed = stats_rtc.underruns == 0 and continuity_ok
    report["pass"] = passed
    report["pass_reasons"] = {
        "rtc_zero_underruns": stats_rtc.underruns == 0,
        "rtc_leftover_closer_than_nortc": continuity_ok,
        "rtc_mae": None if rtc_mae != rtc_mae else round(rtc_mae, 6),
        "nortc_mae": None if nortc_mae != nortc_mae else round(nortc_mae, 6),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if passed:
        print(
            f"PASS: L2 RTC queue loop ok "
            f"(underruns=0, leftover_head_mae rtc={rtc_mae:.4f} < no_rtc={nortc_mae:.4f})"
        )
        return 0

    print("FAIL: L2 criteria not met", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
