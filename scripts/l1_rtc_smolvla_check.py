#!/usr/bin/env python3
"""L1 offline check: SmolVLA with vs without RTC prefix guidance.

Success criterion
-----------------
Same observation + same noise, leftover from a *different* reference chunk:
MAE(rtc_actions[:, :H], leftover[:, :H]) < MAE(no_rtc_actions[:, :H], leftover[:, :H])

Usage (inside rm65_dev2)::

    cd /home/sdznkj/zkx_ws/sendlerobot
    PYTHONPATH=. python3 scripts/l1_rtc_smolvla_check.py \\
        --policy_path outputs/test/060000/pretrained_model \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _build_batch(policy, device: torch.device, seed: int) -> dict:
    """Synthetic observation matching policy feature shapes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    state_dim = policy.config.robot_state_feature.shape[0]
    # Smaller than native cam is fine; prepare_images resizes/pads to 512x512.
    img = torch.rand(1, 3, 240, 320, generator=g)
    state = torch.randn(1, state_dim, generator=g)
    batch = {
        "observation.state": state.to(device),
        "task": "Grasp a lego block and put it in the bin.",
    }
    for key in policy.config.image_features:
        batch[key] = img.clone().to(device)
    return batch


def _prefix_mae(pred: torch.Tensor, target: torch.Tensor, horizon: int) -> float:
    h = min(horizon, pred.shape[1], target.shape[1])
    # pred/target: (B, T, A)
    err = (pred[:, :h] - target[:, :h]).abs().mean().item()
    return float(err)


def _set_rtc(policy, *, enabled: bool, execution_horizon: int, max_guidance_weight: float) -> None:
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


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="L1 RTC offline check for local SmolVLA")
    parser.add_argument(
        "--policy_path",
        type=str,
        default="/home/sdznkj/zkx_ws/sendlerobot/outputs/test/060000/pretrained_model",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference_delay", type=int, default=4)
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument("--max_guidance_weight", type=float, default=10.0)
    parser.add_argument(
        "--min_improve_ratio",
        type=float,
        default=0.05,
        help="Require RTC prefix MAE to be at least this fraction lower than no-RTC.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy_path = Path(args.policy_path)
    if not policy_path.exists():
        print(f"FAIL: policy path not found: {policy_path}", file=sys.stderr)
        return 2

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"Loading policy from {policy_path} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(str(policy_path))
    policy.to(device)
    policy.eval()

    chunk = policy.config.chunk_size
    action_dim = policy.config.action_feature.shape[0]
    noise_shape = (1, chunk, policy.config.max_action_dim)

    # Reference chunk (different obs) → leftover target for guidance
    _set_rtc(policy, enabled=False, execution_horizon=args.execution_horizon, max_guidance_weight=args.max_guidance_weight)
    batch_prev = _build_batch(policy, device, seed=args.seed)
    noise_prev = torch.randn(noise_shape, device=device)
    prev_chunk = policy.predict_action_chunk(batch_prev, noise=noise_prev)
    print(f"prev_chunk shape={tuple(prev_chunk.shape)}")

    # New prediction on a different obs, same noise for RTC vs no-RTC
    batch_new = _build_batch(policy, device, seed=args.seed + 1)
    noise_new = torch.randn(noise_shape, device=device)
    torch.manual_seed(args.seed + 99)  # isolate any nondeterminism around calls

    _set_rtc(policy, enabled=False, execution_horizon=args.execution_horizon, max_guidance_weight=args.max_guidance_weight)
    no_rtc = policy.predict_action_chunk(batch_new, noise=noise_new.clone())

    _set_rtc(policy, enabled=True, execution_horizon=args.execution_horizon, max_guidance_weight=args.max_guidance_weight)
    rtc = policy.predict_action_chunk(
        batch_new,
        noise=noise_new.clone(),
        prev_chunk_left_over=prev_chunk,
        inference_delay=args.inference_delay,
        execution_horizon=args.execution_horizon,
    )

    # Align leftover length; RTC pads internally, we compare on action dims only
    leftover = prev_chunk[:, :, :action_dim]
    no_rtc_a = no_rtc[:, :, :action_dim]
    rtc_a = rtc[:, :, :action_dim]

    h = args.execution_horizon
    mae_no = _prefix_mae(no_rtc_a, leftover, h)
    mae_rtc = _prefix_mae(rtc_a, leftover, h)
    mae_full_no = _prefix_mae(no_rtc_a, leftover, chunk)
    mae_full_rtc = _prefix_mae(rtc_a, leftover, chunk)

    # First-chunk sanity: no leftover → RTC path should still run
    rtc_first = policy.predict_action_chunk(
        batch_new,
        noise=noise_new.clone(),
        prev_chunk_left_over=None,
        inference_delay=args.inference_delay,
        execution_horizon=args.execution_horizon,
    )
    first_diff = (rtc_first - no_rtc).abs().mean().item()

    improve = (mae_no - mae_rtc) / max(mae_no, 1e-8)
    passed = mae_rtc < mae_no * (1.0 - args.min_improve_ratio)

    report = {
        "policy_path": str(policy_path),
        "device": str(device),
        "inference_delay": args.inference_delay,
        "execution_horizon": args.execution_horizon,
        "prefix_mae_no_rtc": round(mae_no, 6),
        "prefix_mae_rtc": round(mae_rtc, 6),
        "prefix_improve_ratio": round(improve, 4),
        "full_chunk_mae_no_rtc": round(mae_full_no, 6),
        "full_chunk_mae_rtc": round(mae_full_rtc, 6),
        "first_chunk_vs_nortc_mae": round(first_diff, 6),
        "min_improve_ratio_required": args.min_improve_ratio,
        "pass": passed,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if passed:
        print(
            f"PASS: RTC prefix MAE {mae_rtc:.4f} < no-RTC {mae_no:.4f} "
            f"(improve {improve:.1%} ≥ {args.min_improve_ratio:.0%})"
        )
        return 0

    print(
        f"FAIL: RTC did not sufficiently pull prefix toward leftover "
        f"(rtc={mae_rtc:.4f}, no_rtc={mae_no:.4f}, improve={improve:.1%})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
