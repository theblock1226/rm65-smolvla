#!/usr/bin/env python3
"""Time 4-step guided RTC on the stack checkpoint and check the prefix.

Does not rewrite config.json. Prints whether the measured guided latency
still fits inside chunk_size at 20 Hz.
"""

from __future__ import annotations

import argparse
import time

import torch

CKPT = "/home/sdznkj/zkx_ws/sendlerobot/smolvla_rm65_stack_30ep_s1000"


def _batch(policy, device: torch.device) -> dict:
    g = torch.Generator(device="cpu").manual_seed(0)
    state_dim = int(policy.config.robot_state_feature.shape[0])
    img = torch.rand(1, 3, 240, 320, generator=g)
    state = torch.randn(1, state_dim, generator=g)
    batch = {"observation.state": state.to(device), "task": "Stack three cylinder blocks."}
    for key in policy.config.image_features:
        batch[key] = img.clone().to(device)
    return batch


def _ms(fn) -> tuple[float, torch.Tensor]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    from lerobot.scripts.server.rm65_ckpt import load_smolvla_for_server
    from lerobot.scripts.server.rm65_rtc import enable_smolvla_rtc

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = load_smolvla_for_server(CKPT, str(device), sanitize=False)
    enable_smolvla_rtc(policy, num_steps=4, execution_horizon=13, ckpt_dir=CKPT)
    policy.eval()

    batch = _batch(policy, device)
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device=device)

    _ms(lambda: policy.predict_action_chunk(batch, noise=noise.clone()))
    plain_ms, plain = _ms(lambda: policy.predict_action_chunk(batch, noise=noise.clone()))

    leftover = plain.detach()
    if leftover.ndim == 3:
        leftover = leftover[0]
    # Drop the steps a 20 Hz loop would already have executed.
    delay = 8
    leftover = leftover[delay:].cpu()
    guided_ms, guided = _ms(
        lambda: policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=leftover,
            inference_delay=delay,
            execution_horizon=13,
        )
    )
    guided = guided[0] if guided.ndim == 3 else guided
    again_ms, again = _ms(lambda: policy.predict_action_chunk(batch, noise=noise.clone()))
    again = again[0] if again.ndim == 3 else again
    ref = leftover[:5].to(guided.device)
    width = ref.shape[-1]
    rtc_mae = (guided[:5, :width] - ref).abs().mean().item()
    base_mae = (again[:5, :width] - ref).abs().mean().item()

    dt_ms = 1000.0 / args.fps
    plain_steps = plain_ms / dt_ms
    guided_steps = guided_ms / dt_ms
    chunk = int(policy.config.chunk_size)
    print(f"relative={policy.config.use_relative_actions} num_steps={policy.config.num_steps} chunk={chunk}")
    print(f"plain_ms={plain_ms:.0f}  (~{plain_steps:.1f} steps @ {args.fps:.0f}Hz)")
    print(f"guided_ms={guided_ms:.0f}  (~{guided_steps:.1f} steps @ {args.fps:.0f}Hz)  repeat_plain_ms={again_ms:.0f}")
    print(f"prefix_mae_first5 rtc={rtc_mae:.4f} unguided={base_mae:.4f} leftover_T={tuple(leftover.shape)}")
    fits = guided_steps < chunk - 2
    print(f"fits_in_chunk={fits} prefix_improved={rtc_mae < base_mae}")
    return 0 if fits and rtc_mae < base_mae else 1


if __name__ == "__main__":
    raise SystemExit(main())
