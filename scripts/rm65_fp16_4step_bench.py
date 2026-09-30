#!/usr/bin/env python3
"""Time 4-step FP16 SmolVLA against the on-disk 10-step FP32 sampler.

Does not write the checkpoint. ``num_steps`` is changed only on the loaded
config object. FP16 autocast wraps the sampler; state normalization and
unnormalization stay in FP32.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"
OUT = ROOT / "outputs/diag/fp16_4step_bench.json"
FRAMES = (0, 160, 340, 560)


def sync(torch) -> None:
    torch.cuda.synchronize()


def load_frame(info: dict, key: str, index: int) -> np.ndarray:
    import av

    path = DS / info["video_path"].format(episode_chunk=0, episode_index=0, video_key=key)
    container = av.open(str(path))
    frame = None
    for i, fr in enumerate(container.decode(video=0)):
        if i == index:
            frame = fr.to_ndarray(format="rgb24")
            break
    container.close()
    if frame is None:
        raise RuntimeError(f"{key} frame {index}")
    return frame


def load_states() -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(DS / "data/chunk-000/episode_000000.parquet", columns=["observation.state"])
    return np.stack(table.column("observation.state").to_pylist()).astype(np.float32)


def make_batch(torch, device, front, astra, state, task: str) -> dict:
    return {
        "task": task,
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.images.camera1": torch.from_numpy(np.ascontiguousarray(front))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
        / 255.0,
        "observation.images.camera2": torch.from_numpy(np.ascontiguousarray(astra))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
        / 255.0,
    }


def forward(policy, torch, batch, n_steps: int, fp16: bool, noise):
    from lerobot.common.constants import ACTION

    policy.config.num_steps = int(n_steps)
    policy.reset()
    raw_state = batch["observation.state"].detach().clone()
    prepared = policy._prepare_batch(batch)
    images, img_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    lang_tokens, lang_masks = policy.prepare_language(prepared)
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if fp16 else nullcontext()
    sync(torch)
    t0 = time.perf_counter()
    with ctx:
        actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
    actions = actions.float()
    dim = policy.config.action_feature.shape[0]
    actions = actions[:, :, :dim]
    actions = policy.unnormalize_outputs({ACTION: actions})[ACTION]
    actions = policy._postprocess_action(actions, raw_state)
    sync(torch)
    elapsed = time.perf_counter() - t0
    return actions.detach().float().cpu().numpy()[0], elapsed


def joint_stats(a: np.ndarray, b: np.ndarray, n_exec: int) -> dict:
    d = np.abs(a[:n_exec, :6] - b[:n_exec, :6])
    grip_a = a[:n_exec, 6]
    grip_b = b[:n_exec, 6]
    return {
        "joint_mae_deg": float(d.mean()),
        "joint_max_deg": float(d.max()),
        "gripper_mismatch": int(np.sum(np.abs(grip_a - grip_b) > 0.5)),
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
    }


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    import torch

    disk = json.loads((CKPT / "config.json").read_text())
    if int(disk["num_steps"]) != 10:
        raise SystemExit(f"config.json num_steps is {disk['num_steps']}, expected 10")

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.scripts.server.rm65_ckpt import load_dataset_stats_from_ckpt, trim_stats_to_config

    config = PreTrainedConfig.from_pretrained(str(CKPT))
    stats = trim_stats_to_config(load_dataset_stats_from_ckpt(CKPT), config)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats or None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no cuda")
    policy.to(device)
    policy.eval()

    info = json.loads((DS / "meta/info.json").read_text())
    states = load_states()
    task = "Stack three cylinder blocks. Grasp the left cylinder."
    n_exec = int(policy.config.n_action_steps)
    modes = (("fp32_10", 10, False), ("fp32_4", 4, False), ("fp16_4", 4, True))

    cache = {i: (load_frame(info, "observation.images.front", i), load_frame(info, "observation.images.astra", i)) for i in FRAMES}
    front0, astra0 = cache[0]
    batch0 = make_batch(torch, device, front0, astra0, states[0], task)
    noise0 = policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), device)

    print("warmup", flush=True)
    for name, steps, fp16 in modes:
        forward(policy, torch, batch0, steps, fp16, noise0)
        if fp16:
            forward(policy, torch, batch0, steps, fp16, noise0)
        print(f"  warmed {name}", flush=True)

    timing = {}
    for name, steps, fp16 in modes:
        samples = []
        for i in range(5):
            _, elapsed = forward(policy, torch, batch0, steps, fp16, noise0)
            samples.append(elapsed * 1000)
            print(f"{name} iter {i + 1} {samples[-1]:.1f} ms", flush=True)
        timing[name] = {
            "median_ms": float(np.median(samples)),
            "min_ms": float(np.min(samples)),
            "max_ms": float(np.max(samples)),
        }

    quality = []
    for index in FRAMES:
        front, astra = cache[index]
        batch = make_batch(torch, device, front, astra, states[index], task)
        torch.manual_seed(1000 + index)
        noise = policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), device)
        outs = {}
        for name, steps, fp16 in modes:
            outs[name], _ = forward(policy, torch, batch, steps, fp16, noise)
        row = {"frame": index}
        for name in ("fp32_4", "fp16_4"):
            row[name] = joint_stats(outs["fp32_10"], outs[name], n_exec)
        quality.append(row)
        print(
            f"frame {index}  4-step fp32 mae {row['fp32_4']['joint_mae_deg']:.3f} max {row['fp32_4']['joint_max_deg']:.3f}"
            f"  |  4-step fp16 mae {row['fp16_4']['joint_mae_deg']:.3f} max {row['fp16_4']['joint_max_deg']:.3f}"
            f"  finite {row['fp16_4']['finite']}",
            flush=True,
        )

    policy.config.num_steps = 10
    disk_after = json.loads((CKPT / "config.json").read_text())
    report = {
        "checkpoint_num_steps_on_disk": int(disk_after["num_steps"]),
        "timing_ms": timing,
        "vs_fp32_10": quality,
        "n_action_steps_compared": n_exec,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if int(disk_after["num_steps"]) != 10:
        raise SystemExit("config.json was modified")


if __name__ == "__main__":
    main()
