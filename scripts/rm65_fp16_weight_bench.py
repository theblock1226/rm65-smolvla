#!/usr/bin/env python3
"""Cast a backup of the stacking checkpoint to FP16 and time it.

The original directory is only read. FP16 weights are written under outputs/diag.
Normalization stays FP32. num_steps is changed only on the loaded config.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
BACKUP = ROOT / "outputs/diag/smolvla_rm65_stack_30ep_s1000_backup"
FP16_DIR = ROOT / "outputs/diag/smolvla_rm65_stack_30ep_s1000_fp16"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"
OUT = ROOT / "outputs/diag/fp16_weight_bench.json"
FRAMES = (0, 160, 340, 560)


def sync(torch) -> None:
    torch.cuda.synchronize()


def convert(torch) -> dict:
    from safetensors.torch import load_file, save_file

    if not (BACKUP / "model.safetensors").is_file():
        raise SystemExit(f"missing backup {BACKUP}")
    src_size = (CKPT / "model.safetensors").stat().st_size
    bak_size = (BACKUP / "model.safetensors").stat().st_size
    if src_size != bak_size:
        raise SystemExit(f"backup size {bak_size} != original {src_size}")

    FP16_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "config.hub.json",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_6_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "README.md",
    ):
        src = BACKUP / name
        if src.is_file():
            shutil.copy2(src, FP16_DIR / name)
    tok = FP16_DIR / "tokenizer"
    if not tok.exists():
        shutil.copytree(BACKUP / "tokenizer", tok)

    tensors = load_file(str(BACKUP / "model.safetensors"))
    out = {}
    n_fp32 = n_other = 0
    for key, value in tensors.items():
        if value.is_floating_point() and value.dtype == torch.float32:
            out[key] = value.half()
            n_fp32 += 1
        else:
            out[key] = value
            n_other += 1
    dest = FP16_DIR / "model.safetensors"
    save_file(out, str(dest))
    return {
        "fp32_tensors": n_fp32,
        "other_tensors": n_other,
        "fp16_bytes": dest.stat().st_size,
        "fp32_bytes": bak_size,
    }


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
    return actions.detach().float().cpu().numpy()[0], (time.perf_counter() - t0) * 1000


def joint_stats(a: np.ndarray, b: np.ndarray, n_exec: int) -> dict:
    d = np.abs(a[:n_exec, :6] - b[:n_exec, :6])
    return {
        "joint_mae_deg": float(d.mean()),
        "joint_max_deg": float(d.max()),
        "gripper_mismatch": int(np.sum(np.abs(a[:n_exec, 6] - b[:n_exec, 6]) > 0.5)),
        "finite": bool(np.isfinite(b).all()),
    }


def time_mode(policy, torch, batch, noise, name, steps, fp16) -> dict:
    forward(policy, torch, batch, steps, fp16, noise)
    if fp16:
        forward(policy, torch, batch, steps, fp16, noise)
    samples = []
    for i in range(5):
        _, ms = forward(policy, torch, batch, steps, fp16, noise)
        samples.append(ms)
        print(f"{name} iter {i + 1} {ms:.1f} ms", flush=True)
    return {"median_ms": float(np.median(samples)), "min_ms": float(min(samples)), "max_ms": float(max(samples))}


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    import torch
    disk_before = json.loads((CKPT / "config.json").read_text())["num_steps"]
    meta = convert(torch)
    print(
        f"converted {meta['fp32_tensors']} fp32 tensors, file {meta['fp32_bytes']} -> {meta['fp16_bytes']}",
        flush=True,
    )

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.scripts.server.rm65_ckpt import load_dataset_stats_from_ckpt, trim_stats_to_config

    config = PreTrainedConfig.from_pretrained(str(CKPT))
    stats = trim_stats_to_config(load_dataset_stats_from_ckpt(CKPT), config)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats or None)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("no cuda")
    policy.to(device)
    policy.eval()

    info = json.loads((DS / "meta/info.json").read_text())
    states = load_states()
    task = "Stack three cylinder blocks. Grasp the left cylinder."
    frames = {
        i: (load_frame(info, "observation.images.front", i), load_frame(info, "observation.images.astra", i))
        for i in FRAMES
    }
    batch0 = make_batch(torch, device, *frames[0], states[0], task)
    noise0 = policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), device)

    timing = {
        "fp32_10": time_mode(policy, torch, batch0, noise0, "fp32_10", 10, False),
        "fp32_4": time_mode(policy, torch, batch0, noise0, "fp32_4", 4, False),
    }

    from lerobot.common.policies.smolvla.modeling_smolvla import load_smolvla

    policy.model.half()
    load_smolvla(
        policy,
        FP16_DIR / "model.safetensors",
        device=str(device),
        checkpoint_keys_mapping="model._orig_mod.//model.",
    )
    weight_dtypes = sorted({str(p.dtype) for p in policy.model.parameters()})
    print(f"fp16 weight dtypes {weight_dtypes}", flush=True)
    timing["fp16_weight_4"] = time_mode(policy, torch, batch0, noise0, "fp16_weight_4", 4, True)
    timing["fp16_weight_10"] = time_mode(policy, torch, batch0, noise0, "fp16_weight_10", 10, True)

    # Reload a fresh FP32 policy for the action reference. The live module is now FP16.
    policy_fp32 = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats or None)
    policy_fp32.to(device)
    policy_fp32.eval()
    n_exec = int(policy_fp32.config.n_action_steps)
    quality = []
    for index in FRAMES:
        batch = make_batch(torch, device, *frames[index], states[index], task)
        torch.manual_seed(1000 + index)
        noise = policy_fp32.model.sample_noise(
            (1, policy_fp32.config.chunk_size, policy_fp32.config.max_action_dim), device
        )
        ref, _ = forward(policy_fp32, torch, batch, 10, False, noise)
        fp16_out, _ = forward(policy, torch, batch, 4, True, noise)
        row = {"frame": index, "fp16_weight_4": joint_stats(ref, fp16_out, n_exec)}
        quality.append(row)
        print(
            f"frame {index} fp16-weight 4-step vs fp32 10-step mae {row['fp16_weight_4']['joint_mae_deg']:.3f} "
            f"max {row['fp16_weight_4']['joint_max_deg']:.3f} finite {row['fp16_weight_4']['finite']}",
            flush=True,
        )

    policy_fp32.config.num_steps = 10
    policy.config.num_steps = 10
    disk_after = json.loads((CKPT / "config.json").read_text())["num_steps"]
    report = {
        "original_num_steps_before": disk_before,
        "original_num_steps_after": disk_after,
        "original_bytes": (CKPT / "model.safetensors").stat().st_size,
        "backup": str(BACKUP),
        "fp16_dir": str(FP16_DIR),
        "conversion": meta,
        "weight_dtypes": weight_dtypes,
        "timing_ms": timing,
        "vs_fp32_10": quality,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if disk_after != 10 or (CKPT / "model.safetensors").stat().st_size != meta["fp32_bytes"]:
        raise SystemExit("original checkpoint changed")


if __name__ == "__main__":
    main()
