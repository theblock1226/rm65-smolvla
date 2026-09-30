#!/usr/bin/env python3
"""Try torch.compile on the current stacking checkpoint. Does not write weights.

Inductor needs Triton, which this container does not have. The runnable
torch.compile backend here is cudagraphs, applied to one denoise step.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"
OUT = ROOT / "outputs/diag/torch_compile_bench.json"


def sync(torch) -> None:
    torch.cuda.synchronize()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    import torch

    from lerobot.common.constants import ACTION
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.scripts.server.rm65_ckpt import load_dataset_stats_from_ckpt, trim_stats_to_config

    disk_steps = json.loads((CKPT / "config.json").read_text())["num_steps"]
    config = PreTrainedConfig.from_pretrained(str(CKPT))
    stats = trim_stats_to_config(load_dataset_stats_from_ckpt(CKPT), config)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats or None)
    device = torch.device("cuda")
    policy.to(device)
    policy.eval()

    import av

    info = json.loads((DS / "meta/info.json").read_text())

    def grab(key: str) -> np.ndarray:
        path = DS / info["video_path"].format(episode_chunk=0, episode_index=0, video_key=key)
        container = av.open(str(path))
        frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
        container.close()
        return frame

    import pyarrow.parquet as pq

    state = np.stack(
        pq.read_table(DS / "data/chunk-000/episode_000000.parquet", columns=["observation.state"])
        .column("observation.state")
        .to_pylist()
    ).astype(np.float32)[0]
    front, astra = grab("observation.images.front"), grab("observation.images.astra")
    batch = {
        "task": "Stack three cylinder blocks. Grasp the left cylinder.",
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.images.camera1": torch.from_numpy(np.ascontiguousarray(front)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
        "observation.images.camera2": torch.from_numpy(np.ascontiguousarray(astra)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
    }
    noise = policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), device)

    def once():
        policy.config.num_steps = 4
        policy.reset()
        raw_state = batch["observation.state"].detach().clone()
        prepared = policy._prepare_batch(batch)
        images, img_masks = policy.prepare_images(prepared)
        state_t = policy.prepare_state(prepared)
        lang_tokens, lang_masks = policy.prepare_language(prepared)
        sync(torch)
        t0 = time.perf_counter()
        actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state_t, noise=noise)
        actions = actions.float()[:, :, : policy.config.action_feature.shape[0]]
        actions = policy.unnormalize_outputs({ACTION: actions})[ACTION]
        actions = policy._postprocess_action(actions, raw_state)
        sync(torch)
        return actions.detach().float().cpu().numpy()[0], (time.perf_counter() - t0) * 1000

    print("eager warmup", flush=True)
    once()
    eager_ms = []
    for i in range(5):
        _, ms = once()
        eager_ms.append(ms)
        print(f"eager_4 iter {i + 1} {ms:.1f} ms", flush=True)
    eager_ref, _ = once()

    model = policy.model
    raw_denoise = model.denoise_step
    report = {"original_num_steps": disk_steps, "eager_4_median_ms": float(np.median(eager_ms))}

    try:
        model.denoise_step = torch.compile(raw_denoise, backend="inductor")
        t0 = time.perf_counter()
        once()
        report["inductor"] = {"ok": True, "first_call_s": time.perf_counter() - t0}
    except Exception as e:
        report["inductor"] = {"ok": False, "error": f"{type(e).__name__}: {e}".splitlines()[0][:400]}
        print("inductor", report["inductor"]["error"], flush=True)
    model.denoise_step = raw_denoise

    model.denoise_step = torch.compile(raw_denoise, backend="cudagraphs")
    print("cudagraphs warmup", flush=True)
    t0 = time.perf_counter()
    try:
        once()
        first_s = time.perf_counter() - t0
        print(f"cudagraphs first call {first_s:.1f} s", flush=True)
        compiled_ms = []
        for i in range(5):
            _, ms = once()
            compiled_ms.append(ms)
            print(f"cudagraphs_4 iter {i + 1} {ms:.1f} ms", flush=True)
        compiled_out, _ = once()
        n_exec = int(policy.config.n_action_steps)
        diff = np.abs(eager_ref[:n_exec, :6] - compiled_out[:n_exec, :6])
        report["cudagraphs"] = {
            "ok": True,
            "first_call_s": first_s,
            "median_ms": float(np.median(compiled_ms)),
            "min_ms": float(min(compiled_ms)),
            "max_ms": float(max(compiled_ms)),
            "joint_mae_vs_eager_deg": float(diff.mean()),
            "joint_max_vs_eager_deg": float(diff.max()),
            "finite": bool(np.isfinite(compiled_out).all()),
        }
    except Exception as e:
        report["cudagraphs"] = {"ok": False, "error": f"{type(e).__name__}: {e}".splitlines()[0][:500]}
        print("cudagraphs", report["cudagraphs"]["error"], flush=True)
    finally:
        model.denoise_step = raw_denoise
        policy.config.num_steps = 10

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    after = json.loads((CKPT / "config.json").read_text())["num_steps"]
    if after != disk_steps:
        raise SystemExit("config.json changed")


if __name__ == "__main__":
    main()
