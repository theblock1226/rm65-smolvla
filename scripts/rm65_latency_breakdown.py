#!/usr/bin/env python3
"""Break one SmolVLA forward into stages. No robot."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"


def sync(torch):
    torch.cuda.synchronize()


def main() -> None:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    import torch

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

    import av
    import json

    info = json.loads((DS / "meta/info.json").read_text())

    def grab(key: str, index: int) -> np.ndarray:
        path = DS / info["video_path"].format(episode_chunk=0, episode_index=0, video_key=key)
        container = av.open(str(path))
        frame = None
        for i, fr in enumerate(container.decode(video=0)):
            if i == index:
                frame = fr.to_ndarray(format="rgb24")
                break
        container.close()
        if frame is None:
            raise RuntimeError(key)
        return frame

    front = grab("observation.images.front", 0)
    astra = grab("observation.images.astra", 0)
    state = np.zeros(7, np.float32)
    task = "Stack three cylinder blocks. Grasp the left cylinder."

    def h2d():
        batch = {
            "task": task,
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
            "observation.images.camera1": torch.from_numpy(np.ascontiguousarray(front)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
            "observation.images.camera2": torch.from_numpy(np.ascontiguousarray(astra)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
        }
        return batch

    names = [
        "host_to_gpu",
        "normalize_state",
        "resize_and_siglip",
        "tokenize",
        "embed_prefix",
        "vlm_kv_cache",
        "denoise_10",
        "unnorm_q_plus_dq",
        "copy_to_cpu",
    ]
    rows: list[list[float]] = []

    def once(record: bool) -> None:
        sync(torch)
        t = time.perf_counter()
        marks = [t]
        batch = h2d()
        sync(torch)
        marks.append(time.perf_counter())
        raw_state = batch["observation.state"].detach().clone()
        batch = policy._prepare_batch(batch)
        sync(torch)
        marks.append(time.perf_counter())
        images, img_masks = policy.prepare_images(batch)
        sync(torch)
        marks.append(time.perf_counter())
        state_t = policy.prepare_state(batch)
        lang_tokens, lang_masks = policy.prepare_language(batch)
        sync(torch)
        marks.append(time.perf_counter())
        model = policy.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state_t
        )
        sync(torch)
        marks.append(time.perf_counter())
        from lerobot.common.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        sync(torch)
        marks.append(time.perf_counter())
        noise = model.sample_noise((1, config.chunk_size, config.max_action_dim), device)
        x_t = noise
        dt = -1.0 / config.num_steps
        denoise_marks = []
        for step in range(config.num_steps):
            ts = time.perf_counter()
            time_tensor = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=device).expand(1)
            v_t = model.denoise_step(prefix_pad_masks, past_key_values, x_t, time_tensor)
            x_t = x_t + dt * v_t
            sync(torch)
            denoise_marks.append(time.perf_counter() - ts)
        marks.append(time.perf_counter())
        actions = x_t[:, :, : config.action_feature.shape[0]]
        actions = policy.unnormalize_outputs({"action": actions})["action"]
        actions = policy._postprocess_action(actions, raw_state)
        sync(torch)
        marks.append(time.perf_counter())
        _ = actions.detach().float().cpu().numpy()
        sync(torch)
        marks.append(time.perf_counter())
        if record:
            dts = [(marks[i + 1] - marks[i]) * 1000 for i in range(len(marks) - 1)]
            # replace the lumped denoise bucket with the sum; per-step stored separately
            dts.append(sum(denoise_marks) * 1000)  # noqa: keep alignment check
            rows.append(dts[: len(names)] + [d * 1000 for d in denoise_marks])
        if record and not rows:
            pass
        return denoise_marks if record else None

    print(f"warmup  resize={config.resize_imgs_with_padding}  layers={config.num_vlm_layers}  denoise={config.num_steps}", flush=True)
    once(False)
    per_step: list[list[float]] = []
    for i in range(5):
        sync(torch)
        t0 = time.perf_counter()
        # re-call to capture denoise steps: once(True) appends
        marks_start = time.perf_counter()
        once(True)
        print(f"iter {i+1} recorded", flush=True)
        _ = marks_start

    arr = np.asarray([r[: len(names)] for r in rows], dtype=np.float64)
    med = np.median(arr, axis=0)
    total = float(med.sum())
    print()
    print(f"{'stage':<22} {'median_ms':>10} {'share':>8}")
    for name, ms in zip(names, med):
        print(f"{name:<22} {ms:10.1f} {100 * ms / total:7.1f}%")
    print(f"{'TOTAL':<22} {total:10.1f}")
    steps = np.asarray([r[len(names) :] for r in rows], dtype=np.float64)
    step_med = np.median(steps, axis=0)
    print(f"denoise each step ms: {' '.join(f'{v:.1f}' for v in step_med)}  sum {step_med.sum():.1f}")
    print(f"prefix tokens measured via embed; image in {front.shape}")


if __name__ == "__main__":
    main()
