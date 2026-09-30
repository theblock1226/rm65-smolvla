#!/usr/bin/env python3
"""Offline SmolVLA diagnostics: attention heatmaps + physical action chunk.

Does not touch the 20 Hz robot loop. Standard HF attention hooks do not fire
because SmolVLA computes softmax in ``eager_attention_forward``.

Example::

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/smolvla_diag_visualize.py'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL_VLM = ROOT / "base_models" / "SmolVLM2-500M-Video-Instruct"
COMPAT_DROP_KEYS = {
    "use_peft",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
    "pretrained_revision",
    "compile_model",
    "compile_mode",
    "use_relative_actions",
    "relative_exclude_joints",
    "action_feature_names",
    "use_dataset_features",
}
JOINT_NAMES = [f"j{i}" for i in range(1, 7)] + ["grip"]


def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA attention + action diagnostics")
    p.add_argument("--ckpt", type=str, default=str(ROOT / "astra_smolvla"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cam1", type=str, default=str(ROOT / "logs/ood_live_probe/camera1_front.jpg"))
    p.add_argument("--cam2", type=str, default=str(ROOT / "logs/ood_live_probe/camera2_astra.jpg"))
    p.add_argument("--task", type=str, default="Stack three cylinder blocks. Grasp the left cylinder.")
    p.add_argument("--out", type=str, default=str(ROOT / "outputs/diag/smolvla_attn"))
    p.add_argument("--layers", type=str, default="0,7,15")
    p.add_argument("--max-step-deg", type=float, default=0.4)
    p.add_argument("--cell", type=int, default=256)
    p.add_argument(
        "--episode",
        type=str,
        default="",
        help="DAgger episode dir with images/ and trajectory.npz",
    )
    p.add_argument("--frame", type=int, default=200, help="Frame index inside --episode")
    p.add_argument(
        "--state",
        type=str,
        default="",
        help="Comma-separated state; default zeros unless --episode",
    )
    return p.parse_args()


def load_episode_frame(ep_dir: Path, frame: int) -> dict:
    ep_dir = Path(ep_dir)
    traj = np.load(ep_dir / "trajectory.npz")
    n = int(traj["states"].shape[0])
    if frame < 0:
        frame = n + frame
    if not (0 <= frame < n):
        raise IndexError(f"frame {frame} out of range 0..{n-1} in {ep_dir}")
    cam1 = ep_dir / "images" / "camera1" / f"frame_{frame:06d}.ppm"
    cam2 = ep_dir / "images" / "camera2" / f"frame_{frame:06d}.ppm"
    if not cam1.is_file() or not cam2.is_file():
        raise FileNotFoundError(f"missing ppm for frame {frame}: {cam1} {cam2}")
    meta = {}
    meta_path = ep_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
    return {
        "cam1": cam1,
        "cam2": cam2,
        "state": np.asarray(traj["states"][frame], dtype=np.float32),
        "action": np.asarray(traj["actions"][frame], dtype=np.float32),
        "policy_action": np.asarray(traj["policy_actions"][frame], dtype=np.float32),
        "intervention": bool(traj["intervention"][frame]),
        "n_frames": n,
        "frame": frame,
        "task": meta.get("task") or "",
    }


def parse_state_arg(raw: str, need: int) -> np.ndarray | None:
    if not raw.strip():
        return None
    vals = [float(x) for x in raw.split(",") if x.strip()]
    arr = np.asarray(vals, dtype=np.float32)
    if arr.shape[0] < need:
        arr = np.pad(arr, (0, need - arr.shape[0]))
    return arr[:need]


def force_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def sanitize_ckpt_config(ckpt_dir: Path) -> None:
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    changed = False
    for k in list(cfg):
        if k in COMPAT_DROP_KEYS:
            cfg.pop(k, None)
            changed = True
    if LOCAL_VLM.is_dir() and cfg.get("vlm_model_name") != str(LOCAL_VLM):
        cfg["vlm_model_name"] = str(LOCAL_VLM)
        changed = True
    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")


def load_stats(ckpt_dir: Path, config) -> dict:
    from safetensors.torch import load_file

    candidates: list[Path] = []
    pre = ckpt_dir / "policy_preprocessor.json"
    if pre.is_file():
        meta = json.loads(pre.read_text())
        for step in meta.get("steps", []):
            sf = step.get("state_file")
            if sf and "normalizer" in str(sf):
                candidates.append(ckpt_dir / sf)
    candidates.append(ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    candidates.append(ckpt_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no normalizer stats under {ckpt_dir}")
    print("normalizer stats:", path.name)
    raw = load_file(str(path))
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        if "." not in key:
            continue
        feat, stat = key.rsplit(".", 1)
        if stat not in ("mean", "std", "min", "max"):
            continue
        stats.setdefault(feat, {})[stat] = tensor
    out = {}
    for key, ft in {**config.input_features, **config.output_features}.items():
        if key not in stats:
            continue
        need = int(ft.shape[0]) if len(ft.shape) >= 1 else None
        entry = {}
        for sk, val in stats[key].items():
            t = val.clone()
            if need is not None and t.ndim >= 1 and t.shape[0] != need and t.shape[0] > need:
                t = t[:need]
            entry[sk] = t
        out[key] = entry
    return out


def load_rgb_tensor(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def tensor_to_u8(img: torch.Tensor) -> np.ndarray:
    """Model input CHW in [-1,1] or [0,1] → HxWx3 uint8."""
    x = img.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if float(x.min()) < -0.01:
        x = (x + 1.0) * 0.5
    x = x.clamp(0, 1).numpy()
    return (x * 255.0).astype(np.uint8)


def scores_to_grid(scores: np.ndarray) -> np.ndarray:
    n = int(scores.reshape(-1).shape[0])
    side = int(round(math.sqrt(n)))
    if side * side == n:
        return scores.reshape(side, side)
    h = max(1, int(math.floor(math.sqrt(n))))
    w = int(math.ceil(n / h))
    pad = np.zeros(h * w, dtype=np.float32)
    pad[:n] = scores.reshape(-1)
    return pad.reshape(h, w)


def overlay_heatmap(rgb: np.ndarray, scores: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    grid = scores_to_grid(scores.astype(np.float32))
    g = grid - float(grid.min())
    g = g / (float(g.max()) + 1e-6)
    heat = cv2.applyColorMap((g * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = cv2.addWeighted(rgb, 1.0 - alpha, heat, alpha, 0)
    return out


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text[:48], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_pad(imgs: list[np.ndarray], pad: int = 4) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    parts = []
    for im in imgs:
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        parts.append(im)
        parts.append(np.full((h, pad, 3), 30, dtype=np.uint8))
    return np.concatenate(parts[:-1], axis=1)


def vstack_pad(imgs: list[np.ndarray], pad: int = 4) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    parts = []
    for im in imgs:
        if im.shape[1] != w:
            im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
        parts.append(im)
        parts.append(np.full((pad, w, 3), 30, dtype=np.uint8))
    return np.concatenate(parts[:-1], axis=0)


def resize_cell(img: np.ndarray, cell: int) -> np.ndarray:
    return cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)


def mean_heads(probs: torch.Tensor, q_sl: slice, k_sl: slice) -> np.ndarray:
    # probs: [B,H,Q,K]
    x = probs[0, :, q_sl, k_sl].float().mean(dim=0)
    return x.numpy()


def prefix_slices(policy, images, lang_tokens) -> dict[str, tuple[int, int]]:
    slices = {}
    pos = 0
    expert = policy.model.vlm_with_expert
    for i, img in enumerate(images):
        n = int(expert.embed_image(img).shape[1])
        slices[f"image{i}"] = (pos, pos + n)
        pos += n
    n_lang = int(lang_tokens.shape[1])
    slices["lang"] = (pos, pos + n_lang)
    pos += n_lang
    slices["state"] = (pos, pos + 1)
    pos += 1
    slices["prefix_len"] = (0, pos)
    return slices


def content_token_indices(tokenizer, ids: list[int], max_n: int = 6) -> list[tuple[int, str]]:
    special = set(tokenizer.all_special_ids)
    out = []
    for i, tid in enumerate(ids):
        if tid in special or tid == tokenizer.pad_token_id:
            continue
        piece = tokenizer.decode([tid], skip_special_tokens=False).strip()
        if not piece or piece in ("\n", "▁", " "):
            continue
        # skip pure punctuation
        if all(not ch.isalnum() for ch in piece.replace("▁", "")):
            continue
        out.append((i, piece.replace("▁", "")))
        if len(out) >= max_n:
            break
    return out


def pick_record(records: list[dict], layer: int, phase: str) -> dict | None:
    for rec in records:
        if rec["layer_idx"] == layer and rec["phase"] == phase:
            return rec
    return None


def draw_joint_plot(actions: np.ndarray, state: np.ndarray, max_step: float) -> np.ndarray:
    """actions: [T, D] physical units."""
    t, d = actions.shape
    h, w = 420, 920
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Physical action chunk (deg / grip) vs current state",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    plot = np.full((340, 880, 3), 28, dtype=np.uint8)
    colors = [
        (80, 180, 255),
        (80, 255, 160),
        (255, 200, 80),
        (255, 100, 140),
        (180, 140, 255),
        (120, 220, 220),
        (240, 240, 240),
    ]
    ymin = float(min(actions.min(), state.min()) - 1.0)
    ymax = float(max(actions.max(), state.max()) + 1.0)
    if ymax - ymin < 1e-3:
        ymax = ymin + 1.0

    def xy(i, val):
        x = int(i / max(t - 1, 1) * (plot.shape[1] - 1))
        y = int((1.0 - (val - ymin) / (ymax - ymin)) * (plot.shape[0] - 1))
        return x, y

    for j in range(d):
        pts = np.array([xy(i, actions[i, j]) for i in range(t)], dtype=np.int32)
        cv2.polylines(plot, [pts], False, colors[j % len(colors)], 2, cv2.LINE_AA)
        sx, sy = xy(0, state[j] if j < len(state) else actions[0, j])
        cv2.circle(plot, (sx, sy), 4, colors[j % len(colors)], -1)
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"d{j}"
        cv2.putText(plot, name, (8, 18 + j * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[j % len(colors)], 1)
    canvas[60 : 60 + plot.shape[0], 20 : 20 + plot.shape[1]] = plot
    dq = np.abs(np.diff(actions, axis=0))
    hit = float((dq > max_step).mean()) if dq.size else 0.0
    cv2.putText(
        canvas,
        f"max|dq|={dq.max():.3f}  frac>|dq|>{max_step:g}={hit:.2%}  T={t} D={d}",
        (16, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_delta_heatmap(actions: np.ndarray) -> np.ndarray:
    dq = np.abs(np.diff(actions, axis=0)).T  # [D, T-1]
    if dq.size == 0:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    vis = dq / (dq.max() + 1e-6)
    vis = (vis * 255.0).astype(np.uint8)
    vis = cv2.resize(vis, (800, 200), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    canvas = np.full((240, 820, 3), 18, dtype=np.uint8)
    canvas[30:230, 10:810] = heat
    cv2.putText(
        canvas,
        "|delta q| over chunk (rows=joints, cols=steps)",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas


@torch.no_grad()
def main() -> int:
    args = parse_args()
    force_offline()
    ckpt = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig

    sanitize_ckpt_config(ckpt)
    config = PreTrainedConfig.from_pretrained(str(ckpt))
    stats = load_stats(ckpt, config)
    policy = SmolVLAPolicy.from_pretrained(str(ckpt), config=config, dataset_stats=stats)
    policy.to(device)
    policy.eval()

    img_keys = [k for k in policy.config.image_features if "empty" not in k]
    cam_paths = [Path(args.cam1), Path(args.cam2)]
    batch: dict = {"task": args.task}
    state_dim = int(policy.config.robot_state_feature.shape[0])
    ep_info = None
    if args.episode:
        ep_info = load_episode_frame(Path(args.episode), args.frame)
        cam_paths = [ep_info["cam1"], ep_info["cam2"]]
        if ep_info["task"]:
            batch["task"] = ep_info["task"]
            args.task = ep_info["task"]
        state_np = np.asarray(ep_info["state"], dtype=np.float32)
        if state_np.shape[0] < state_dim:
            state_np = np.pad(state_np, (0, state_dim - state_np.shape[0]))
        state_np = state_np[:state_dim]
    else:
        parsed = parse_state_arg(args.state, state_dim)
        state_np = parsed if parsed is not None else np.zeros((state_dim,), dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state_np).unsqueeze(0).to(device)
    for key, path in zip(img_keys[:2], cam_paths, strict=False):
        batch[key] = load_rgb_tensor(path).to(device)

    expert = policy.model.vlm_with_expert
    n_denoise = int(policy.config.num_steps)
    keep_phases = {"prefix", "denoise_0", f"denoise_{n_denoise - 1}"}
    expert._attn_capture = {"records": [], "layers": set(layers), "phases": keep_phases}
    expert._attn_phase = "prefix"

    orig_denoise = policy.model.denoise_step
    step_i = {"n": 0}

    def denoise_step(*a, **k):
        expert._attn_phase = f"denoise_{step_i['n']}"
        out = orig_denoise(*a, **k)
        step_i["n"] += 1
        return out

    policy.model.denoise_step = denoise_step  # type: ignore[method-assign]

    actions = policy.predict_action_chunk(batch)
    policy.model.denoise_step = orig_denoise  # type: ignore[method-assign]

    # Token layout from the same prepared tensors.
    batch_n = policy._prepare_batch(dict(batch))
    images, img_masks = policy.prepare_images(batch_n)
    lang_tokens, lang_masks = policy.prepare_language(batch_n)
    slices = prefix_slices(policy, images, lang_tokens)
    vis_imgs = [tensor_to_u8(im) for im in images[:2]]

    tokenizer = policy.language_tokenizer
    ids = lang_tokens[0].tolist()
    token_items = content_token_indices(tokenizer, ids, max_n=5)
    records = expert._attn_capture["records"]
    expert._attn_capture = None

    cell = int(args.cell)
    # --- language → image ---
    for cam_i, vis in enumerate(vis_imgs):
        img_sl = slice(*slices[f"image{cam_i}"])
        rows = []
        for tok_i, tok_s in token_items:
            cols = []
            for layer in layers:
                rec = pick_record(records, layer, "prefix")
                if rec is None:
                    cols.append(put_label(resize_cell(vis, cell), f"L{layer} missing"))
                    continue
                q0, q1 = slices["lang"]
                vec = mean_heads(rec["probs"], slice(q0 + tok_i, q0 + tok_i + 1), img_sl)[0]
                ov = overlay_heatmap(vis, vec)
                cols.append(put_label(resize_cell(ov, cell), f"L{layer} '{tok_s}' -> cam{cam_i+1}"))
            rows.append(hstack_pad(cols))
        grid = vstack_pad(rows)
        path = out / f"lang2img_camera{cam_i+1}.png"
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print("wrote", path)

    # --- action → image (first denoise step, mean over action queries) ---
    act_rows = []
    for cam_i, vis in enumerate(vis_imgs):
        img_sl = slice(*slices[f"image{cam_i}"])
        cols = []
        for layer in layers:
            rec = pick_record(records, layer, "denoise_0")
            if rec is None:
                cols.append(put_label(resize_cell(vis, cell), f"L{layer} no denoise"))
                continue
            k_len = rec["k_len"]
            q_len = rec["q_len"]
            # K is prefix (cross) or prefix+suffix (self-attn layers)
            k0, k1 = img_sl.start, min(img_sl.stop, k_len)
            att = mean_heads(rec["probs"], slice(0, q_len), slice(k0, k1)).mean(axis=0)
            ov = overlay_heatmap(vis, att)
            cols.append(put_label(resize_cell(ov, cell), f"act->cam{cam_i+1} L{layer} denoise0"))
        act_rows.append(hstack_pad(cols))
    act_grid = vstack_pad(act_rows)
    act_path = out / "action2img_denoise0.png"
    cv2.imwrite(str(act_path), cv2.cvtColor(act_grid, cv2.COLOR_RGB2BGR))
    print("wrote", act_path)

    # --- physical actions ---
    act_np = actions[0].detach().float().cpu().numpy()
    state_np = batch["observation.state"][0].detach().float().cpu().numpy()
    # Convention A: first 6 dims are Δq vs current encoder; gripper stays absolute.
    if act_np.shape[1] >= 6 and float(np.abs(act_np[:, :6]).mean()) < 15.0 and float(np.abs(state_np[:6]).mean()) > 20.0:
        abs_np = act_np.copy()
        abs_np[:, :6] = state_np[:6][None, :] + act_np[:, :6]
        print("physical plot: treating action[:6] as Δq, overlaying q_now+Δq")
    else:
        abs_np = act_np
    joints = draw_joint_plot(abs_np, state_np, args.max_step_deg)
    delta = draw_delta_heatmap(act_np)
    combo = vstack_pad([joints, delta])
    phys_path = out / "action_physical.png"
    cv2.imwrite(str(phys_path), cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    print("wrote", phys_path)

    dq = np.abs(np.diff(act_np, axis=0))
    metrics = {
        "ckpt": str(ckpt),
        "task": args.task,
        "device": str(device),
        "layers": layers,
        "prefix_slices": {k: list(v) for k, v in slices.items()},
        "n_attn_records": len(records),
        "attn_index": [{"layer": r["layer_idx"], "phase": r["phase"], "q": r["q_len"], "k": r["k_len"]} for r in records],
        "tokens": [[i, s] for i, s in token_items],
        "action_shape": list(act_np.shape),
        "action_min": act_np.min(axis=0).tolist(),
        "action_max": act_np.max(axis=0).tolist(),
        "max_abs_dq": float(dq.max()) if dq.size else 0.0,
        "frac_steps_over_max_step": float((dq > args.max_step_deg).mean()) if dq.size else 0.0,
        "mean_abs_dq": float(dq.mean()) if dq.size else 0.0,
        "state": state_np.tolist(),
        "cam1": str(cam_paths[0]),
        "cam2": str(cam_paths[1]),
        "episode": str(args.episode) if args.episode else "",
        "frame": int(ep_info["frame"]) if ep_info else None,
        "recorded_action": ep_info["action"].tolist() if ep_info else None,
        "recorded_policy_action": ep_info["policy_action"].tolist() if ep_info else None,
        "intervention": ep_info["intervention"] if ep_info else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("wrote", out / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
