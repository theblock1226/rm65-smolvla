#!/usr/bin/env python3
"""Offline open-loop eval of the Astra delta-q SmolVLA checkpoint.

Infers 24-step chunks on DAgger ep000, converts Δq → absolute joints, overlays
human trajectory. Attention heatmaps reuse smolvla_diag_visualize.py.

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/eval_astra_deltaq_offline.py --infer'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diag/astra_deltaq_offline"
EP = ROOT / "datasets/dagger_rm65_20260923_175937/episode_000000"
CKPT = ROOT / "astra_smolvla"
CHUNK_CACHE = OUT / "offline_chunks.npz"
INFER_FRAMES = [0, 48, 74, 120, 168, 216, 264, 312, 360, 408]
HIGHLIGHT = 74
JOINT_NAMES = ["关节1", "关节2", "关节3", "关节4", "关节5", "关节6", "夹爪"]
HEX = ["#50B4FF", "#50FFA0", "#FFC850", "#FF648C", "#B48CFF", "#78DCDC", "#F0F0F0"]


def _setup_cn_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.is_file():
        font_manager.fontManager.addfont(str(cjk))
        name = font_manager.FontProperties(fname=str(cjk)).get_name()
        plt.rcParams["font.sans-serif"] = [name, "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#161616",
            "axes.edgecolor": "#555555",
            "text.color": "#EEEEEE",
            "xtick.color": "#BBBBBB",
            "ytick.color": "#BBBBBB",
            "axes.labelcolor": "#DDDDDD",
            "legend.facecolor": "#1C1C1C",
            "legend.edgecolor": "#444444",
            "grid.color": "#2A2A2A",
            "savefig.facecolor": "#111111",
        }
    )
    return plt


def infer_chunks(frames: list[int]) -> Path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch

    from smolvla_diag_visualize import (
        force_offline,
        load_episode_frame,
        load_rgb_tensor,
        load_stats,
        sanitize_ckpt_config,
    )
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig

    force_offline()
    sanitize_ckpt_config(CKPT)
    config = PreTrainedConfig.from_pretrained(str(CKPT))
    stats = load_stats(CKPT, config)
    print("action stats mean", stats["action"]["mean"][:6].tolist())
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    if hasattr(policy.model, "vlm_with_expert"):
        policy.model.vlm_with_expert._attn_capture = None

    img_keys = [k for k in policy.config.image_features if "empty" not in k]
    state_dim = int(policy.config.robot_state_feature.shape[0])
    meta = json.loads((EP / "meta.json").read_text())
    task = meta.get("task") or "Stack three cylinder blocks. Grasp the left cylinder."
    chunk_n = int(policy.config.chunk_size)

    starts, deltas, abs_chunks = [], [], []
    with torch.no_grad():
        for fi, frame in enumerate(frames):
            info = load_episode_frame(EP, frame)
            state_np = np.asarray(info["state"], dtype=np.float32)
            if state_np.shape[0] < state_dim:
                state_np = np.pad(state_np, (0, state_dim - state_np.shape[0]))
            state_np = state_np[:state_dim]
            batch = {
                "task": task,
                "observation.state": torch.from_numpy(state_np).unsqueeze(0).to(device),
            }
            for key, path in zip(img_keys[:2], [info["cam1"], info["cam2"]], strict=False):
                batch[key] = load_rgb_tensor(path).to(device)
            pred = policy.predict_action_chunk(batch)[0].detach().float().cpu().numpy()
            pred = pred[:chunk_n]
            abs_q = pred.copy()
            abs_q[:, :6] = state_np[:6][None, :] + pred[:, :6]
            starts.append(int(frame))
            deltas.append(pred.astype(np.float32))
            abs_chunks.append(abs_q.astype(np.float32))
            print(
                f"infer {fi + 1}/{len(frames)} f={frame} "
                f"dmean={np.abs(pred[:, :6]).mean():.3f} "
                f"dmax={np.abs(pred[:, :6]).max():.3f} "
                f"grip=[{pred[:, 6].min():.2f},{pred[:, 6].max():.2f}]",
                flush=True,
            )
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHUNK_CACHE,
        starts=np.asarray(starts, np.int32),
        deltas=np.stack(deltas),
        chunks=np.stack(abs_chunks),
        frames=np.asarray(frames, np.int32),
    )
    print("wrote", CHUNK_CACHE)
    return CHUNK_CACHE


def plot_trajectory() -> Path:
    plt = _setup_cn_mpl()
    z = np.load(EP / "trajectory.npz")
    st = z["states"].astype(np.float32)
    ac = z["actions"].astype(np.float32)
    t, d = st.shape
    d = min(d, 7)
    x = np.arange(t)
    cache = np.load(CHUNK_CACHE)
    starts = cache["starts"].astype(int)
    chunks = cache["chunks"].astype(np.float32)
    deltas = cache["deltas"].astype(np.float32)

    fig, axes = plt.subplots(d, 1, figsize=(16.5, 12.4), sharex=True, gridspec_kw={"hspace": 0.08})
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.98)
    fig.suptitle("Astra Δq 开环：示教轨迹 vs 预测 chunk（已加回当前编码器）", fontsize=18, fontweight="bold", y=0.995)
    fig.text(
        0.01,
        0.965,
        "实线=编码器　　虚线=人手绝对角　　橙色=模型 Δq + 当前关节（夹爪为绝对开合）　　亮橙=第 74 帧\n"
        "chunk_size=24。旧版橙峰是绝对角乱甩；这版应贴着虚线小步走。",
        fontsize=11,
        color="#C8C8C8",
        va="top",
    )
    legend_done = False
    for j, ax in enumerate(axes):
        c = HEX[j]
        for start, ch in zip(starts, chunks):
            ax.axvspan(int(start), int(start) + ch.shape[0], color="#2A3348" if int(start) == HIGHLIGHT else "#1E1E22", alpha=0.8, lw=0)
        (h_line,) = ax.plot(x, ac[:, j], ls="--", lw=1.2, color=c, alpha=0.55, zorder=2)
        (s_line,) = ax.plot(x, st[:, j], ls="-", lw=2.2, color=c, zorder=3)
        vla = None
        for start, ch in zip(starts, chunks):
            xs = np.arange(int(start), int(start) + ch.shape[0])
            hl = int(start) == HIGHLIGHT
            (vla,) = ax.plot(
                xs,
                ch[:, j],
                color="#FF9F1C" if hl else "#E0A14A",
                lw=2.6 if hl else 1.7,
                zorder=4,
            )
            ax.plot(xs[0], ch[0, j], "o", color="#FF9F1C", ms=4.2 if hl else 3.0, zorder=5)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=12, color=c, rotation=0, ha="right", va="center", labelpad=14)
        ax.grid(True, axis="x", alpha=0.25)
        if not legend_done:
            ax.legend(
                [s_line, h_line, vla],
                ["编码器", "人手示教", "VLA Δq→绝对角"],
                loc="upper right",
                fontsize=10,
            )
            legend_done = True
    axes[-1].set_xlabel("帧（约 20 Hz）", fontsize=12)
    axes[-1].set_xlim(0, t - 1)
    human_dq = float(np.abs(np.diff(ac[:, :6], axis=0)).mean())
    pred_d = float(np.abs(deltas[:, :, :6]).mean())
    pred_max = float(np.abs(deltas[:, :, :6]).max())
    fig.text(
        0.01,
        0.008,
        f"人手逐步 |Δq|={human_dq:.3f}°　　模型输出 |Δq| 均值={pred_d:.3f}° 最大={pred_max:.3f}°　　"
        f"推理帧 {list(map(int, starts))}",
        fontsize=10.5,
        color="#D2C878",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "01_trajectory_chunk.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote", path)
    return path


def plot_delta_horizon() -> Path:
    """Convention A: pred[:6] vs human remaining (action - q_now), 24-step horizon."""
    plt = _setup_cn_mpl()
    z = np.load(EP / "trajectory.npz")
    st = z["states"].astype(np.float32)
    ac = z["actions"].astype(np.float32)
    cache = np.load(CHUNK_CACHE)
    starts = cache["starts"].astype(int)
    deltas = cache["deltas"].astype(np.float32)
    n = len(starts)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18.2, 7.6), sharex=True, sharey=False)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.05, right=0.99, hspace=0.38, wspace=0.22)
    fig.suptitle("开环 Δq 视界：模型 vs 人手剩余位移（相对当前编码器）", fontsize=17, fontweight="bold", y=0.995)
    fig.text(
        0.01,
        0.94,
        "每格一个推理时刻。实线=模型 chunk[:6]　　虚线=人手 action[t:t+24]−q_now。约定 A：第 0 步应接近单帧位移，末步接近 1.2 s 后到达的角位移。",
        fontsize=11,
        color="#C8C8C8",
        va="top",
    )
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        t0 = int(starts[i])
        pred = deltas[i]
        nstep = min(pred.shape[0], len(st) - t0)
        xs = np.arange(nstep)
        human = ac[t0 : t0 + nstep, :6] - st[t0, :6]
        for j in range(6):
            ax.plot(xs, human[:, j], ls="--", lw=1.1, color=HEX[j], alpha=0.7)
            ax.plot(xs, pred[:nstep, j], ls="-", lw=1.6, color=HEX[j])
        mae = float(np.abs(pred[:nstep, :6] - human).mean())
        ax.set_title(f"帧 {t0}  MAE={mae:.2f}°", fontsize=11, color="#FFC878" if t0 == HIGHLIGHT else "#DDDDDD")
        ax.axhline(0, color="#555555", lw=0.6)
        ax.grid(True, alpha=0.2)
        if i // cols == rows - 1:
            ax.set_xlabel("chunk 步", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("Δq (°)", fontsize=10)
    path = OUT / "02_deltaq_horizon.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("wrote", path)
    return path


def write_index(paths: list[Path]) -> Path:
    items = "".join(f"<li><a href='{p.name}'>{p.name}</a></li>" for p in paths if p is not None)
    attn = ROOT / "outputs/diag/smolvla_attn_deltaq_f074"
    extra = ""
    if attn.is_dir():
        extra = "<h4>注意力（第 74 帧）</h4><ul>" + "".join(
            f"<li><a href='../smolvla_attn_deltaq_f074/{p.name}'>{p.name}</a></li>"
            for p in sorted(attn.glob("*.png"))
        ) + "</ul>"
    idx = OUT / "index.html"
    idx.write_text(
        "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
        "<h3>Astra Δq 离线开环</h3><ul>"
        + items
        + "</ul>"
        + extra
        + "</body></html>\n",
        encoding="utf-8",
    )
    return idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--infer", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.infer or not CHUNK_CACHE.is_file():
        if not args.infer:
            raise SystemExit(f"missing {CHUNK_CACHE}; run with --infer in docker")
        infer_chunks(INFER_FRAMES)
    p1 = plot_trajectory()
    p2 = plot_delta_horizon()
    idx = write_index([p1, p2])
    print("index", idx)


if __name__ == "__main__":
    main()
