#!/usr/bin/env python3
"""Judge plots 1 (trajectory+chunk) and 4 (latency waterfall).

Plot 1 needs offline SmolVLA chunks. Run once with --infer inside docker::

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/eval_judge_plots.py --infer'

After that, host python can redraw::

    python3 scripts/eval_judge_plots.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diag/judge_plots"
EP = ROOT / "datasets/dagger_rm65_20260923_175937/episode_000000"
CHUNK_CACHE = OUT / "offline_chunks.npz"
SUMMARY = ROOT / "logs/astra_stack_20260923_142308_opengrip/summary.json"
CKPT = ROOT / "astra_smolvla"
# Evenly spaced + the previously diagnosed frame 74.
INFER_FRAMES = [0, 50, 74, 125, 175, 225, 275, 325, 375, 425]
MAX_STEP_DEG = 0.4
JOINT_NAMES = ["关节1", "关节2", "关节3", "关节4", "关节5", "关节6", "夹爪"]


def _setup_cn_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.is_file():
        font_manager.fontManager.addfont(str(cjk))
        cjk_name = font_manager.FontProperties(fname=str(cjk)).get_name()
    else:
        cjk_name = "Noto Sans CJK JP"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [cjk_name, "Noto Sans CJK JP", "DejaVu Sans"]
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
            "axes.titlecolor": "#F0F0F0",
            "legend.facecolor": "#1C1C1C",
            "legend.edgecolor": "#444444",
            "grid.color": "#2A2A2A",
            "savefig.facecolor": "#111111",
            "savefig.dpi": 180,
        }
    )
    return plt


def infer_chunks(frames: list[int]) -> Path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch

    from smolvla_diag_visualize import (  # type: ignore
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

    starts = []
    chunks = []
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
            act = policy.predict_action_chunk(batch)[0].detach().float().cpu().numpy()
            starts.append(int(frame))
            chunks.append(act.astype(np.float32))
            print(f"infer {fi + 1}/{len(frames)} frame={frame} chunk={act.shape}", flush=True)
            del batch, act
            if device.type == "cuda":
                torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHUNK_CACHE,
        starts=np.asarray(starts, dtype=np.int32),
        chunks=np.stack(chunks, axis=0),
        frames=np.asarray(frames, dtype=np.int32),
    )
    print("wrote", CHUNK_CACHE)
    return CHUNK_CACHE


def plot_trajectory_chunk() -> Path:
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
    highlight = 74
    hex_colors = ["#50B4FF", "#50FFA0", "#FFC850", "#FF648C", "#B48CFF", "#78DCDC", "#F0F0F0"]

    human_dq = float(np.abs(np.diff(ac[:, :6], axis=0)).mean()) if t > 1 else 0.0
    chunk_dq, over = [], []
    for ch in chunks:
        dq = np.abs(np.diff(ch[:, :6], axis=0))
        chunk_dq.append(float(dq.mean()))
        over.append(float((dq > MAX_STEP_DEG).mean()))

    fig, axes = plt.subplots(d, 1, figsize=(16.5, 12.2), sharex=True, gridspec_kw={"hspace": 0.08})
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.98)
    fig.suptitle("离线开环：动作轨迹跟踪与 Chunk 叠加", fontsize=18, fontweight="bold", y=0.995)
    fig.text(
        0.01,
        0.965,
        "连续实线 = 编码器实际关节角（整段都发生过）　　虚线 = 人手示教动作（本段 100% 人工，几乎贴着实线）\n"
        "橙色短峰线 = 模型在该时刻开环预测的未来 50 步，不是真机走过的路；亮橙 = 第 74 帧那一截",
        fontsize=11,
        color="#C8C8C8",
        va="top",
    )

    legend_done = False
    for j, ax in enumerate(axes):
        c = hex_colors[j]
        for start, ch in zip(starts, chunks):
            x0, x1 = int(start), int(start) + ch.shape[0]
            ax.axvspan(x0, x1, color="#2A3348" if int(start) == highlight else "#1E1E22", alpha=0.85, lw=0)
        (h_line,) = ax.plot(x, ac[:, j], ls="--", lw=1.2, color=c, alpha=0.55, zorder=2)
        (s_line,) = ax.plot(x, st[:, j], ls="-", lw=2.2, color=c, zorder=3)
        vla_lines = []
        for start, ch in zip(starts, chunks):
            if j >= ch.shape[1]:
                continue
            xs = np.arange(int(start), int(start) + ch.shape[0])
            is_hl = int(start) == highlight
            (ln,) = ax.plot(
                xs,
                ch[:, j],
                color="#FF9F1C" if is_hl else "#E0A14A",
                lw=2.6 if is_hl else 1.7,
                alpha=0.95 if is_hl else 0.78,
                zorder=4,
                solid_capstyle="round",
            )
            ax.plot(xs[0], ch[0, j], "o", color="#FF9F1C", ms=4.5 if is_hl else 3.2, zorder=5)
            vla_lines.append(ln)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=12, color=c, rotation=0, ha="right", va="center", labelpad=14)
        ax.grid(True, axis="x", alpha=0.25)
        ax.tick_params(labelsize=10)
        if not legend_done:
            ax.legend(
                [s_line, h_line, vla_lines[0]],
                ["编码器实际关节（连续实线）", "人手示教（虚线）", "VLA 预测 chunk（橙色峰线）"],
                loc="upper right",
                fontsize=10,
                framealpha=0.92,
            )
            legend_done = True

    axes[-1].set_xlabel("帧（约 20 Hz 采样）", fontsize=12)
    axes[-1].set_xlim(0, t - 1)
    fig.text(
        0.01,
        0.008,
        f"人手平均每步 {human_dq:.3f} 度　　VLA chunk 平均每步 {float(np.mean(chunk_dq)):.3f} 度　　"
        f"超过 {MAX_STEP_DEG:g} 度限幅的步占比 {float(np.mean(over)):.0%}　　"
        f"离线推理帧 {list(map(int, starts))}",
        fontsize=10.5,
        color="#D2C878",
    )
    path = OUT / "01_offline_trajectory_chunk.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return path


def plot_latency_waterfall() -> Path:
    plt = _setup_cn_mpl()
    from matplotlib.patches import Patch

    summary = json.loads(SUMMARY.read_text())
    lat = summary["latency_ms"]
    infer = lat["inference"]
    chunk = lat["chunk_total"]
    post = max(0.0, float(chunk["mean"]) - float(infer["mean"]))
    rows = [
        ("1  相机缓存 / USB缓冲", 80.0, False),
        ("2  gRPC观测单向传输", float(lat["one_way"]["mean"]), True),
        ("3  VLA推理（一个chunk）", float(infer["mean"]), True),
        ("4  chunk前后处理", post, True),
        ("5  首个控制周期", 50.0, False),
        ("6  指令后伺服跟踪", 20.0, False),
    ]
    total = sum(r[1] for r in rows)
    p50, p95, mx = float(infer["p50"]), float(infer["p95"]), float(infer["max"])

    fig = plt.figure(figsize=(15.2, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle("端到端延迟瀑布图", fontsize=20, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.935,
        "从快门到第一次运动　|　astra_stack_20260923_142308_opengrip　|　实心=本轮实测，斜纹=经验估计（未打时间戳）",
        ha="center",
        fontsize=12,
        color="#C0C0C0",
    )

    t0 = 0.0
    y_pos = list(range(len(rows) - 1, -1, -1))
    for i, ((name, ms, measured), y) in enumerate(zip(rows, y_pos)):
        color = "#47A8FF" if measured else "#7A7A7A"
        bar = ax.barh(
            y,
            ms,
            left=t0,
            height=0.62,
            color=color,
            edgecolor="#111111",
            linewidth=0.4,
            hatch="///" if not measured else None,
            zorder=3,
        )
        ax.text(
            total * 1.01,
            y,
            f"{ms:.1f} ms　{'实测' if measured else '估计'}",
            va="center",
            ha="left",
            fontsize=12,
            color="#F0E080" if measured else "#B0B0B0",
        )
        t0 += ms

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, total * 1.28)
    ax.set_xlabel("累计时间（毫秒）", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.axvline(total, color="#666666", ls=":", lw=1)
    ax.text(total, len(rows) - 0.15, f"合计约 {total:.0f} ms", color="#AAAAAA", fontsize=11, ha="right")
    ax.legend(
        handles=[
            Patch(facecolor="#47A8FF", label="本轮实测"),
            Patch(facecolor="#7A7A7A", hatch="///", label="经验估计"),
        ],
        loc="lower right",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)

    # inference spread
    ax2.barh(0, mx, height=0.45, color="#2A3C50", label="最大")
    ax2.barh(0, p95, height=0.45, color="#3C78B4", label="P95")
    ax2.barh(0, p50, height=0.45, color="#47A8FF", label="P50")
    ax2.set_yticks([0])
    ax2.set_yticklabels(["VLA 推理分布"])
    ax2.set_xlabel("毫秒", fontsize=13)
    ax2.set_xlim(0, mx * 1.12)
    ax2.tick_params(labelsize=11)
    ax2.legend(loc="upper right", ncol=3, fontsize=11)
    ax2.text(
        p50,
        -0.42,
        f"P50 = {p50:.0f} ms　　P95 = {p95:.0f} ms　　最大 = {mx:.0f} ms",
        ha="left",
        fontsize=12,
        color="#D0D8E8",
    )
    ax2.set_ylim(-0.85, 0.7)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.text(
        0.01,
        0.015,
        f"合计约 {total:.0f} ms　　chunk 数 {infer['n']}　　观测约 {summary['obs']['fps_mean']} FPS（目标 20）　　"
        f"真机执行每步平均 {summary['arm']['step']['mean']} 度",
        fontsize=11.5,
        color="#D0D0D0",
    )
    path = OUT / "04_latency_waterfall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def write_index(p1: Path, p4: Path) -> Path:
    idx = OUT / "index.html"
    idx.write_text(
        "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
        "<h3>Judge plots v1 (1 and 4)</h3><ul>"
        f"<li><a href='{p1.name}'>01 trajectory &amp; chunk</a></li>"
        f"<li><a href='{p4.name}'>04 latency waterfall</a></li>"
        "</ul></body></html>\n",
        encoding="utf-8",
    )
    return idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--infer", action="store_true", help="re-run offline VLA chunks (needs GPU/torch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.infer or not CHUNK_CACHE.is_file():
        if not args.infer:
            raise SystemExit(
                f"missing {CHUNK_CACHE}; run with --infer inside docker rm65_dev2"
            )
        infer_chunks(INFER_FRAMES)
    p1 = plot_trajectory_chunk()
    p4 = plot_latency_waterfall()
    idx = write_index(p1, p4)
    print("wrote", p1)
    print("wrote", p4)
    print("index", idx)


if __name__ == "__main__":
    main()
