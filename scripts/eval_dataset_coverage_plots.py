#!/usr/bin/env python3
"""Dataset coverage plots for RM65 cleaned SFT data.

2) Instruction diversity cosine matrix
3) Per-joint Delta-q histograms (action - state)
4) Per-task action mean/std vs astra_smolvla MEAN_STD
1) Vision feature coverage (PCA 2D of SmolVLM embed_image)

Host (2/3/4)::

    python3 scripts/eval_dataset_coverage_plots.py

Vision (GPU, docker)::

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/eval_dataset_coverage_plots.py --vision'
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "datasets/cleaned"
OUT = ROOT / "outputs/diag/dataset_plots"
CKPT = ROOT / "astra_smolvla"
NORM_JSON = ROOT / "outputs/diag/astra_norm_stats.json"
EMB_CACHE = OUT / "vision_embeddings.npz"

TASKS = [
    ("rm65_stacking_blocks", "堆叠", "observation.images.astra"),
    ("rm65_cleanup_the_desk", "桌面", "observation.images.astra"),
    ("rm65_place_to_slot", "插槽", "observation.images.astra"),
    ("rm65_usb_insert", "USB", "observation.images.mono"),
]
JOINTS = ["关节1", "关节2", "关节3", "关节4", "关节5", "关节6", "夹爪"]
TASK_COLORS = ["#47A8FF", "#50FFA0", "#FFC850", "#FF648C"]
CAM_MARKERS = {"front": "o", "astra": "s", "mono": "^"}

SHORT_INSTR = {
    "Stack three cylinder blocks. Grasp the left cylinder.": "堆叠·抓左柱",
    "Stack three cylinder blocks. Place it on the middle cylinder and release.": "堆叠·放到中柱",
    "Stack three cylinder blocks. Grasp the right cylinder.": "堆叠·抓右柱",
    "Stack three cylinder blocks. Place it on the stack and release.": "堆叠·放到堆上",
    "Put three objects from the desk into the box. Grasp the left object.": "桌面·抓左物",
    "Put three objects from the desk into the box. Place it in the box and release.": "桌面·放进盒",
    "Put three objects from the desk into the box. Grasp the middle object.": "桌面·抓中物",
    "Put three objects from the desk into the box. Grasp the right object.": "桌面·抓右物",
    "Place three objects into matching slots on the silver card. Rotate the gripper and grasp the cylinder.": "插槽·抓圆柱",
    "Place three objects into matching slots on the silver card. Lift and rotate the cylinder to align with the round slot.": "插槽·对准圆孔",
    "Place three objects into matching slots on the silver card. Insert into the round slot and release.": "插槽·插入圆孔",
    "Place three objects into matching slots on the silver card. Rotate the gripper and grasp the T-shaped wooden block.": "插槽·抓T块",
    "Place three objects into matching slots on the silver card. Lift and rotate the T-shaped wooden block to align with the T-shaped slot.": "插槽·对准T孔",
    "Place three objects into matching slots on the silver card. Insert into the T-shaped slot and release.": "插槽·插入T孔",
    "Place three objects into matching slots on the silver card. Rotate the gripper and grasp the blue box.": "插槽·抓蓝盒",
    "Place three objects into matching slots on the silver card. Lift and rotate the blue box to align with the rectangular slot.": "插槽·对准方孔",
    "Place three objects into matching slots on the silver card. Insert into the rectangular slot and release.": "插槽·插入方孔",
    "Insert the blue USB module into the white base. Rotate the gripper and grasp the blue USB module.": "USB·抓住模块",
    "Insert the blue USB module into the white base. Lift and align the USB with the port.": "USB·对准接口",
    "Insert the blue USB module into the white base. Insert into the white base and release.": "USB·插入底座",
    "failed": "USB·failed",
}


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
            "savefig.dpi": 160,
        }
    )
    return plt


def _save(fig, plt, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)
    return path


def _load_unique_instructions() -> list[tuple[str, str, str]]:
    """(dataset_cn, short, full_text) unique by full_text, first occurrence."""
    seen: set[str] = set()
    rows = []
    for folder, cn, _cam in TASKS:
        path = CLEAN / folder / "meta/tasks.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            full = str(obj["task"]).strip()
            if full in seen:
                continue
            seen.add(full)
            short = SHORT_INSTR.get(full, full[:18])
            rows.append((cn, short, full))
    return rows


def plot_instruction_matrix() -> Path:
    plt = _setup_cn_mpl()
    rows = _load_unique_instructions()
    texts = [r[2] for r in rows]
    labels = [r[1] for r in rows]
    n = len(texts)

    def ngrams(s: str, k: int = 3) -> list[str]:
        s = s.lower()
        if len(s) < k:
            return [s]
        return [s[i : i + k] for i in range(len(s) - k + 1)]

    docs = [Counter(ngrams(t)) for t in texts]
    vocab = sorted(set().union(*docs))
    idx = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((n, len(vocab)), np.float64)
    for i, c in enumerate(docs):
        for w, cnt in c.items():
            x[i, idx[w]] = cnt
    df = (x > 0).sum(0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    x *= idf
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    sim = x @ x.T

    fig, ax = plt.subplots(figsize=(11.5, 10.2))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="字符 3-gram TF-IDF 余弦")
    ax.set_title("语言指令多样性与语义距离", fontsize=18, fontweight="bold", pad=12)
    fig.text(
        0.01,
        0.01,
        "桌面三个「放进盒子」子任务共用同一句，矩阵里只出现一次。"
        "USB 的 failed 是废指令。块状高相似来自共用 overall 前缀。",
        fontsize=10,
        color="#C8C8C8",
    )
    return _save(fig, plt, "02_instruction_diversity.png")


def _iter_parquet_action_state(folder: str):
    data_dir = CLEAN / folder / "data"
    for path in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(path, columns=["action", "observation.state"])
        act = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        st = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        if act.ndim != 2 or st.ndim != 2:
            continue
        d = min(act.shape[1], st.shape[1], 7)
        yield act[:, :d], st[:, :d]


def plot_delta_q() -> Path:
    plt = _setup_cn_mpl()
    per_task: dict[str, np.ndarray] = {}
    for folder, cn, _cam in TASKS:
        chunks = []
        for act, st in _iter_parquet_action_state(folder):
            k = min(6, act.shape[1], st.shape[1])
            chunks.append(act[:, :k] - st[:, :k])
        dq = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6), np.float32)
        per_task[cn] = dq
        print(f"delta-q {cn}: n={dq.shape[0]} mean|dq|={np.abs(dq).mean():.4f} std={dq.std(axis=0)}")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), sharey=False)
    fig.suptitle("关节 Δq 各维分布（动作绝对角 − 当前编码器）", fontsize=18, fontweight="bold")
    for j, ax in enumerate(axes.ravel()):
        astra_cols = np.concatenate(
            [per_task[t[1]][:, j] for t in TASKS[:3] if per_task[t[1]].size],
            axis=0,
        )
        p = float(np.percentile(np.abs(astra_cols), 99.7)) if astra_cols.size else 1.0
        lo, hi = -max(p, 0.4), max(p, 0.4)
        bins = np.linspace(lo, hi, 90)
        for (folder, cn, _c), color in zip(TASKS[:3], TASK_COLORS[:3]):
            arr = per_task[cn]
            if arr.size == 0:
                continue
            ax.hist(
                arr[:, j],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=color,
                label=cn,
            )
        ax.axvline(0.0, color="#888888", ls=":", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_title(JOINTS[j], fontsize=13)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=9)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right")
        if j >= 3:
            ax.set_xlabel("度 / 步", fontsize=11)
    fig.text(
        0.01,
        0.01,
        "逐步 Δq = action[:6] − state[:6]。图中只画 Astra 三任务（横轴按它们 99.7 分位）。"
        f"USB mean|dq|={float(np.abs(per_task['USB']).mean()):.1f}°，不在同一尺度上，未叠进此图"
        "（cleaned README：USB mono 相对关节滞后约 10 帧）。",
        fontsize=10,
        color="#D2C878",
    )
    return _save(fig, plt, "03_delta_q_distribution.png")


def plot_task_vs_norm() -> Path:
    plt = _setup_cn_mpl()
    glob = json.loads(NORM_JSON.read_text())
    g_mean = np.asarray(glob["action.mean"], dtype=np.float64)
    g_std = np.asarray(glob["action.std"], dtype=np.float64)

    task_stats = []
    for folder, cn, _cam in TASKS:
        s = json.loads((CLEAN / folder / "meta/stats.json").read_text())
        task_stats.append(
            (
                cn,
                np.asarray(s["action"]["mean"], dtype=np.float64),
                np.asarray(s["action"]["std"], dtype=np.float64),
            )
        )

    d = 7
    x = np.arange(d)
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(14.2, 9.4), gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.32})
    ax = axes[0]
    ax.errorbar(
        x - 2 * width,
        g_mean[:d],
        yerr=g_std[:d],
        fmt="D",
        color="#EEEEEE",
        ecolor="#888888",
        capsize=3,
        ms=7,
        label=f"astra 全局 MEAN_STD  n={int(glob['n'])}",
        zorder=5,
    )
    for i, ((cn, mean, std), color) in enumerate(zip(task_stats, TASK_COLORS)):
        ax.errorbar(
            x + (i - 1.0) * width,
            mean[:d],
            yerr=std[:d],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            ms=6,
            label=cn,
            alpha=0.95,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(JOINTS[:d], fontsize=12)
    ax.set_ylabel("绝对角（度） / 夹爪", fontsize=12)
    ax.set_title("分任务动作目标 vs 旧权重归一化统计", fontsize=18, fontweight="bold")
    ax.legend(fontsize=10, ncol=3, loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    for i, ((cn, mean, std), color) in enumerate(zip(task_stats, TASK_COLORS)):
        z = (mean[:d] - g_mean[:d]) / (g_std[:d] + 1e-8)
        ax2.bar(x + (i - 1.5) * width, z, width=width, color=color, label=cn)
    ax2.axhline(0.0, color="#888888", lw=0.8)
    ax2.axhline(1.0, color="#666666", ls="--", lw=0.7)
    ax2.axhline(-1.0, color="#666666", ls="--", lw=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(JOINTS[:d], fontsize=12)
    ax2.set_ylabel("相对全局 (μ_task − μ_glob) / σ_glob", fontsize=11)
    ax2.set_title("各任务均值偏离旧 MEAN_STD 的 z 分数", fontsize=14)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=9, ncol=4)
    fig.text(
        0.01,
        0.01,
        "全局统计来自 astra_smolvla preprocessor（stack+desk+slot 共 152305 帧，不含 USB）。"
        "虚线 ±1σ。旧权重 state 配置是 6 维，safetensors 里仍存了 7 维。",
        fontsize=10,
        color="#C8C8C8",
    )
    return _save(fig, plt, "04_task_vs_normalizer.png")


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    z = x @ vt[:2].T
    evr = (s[:2] ** 2) / (s ** 2).sum()
    return z.astype(np.float32), evr


def _sample_video_frames(mp4: Path, n_frames_hint: int, picks: int = 3) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        n = max(n_frames_hint, picks)
    idxs = np.linspace(0.08, 0.92, picks)
    frames = []
    for frac in idxs:
        i = int(frac * max(n - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    cap.release()
    return frames


def extract_vision_embeddings(n_ep: int = 12, frames_per_ep: int = 3) -> Path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
    from lerobot.configs.policies import PreTrainedConfig
    from smolvla_diag_visualize import force_offline, load_stats, sanitize_ckpt_config

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

    vecs, task_ids, cam_ids, ep_ids = [], [], [], []
    cam_name = []
    with torch.no_grad():
        for ti, (folder, cn, cam2_key) in enumerate(TASKS):
            info = json.loads((CLEAN / folder / "meta/info.json").read_text())
            n_total = int(info["total_episodes"])
            eps = np.linspace(0, n_total - 1, num=min(n_ep, n_total), dtype=int)
            cams = [("front", "observation.images.front"), ("cam2", cam2_key)]
            for ep in eps:
                for cam_tag, vkey in cams:
                    mp4 = (
                        CLEAN
                        / folder
                        / "videos"
                        / "chunk-000"
                        / vkey
                        / f"episode_{int(ep):06d}.mp4"
                    )
                    if not mp4.is_file():
                        print("missing", mp4)
                        continue
                    bgrs = _sample_video_frames(mp4, 400, picks=frames_per_ep)
                    for bgr in bgrs:
                        rgb = bgr[:, :, ::-1].copy()
                        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
                        t = resize_with_pad(t, 512, 512, pad_value=0)
                        t = t * 2.0 - 1.0
                        tok = policy.model.vlm_with_expert.embed_image(t)
                        vec = tok.float().mean(dim=1).squeeze(0).cpu().numpy()
                        vecs.append(vec.astype(np.float32))
                        task_ids.append(ti)
                        cam_ids.append(0 if cam_tag == "front" else 1)
                        cam_name.append("front" if cam_tag == "front" else ("mono" if "mono" in vkey else "astra"))
                        ep_ids.append(int(ep))
            print(f"vision {cn}: accumulated {len(vecs)} embeddings", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMB_CACHE,
        feat=np.stack(vecs, axis=0),
        task=np.asarray(task_ids, np.int32),
        cam=np.asarray(cam_ids, np.int32),
        cam_name=np.asarray(cam_name),
        ep=np.asarray(ep_ids, np.int32),
    )
    print("wrote", EMB_CACHE, "n=", len(vecs), "dim=", vecs[0].shape)
    return EMB_CACHE


def plot_vision_pca() -> Path:
    plt = _setup_cn_mpl()
    zfile = np.load(EMB_CACHE, allow_pickle=True)
    feat = zfile["feat"]
    task = zfile["task"]
    cam_name = [str(x) for x in zfile["cam_name"]]
    xy, evr = pca2(feat)

    fig, ax = plt.subplots(figsize=(11.8, 8.6))
    for ti, (_f, cn, _c) in enumerate(TASKS):
        for cam, marker in (("front", "o"), ("astra", "s"), ("mono", "^")):
            m = (task == ti) & np.array([c == cam for c in cam_name])
            if not np.any(m):
                continue
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=28,
                c=TASK_COLORS[ti],
                marker=marker,
                alpha=0.82,
                edgecolors="none",
                label=f"{cn} · {cam}",
            )
    ax.set_title("视觉特征覆盖（SmolVLM embed_image → PCA）", fontsize=18, fontweight="bold")
    ax.set_xlabel(f"PC1  解释方差 {evr[0]*100:.1f}%", fontsize=12)
    ax.set_ylabel(f"PC2  解释方差 {evr[1]*100:.1f}%", fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.grid(True, alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "每任务约 12 条 episode × 3 帧 × 两路相机。圆=front，方=astra，三角=USB mono。"
        "本机无 sklearn，用 PCA 代替 t-SNE；分离仍能看覆盖与相机域。",
        fontsize=10,
        color="#C8C8C8",
    )
    return _save(fig, plt, "01_vision_coverage_pca.png")


def write_index(paths: list[Path]) -> Path:
    items = "".join(f"<li><a href='{p.name}'>{p.name}</a></li>" for p in paths)
    idx = OUT / "index.html"
    idx.write_text(
        "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
        "<h3>数据集覆盖图</h3><ul>"
        + items
        + "</ul></body></html>\n",
        encoding="utf-8",
    )
    return idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vision", action="store_true", help="extract SmolVLM embeddings (GPU)")
    p.add_argument("--skip-stats", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not args.skip_stats:
        paths.append(plot_instruction_matrix())
        paths.append(plot_delta_q())
        paths.append(plot_task_vs_norm())
    if args.vision or EMB_CACHE.is_file():
        if args.vision or not EMB_CACHE.is_file():
            extract_vision_embeddings()
        paths.append(plot_vision_pca())
    elif not args.skip_stats:
        print("skip vision (pass --vision inside docker)")
    # keep previously written vision plot in index if present
    vis = OUT / "01_vision_coverage_pca.png"
    if vis.is_file() and vis not in paths:
        paths.insert(0, vis)
    idx = write_index(paths)
    print("index", idx)


if __name__ == "__main__":
    main()
