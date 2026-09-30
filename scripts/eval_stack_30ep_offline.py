#!/usr/bin/env python3
"""Offline open-loop eval for smolvla_rm65_stack_30ep_s1000.

Stitches n_action_steps=16 from each chunk on three cleaned stacking episodes.
Joints: q_cmd[:6] = q_now[:6] + pred[:6]. Gripper: hysteresis 0.35 / 0.65.

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/eval_stack_30ep_offline.py'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"
OUT = ROOT / "outputs/diag/stack_30ep_offline"
LOCAL_VLM = ROOT / "base_models/SmolVLM2-500M-Video-Instruct"
EPISODES = [0, 1, 2]
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
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


class GripperHysteresis:
    def __init__(self, low: float = 0.35, high: float = 0.65, initial: float = 0.5) -> None:
        self.low = low
        self.high = high
        self.initial = initial
        self.state: float | None = None

    def reset(self) -> None:
        self.state = None

    def step(self, value: float) -> float:
        if self.state is None:
            self.state = 1.0 if value >= self.initial else 0.0
        elif self.state >= 0.5:
            if value <= self.low:
                self.state = 0.0
        elif value >= self.high:
            self.state = 1.0
        return self.state


def sanitize_config(ckpt: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import dataclasses

    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig

    allowed = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    allowed |= {f.name for f in dataclasses.fields(PreTrainedConfig)}
    allowed.add("type")
    cfg_path = ckpt / "config.json"
    cfg = json.loads(cfg_path.read_text())
    dropped = sorted(k for k in cfg if k not in allowed)
    for k in dropped:
        cfg.pop(k, None)
    if LOCAL_VLM.is_dir():
        cfg["vlm_model_name"] = str(LOCAL_VLM)
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    print("dropped config keys:", dropped)


def load_episode(ep: int) -> dict:
    import pandas as pd

    info = json.loads((DS / "meta/info.json").read_text())
    tasks = {}
    for line in (DS / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    pq = DS / info["data_path"].format(episode_chunk=0, episode_index=ep)
    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    task_index = df["task_index"].to_numpy().astype(np.int64)
    return {
        "state": state,
        "action": action,
        "task_index": task_index,
        "tasks": tasks,
        "info": info,
        "n": len(df),
    }


def read_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    import av

    want = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    container = av.open(str(path))
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            out[i] = frame.to_ndarray(format="rgb24")
        if len(out) == len(want):
            break
    container.close()
    missing = want - set(out)
    if missing:
        raise FileNotFoundError(f"{path} missing frames {sorted(missing)[:8]}")
    return out


def rgb_tensor(img: np.ndarray, torch):
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def infer_episode(policy, ep: int, device) -> dict:
    import torch

    data = load_episode(ep)
    info = data["info"]
    n = data["n"]
    step = int(policy.config.n_action_steps)
    starts = list(range(0, n, step))
    front = read_video_frames(
        DS / info["video_path"].format(episode_chunk=0, episode_index=ep, video_key="observation.images.front"),
        starts,
    )
    astra = read_video_frames(
        DS / info["video_path"].format(episode_chunk=0, episode_index=ep, video_key="observation.images.astra"),
        starts,
    )
    hyst = GripperHysteresis()
    policy.reset()
    pred_abs = np.full((n, 7), np.nan, dtype=np.float32)
    with torch.no_grad():
        for fi, t0 in enumerate(starts):
            state_np = data["state"][t0, :7].copy()
            task = data["tasks"][int(data["task_index"][t0])]
            batch = {
                "task": task,
                "observation.state": torch.from_numpy(state_np).unsqueeze(0).to(device),
                "observation.images.camera1": rgb_tensor(front[t0], torch).to(device),
                "observation.images.camera2": rgb_tensor(astra[t0], torch).to(device),
            }
            pred = policy.predict_action_chunk(batch)[0].detach().float().cpu().numpy()
            take = pred[: min(step, n - t0)]
            # Policy returns robot commands when use_relative_actions is set.
            if not getattr(policy.config, "use_relative_actions", False):
                take = take.copy()
                take[:, :6] = state_np[:6] + take[:, :6]
                for i in range(take.shape[0]):
                    take[i, 6] = hyst.step(float(take[i, 6]))
            pred_abs[t0 : t0 + take.shape[0]] = take
            print(
                f"ep{ep:03d} {fi + 1}/{len(starts)} f={t0} "
                f"|dq|={np.abs(take[:, :6] - state_np[:6]).mean():.3f} grip={take[0, 6]:.0f}",
                flush=True,
            )
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
    data["pred"] = pred_abs
    data["episode"] = ep
    return data


def plot_episode(data: dict) -> Path:
    plt = _setup_cn_mpl()
    st = data["state"][:, :7]
    ac = data["action"][:, :7]
    pr = data["pred"]
    n = data["n"]
    ep = data["episode"]
    x = np.arange(n)
    fig, axes = plt.subplots(7, 1, figsize=(16.5, 12.4), sharex=True, gridspec_kw={"hspace": 0.08})
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.98)
    fig.suptitle(
        f"stack 30ep/s1000 open-loop  ep{ep:03d}",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.01,
        0.965,
        "solid=encoder    dashed=demo action    orange=open-loop (joints: q + dq, gripper hysteresis 0.35/0.65)\n"
        "infer every 16 frames, execute the first 16 steps of each chunk",
        fontsize=11,
        color="#C8C8C8",
        va="top",
    )
    legend_done = False
    for j, ax in enumerate(axes):
        c = HEX[j]
        (h_line,) = ax.plot(x, ac[:, j], ls="--", lw=1.2, color=c, alpha=0.55, zorder=2)
        (s_line,) = ax.plot(x, st[:, j], ls="-", lw=2.0, color=c, zorder=3)
        (vla,) = ax.plot(x, pr[:, j], color="#FF9F1C", lw=1.8, zorder=4)
        ax.set_ylabel(JOINT_NAMES[j], fontsize=12, color=c, rotation=0, ha="right", va="center", labelpad=14)
        ax.grid(True, axis="x", alpha=0.25)
        if not legend_done:
            ax.legend([s_line, h_line, vla], ["encoder", "demo", "open-loop VLA"], loc="upper right", fontsize=10)
            legend_done = True
    axes[-1].set_xlabel("frame (20 Hz)", fontsize=12)
    axes[-1].set_xlim(0, n - 1)
    valid = np.isfinite(pr[:, 0])
    mae = float(np.abs(pr[valid, :6] - ac[valid, :6]).mean())
    grip_acc = float(np.mean(np.abs(pr[valid, 6] - ac[valid, 6]) < 0.5))
    fig.text(
        0.01,
        0.008,
        f"joint |pred-demo| mean {mae:.2f} deg    gripper agree {grip_acc:.0%}    frames {n}",
        fontsize=10.5,
        color="#D2C878",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ep{ep:03d}_trajectory.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote", path)
    return path


def main() -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from smolvla_diag_visualize import load_stats

    sanitize_config(CKPT)
    config = PreTrainedConfig.from_pretrained(str(CKPT))
    stats = load_stats(CKPT, config)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    print("device", device, "chunk", policy.config.chunk_size, "n_action_steps", policy.config.n_action_steps)

    paths = []
    for ep in EPISODES:
        data = infer_episode(policy, ep, device)
        paths.append(plot_episode(data))
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    items = "".join(f"<li><a href='{p.name}'>{p.name}</a></li>" for p in paths)
    idx = OUT / "index.html"
    idx.write_text(
        "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
        "<h3>叠块 30ep/s1000 离线开环（ep000–002）</h3><ul>"
        + items
        + "</ul></body></html>\n",
        encoding="utf-8",
    )
    print("index", idx)


if __name__ == "__main__":
    main()
