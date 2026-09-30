#!/usr/bin/env python3
"""Unified offline open-loop eval for RM65 SmolVLA task checkpoints.

Covers stack / usb / cleanup / place. Writes per-task trajectory plots,
metrics.json, MAE bar chart, and a cross-task dashboard.

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/eval_task_ckpts_offline.py'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL_VLM = ROOT / "base_models/SmolVLM2-500M-Video-Instruct"
OUT_ROOT = ROOT / "outputs/diag/task_offline"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
HEX = ["#50B4FF", "#50FFA0", "#FFC850", "#FF648C", "#B48CFF", "#78DCDC", "#F0F0F0"]

TASKS = {
    "stack": {
        "ckpt": ROOT / "smolvla_rm65_stack_30ep_s1000",
        "dataset": ROOT / "datasets/cleaned/rm65_stacking_blocks",
        "cam2_key": "observation.images.astra",
        "title": "叠块 stack 30ep/s1000",
        "legacy_out": ROOT / "outputs/diag/stack_30ep_offline",
    },
    "usb": {
        "ckpt": ROOT / "smolvla_rm65_usb_100ep_s40000",
        "dataset": ROOT / "datasets/cleaned/rm65_usb_insert",
        "cam2_key": "observation.images.mono",
        "title": "USB 插入 usb 100ep/s40000",
        "legacy_out": None,
    },
    "cleanup": {
        "ckpt": ROOT / "smolvla_rm65_cleanup_47ep_s40000",
        "dataset": ROOT / "datasets/cleaned/rm65_cleanup_the_desk",
        "cam2_key": "observation.images.astra",
        "title": "桌面清理 cleanup 47ep/s40000",
        "legacy_out": None,
    },
    "place": {
        "ckpt": ROOT / "smolvla_rm65_place_47ep_s40000",
        "dataset": ROOT / "datasets/cleaned/rm65_place_to_slot",
        "cam2_key": "observation.images.astra",
        "title": "槽位放置 place 47ep/s40000",
        "legacy_out": None,
    },
}


def _setup_cn_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Prefer SC face inside the TTC; set family by resolved name.
    cjk_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for cjk in cjk_candidates:
        if not cjk.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(cjk))
        except Exception:
            pass
        # Index 0 of NotoSansCJK TTC is typically JP; pick SC via FontProperties.
        for idx in (3, 0, 1, 2, 4):
            try:
                prop = font_manager.FontProperties(fname=str(cjk), index=idx)
                name = prop.get_name()
                if name:
                    plt.rcParams["font.sans-serif"] = [name, "Noto Sans CJK SC", "DejaVu Sans"]
                    break
            except Exception:
                continue
        break
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
    if dropped:
        print("dropped config keys:", dropped)


def load_episode(ds: Path, ep: int) -> dict:
    import pandas as pd

    info = json.loads((ds / "meta/info.json").read_text())
    tasks = {}
    for line in (ds / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    chunk = ep // int(info.get("chunks_size", 1000))
    pq = ds / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
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
        "chunk": chunk,
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


def infer_episode(policy, ds: Path, cam2_key: str, ep: int, device) -> dict:
    import torch

    data = load_episode(ds, ep)
    info = data["info"]
    n = data["n"]
    chunk_id = data["chunk"]
    step = int(policy.config.n_action_steps)
    starts = list(range(0, n, step))
    relative = bool(getattr(policy.config, "use_relative_actions", False))
    front = read_video_frames(
        ds
        / info["video_path"].format(
            episode_chunk=chunk_id, episode_index=ep, video_key="observation.images.front"
        ),
        starts,
    )
    cam2 = read_video_frames(
        ds
        / info["video_path"].format(episode_chunk=chunk_id, episode_index=ep, video_key=cam2_key),
        starts,
    )
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
                "observation.images.camera2": rgb_tensor(cam2[t0], torch).to(device),
            }
            pred = policy.predict_action_chunk(batch)[0].detach().float().cpu().numpy()
            take = pred[: min(step, n - t0)].copy()
            # Relative ckpts: policy already returns absolute cmds (q+Δq + grip hyst).
            # Non-relative: policy may apply grip hyst but leaves joints as Δq.
            if not relative:
                take[:, :6] = state_np[:6] + take[:, :6]
            pred_abs[t0 : t0 + take.shape[0]] = take
            if fi % 5 == 0 or fi + 1 == len(starts):
                print(
                    f"  ep{ep:03d} {fi + 1}/{len(starts)} f={t0} "
                    f"|dq|={np.abs(take[:, :6] - state_np[:6]).mean():.3f} "
                    f"grip={take[0, 6]:.0f}",
                    flush=True,
                )
            del batch
            if device.type == "cuda" and fi % 8 == 0:
                torch.cuda.empty_cache()
    data["pred"] = pred_abs
    data["episode"] = ep
    data["relative"] = relative
    return data


def episode_metrics(data: dict) -> dict:
    st = data["state"][:, :7]
    ac = data["action"][:, :7]
    pr = data["pred"]
    valid = np.isfinite(pr[:, 0])
    joint_mae = float(np.abs(pr[valid, :6] - ac[valid, :6]).mean())
    joint_mae_vs_enc = float(np.abs(pr[valid, :6] - st[valid, :6]).mean())
    per_joint = {
        JOINT_NAMES[j]: float(np.abs(pr[valid, j] - ac[valid, j]).mean()) for j in range(6)
    }
    grip_acc = float(np.mean(np.abs(pr[valid, 6] - ac[valid, 6]) < 0.5))
    return {
        "episode": int(data["episode"]),
        "frames": int(data["n"]),
        "valid_frames": int(valid.sum()),
        "joint_mae_vs_demo_deg": joint_mae,
        "joint_mae_vs_encoder_deg": joint_mae_vs_enc,
        "per_joint_mae_vs_demo": per_joint,
        "gripper_agree": grip_acc,
        "use_relative_actions": bool(data.get("relative", False)),
    }


def plot_episode(data: dict, out: Path, title: str) -> Path:
    plt = _setup_cn_mpl()
    st = data["state"][:, :7]
    ac = data["action"][:, :7]
    pr = data["pred"]
    n = data["n"]
    ep = data["episode"]
    x = np.arange(n)
    fig, axes = plt.subplots(7, 1, figsize=(16.5, 12.4), sharex=True, gridspec_kw={"hspace": 0.08})
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.98)
    fig.suptitle(f"{title}  open-loop  ep{ep:03d}", fontsize=17, fontweight="bold", y=0.995)
    mode = "relative→abs (policy)" if data.get("relative") else "Δq + q_now (eval)"
    fig.text(
        0.01,
        0.965,
        f"solid=encoder    dashed=demo action    orange=open-loop VLA ({mode})\n"
        "infer every n_action_steps frames, execute first n_action_steps of each chunk",
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
    m = episode_metrics(data)
    fig.text(
        0.01,
        0.008,
        f"joint |pred-demo| mean {m['joint_mae_vs_demo_deg']:.2f} deg    "
        f"gripper agree {m['gripper_agree']:.0%}    frames {n}",
        fontsize=10.5,
        color="#D2C878",
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ep{ep:03d}_trajectory.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote", path)
    return path


def plot_task_mae_bars(metrics: list[dict], out: Path, title: str) -> Path:
    plt = _setup_cn_mpl()
    eps = [f"ep{m['episode']:03d}" for m in metrics]
    maes = [m["joint_mae_vs_demo_deg"] for m in metrics]
    grips = [m["gripper_agree"] * 100 for m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.suptitle(f"{title} — offline metrics", fontsize=14, fontweight="bold")
    axes[0].bar(eps, maes, color="#50B4FF")
    axes[0].set_ylabel("joint MAE vs demo (°)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(eps, grips, color="#50FFA0")
    axes[1].set_ylabel("gripper agree (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, axis="y", alpha=0.3)
    path = out / "metrics_bars.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", path)
    return path


def plot_cross_task_summary(all_metrics: dict[str, list[dict]], out: Path) -> Path:
    plt = _setup_cn_mpl()
    names = list(all_metrics.keys())
    means = [float(np.mean([m["joint_mae_vs_demo_deg"] for m in all_metrics[k]])) for k in names]
    grips = [float(np.mean([m["gripper_agree"] for m in all_metrics[k]])) * 100 for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    fig.suptitle("RM65 SmolVLA — 四任务离线开环汇总", fontsize=15, fontweight="bold")
    axes[0].bar(names, means, color=["#50B4FF", "#FFC850", "#B48CFF", "#FF648C"][: len(names)])
    axes[0].set_ylabel("mean joint MAE vs demo (°)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(names, grips, color=["#50FFA0", "#78DCDC", "#50B4FF", "#FFC850"][: len(names)])
    axes[1].set_ylabel("mean gripper agree (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, axis="y", alpha=0.3)
    path = out / "cross_task_mae.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("wrote", path)
    return path


def write_task_index(out: Path, title: str, paths: list[Path], metrics: list[dict]) -> None:
    items = "".join(f"<li><a href='{p.name}'>{p.name}</a></li>" for p in paths)
    rows = "".join(
        "<tr>"
        f"<td>ep{m['episode']:03d}</td>"
        f"<td>{m['frames']}</td>"
        f"<td>{m['joint_mae_vs_demo_deg']:.3f}</td>"
        f"<td>{m['gripper_agree']:.0%}</td>"
        "</tr>"
        for m in metrics
    )
    (out / "index.html").write_text(
        "<html><body style='background:#111;color:#eee;font-family:sans-serif;padding:16px'>"
        f"<h3>{title}</h3>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;margin:12px 0'>"
        "<tr><th>ep</th><th>frames</th><th>joint MAE°</th><th>grip agree</th></tr>"
        + rows
        + "</table><ul>"
        + items
        + "</ul></body></html>\n",
        encoding="utf-8",
    )


def write_dashboard(all_metrics: dict[str, list[dict]], coverage: dict) -> Path:
    out = OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    plot_cross_task_summary(all_metrics, out)

    rows = []
    for name, mets in all_metrics.items():
        mean_mae = float(np.mean([m["joint_mae_vs_demo_deg"] for m in mets]))
        mean_grip = float(np.mean([m["gripper_agree"] for m in mets]))
        cov = coverage.get(name, {})
        rows.append(
            "<tr>"
            f"<td><b>{name}</b></td>"
            f"<td>{cov.get('ckpt','')}</td>"
            f"<td>{'✓' if cov.get('traj') else '✗'}</td>"
            f"<td>{'✓' if cov.get('attn') else '✗'}</td>"
            f"<td>{'✓' if cov.get('metrics') else '✗'}</td>"
            f"<td>{mean_mae:.2f}</td>"
            f"<td>{mean_grip:.0%}</td>"
            f"<td><a href='{name}/index.html'>{name}/</a></td>"
            "</tr>"
        )

    summary = {
        "tasks": {
            k: {
                "episodes": mets,
                "mean_joint_mae_vs_demo_deg": float(np.mean([m["joint_mae_vs_demo_deg"] for m in mets])),
                "mean_gripper_agree": float(np.mean([m["gripper_agree"] for m in mets])),
            }
            for k, mets in all_metrics.items()
        },
        "coverage": coverage,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    html = out / "index.html"
    html.write_text(
        "<html><head><meta charset='utf-8'><title>RM65 SmolVLA task offline eval</title></head>"
        "<body style='background:#111;color:#eee;font-family:sans-serif;padding:20px;max-width:1100px'>"
        "<h2>RM65 SmolVLA 四任务离线开环评测汇总</h2>"
        "<p>open-loop：按 n_action_steps 切 chunk，与 cleaned demo 关节/夹爪对比。"
        "注意力图见 <code>outputs/diag/smolvla_attn_*</code>。</p>"
        "<p><img src='cross_task_mae.png' style='max-width:100%;border:1px solid #333'/></p>"
        "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%'>"
        "<tr><th>任务</th><th>ckpt</th><th>轨迹图</th><th>注意力</th><th>指标</th>"
        "<th>mean MAE°</th><th>grip</th><th>目录</th></tr>"
        + "".join(rows)
        + "</table>"
        "<h3>覆盖说明</h3>"
        "<ul>"
        "<li>stack：既有 stack_30ep_offline + attn_stack30_f074/f216</li>"
        "<li>usb / cleanup / place：本脚本新绘轨迹 + metrics；注意力由 smolvla_diag_visualize 补</li>"
        "</ul>"
        "</body></html>\n",
        encoding="utf-8",
    )
    print("dashboard", html)
    return html


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default="usb,cleanup,place,stack")
    p.add_argument("--episodes", type=str, default="0,1,2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


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

    args = parse_args()
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("WARN: cuda unavailable, using cpu")

    all_metrics: dict[str, list[dict]] = {}
    coverage: dict[str, dict] = {}

    for name in task_names:
        if name not in TASKS:
            raise SystemExit(f"unknown task {name}; choose from {list(TASKS)}")
        meta = TASKS[name]
        ckpt: Path = meta["ckpt"]
        ds: Path = meta["dataset"]
        out = OUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n==== {name}  ckpt={ckpt.name}  ds={ds.name} ====", flush=True)

        # Reuse legacy stack plots if present and skip-existing
        legacy = meta.get("legacy_out")
        if (
            args.skip_existing
            and legacy
            and all((legacy / f"ep{ep:03d}_trajectory.png").is_file() for ep in episodes)
            and (out / "metrics.json").is_file()
        ):
            mets = json.loads((out / "metrics.json").read_text())
            all_metrics[name] = mets
            coverage[name] = {
                "ckpt": ckpt.name,
                "traj": True,
                "attn": any(
                    (ROOT / "outputs/diag" / d).is_dir()
                    for d in ["smolvla_attn_stack30_f074", "smolvla_attn_stack30_f216"]
                ),
                "metrics": True,
            }
            print("skip existing", name)
            continue

        if not ckpt.is_dir() or not (ckpt / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing ckpt {ckpt}")
        if not ds.is_dir():
            raise FileNotFoundError(f"missing dataset {ds}")

        sanitize_config(ckpt)
        config = PreTrainedConfig.from_pretrained(str(ckpt))
        stats = load_stats(ckpt, config)
        policy = SmolVLAPolicy.from_pretrained(str(ckpt), config=config, dataset_stats=stats)
        policy.to(device)
        policy.eval()
        print(
            f"device={device} chunk={policy.config.chunk_size} "
            f"n_action_steps={policy.config.n_action_steps} "
            f"relative={getattr(policy.config, 'use_relative_actions', False)}",
            flush=True,
        )

        paths: list[Path] = []
        mets: list[dict] = []
        for ep in episodes:
            traj_path = out / f"ep{ep:03d}_trajectory.png"
            npy_path = out / f"ep{ep:03d}_pred.npy"
            if args.skip_existing and traj_path.is_file() and npy_path.is_file():
                print(f"skip existing ep{ep:03d}")
                # still need metrics — recompute from npy if possible
                data = load_episode(ds, ep)
                data["pred"] = np.load(npy_path)
                data["episode"] = ep
                data["relative"] = bool(getattr(policy.config, "use_relative_actions", False))
                m = episode_metrics(data)
                mets.append(m)
                paths.append(traj_path)
                continue
            data = infer_episode(policy, ds, meta["cam2_key"], ep, device)
            np.save(npy_path, data["pred"])
            paths.append(plot_episode(data, out, meta["title"]))
            m = episode_metrics(data)
            mets.append(m)
            print(
                f"  metrics ep{ep:03d}: MAE={m['joint_mae_vs_demo_deg']:.2f}° "
                f"grip={m['gripper_agree']:.0%}",
                flush=True,
            )
            del data
            if device.type == "cuda":
                torch.cuda.empty_cache()

        paths.append(plot_task_mae_bars(mets, out, meta["title"]))
        (out / "metrics.json").write_text(json.dumps(mets, indent=2) + "\n")
        write_task_index(out, meta["title"], paths, mets)
        # symlink/copy legacy stack notice
        if legacy and legacy.is_dir():
            (out / "NOTE_legacy.txt").write_text(
                f"Earlier plots also at {legacy}\n", encoding="utf-8"
            )

        all_metrics[name] = mets
        attn_ok = False
        if name == "stack":
            attn_ok = (ROOT / "outputs/diag/smolvla_attn_stack30_f074").is_dir()
        else:
            attn_ok = (ROOT / f"outputs/diag/smolvla_attn_{name}_f074").is_dir()
        coverage[name] = {
            "ckpt": ckpt.name,
            "traj": True,
            "attn": attn_ok,
            "metrics": True,
        }

        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_dashboard(all_metrics, coverage)
    print("DONE")


if __name__ == "__main__":
    main()
