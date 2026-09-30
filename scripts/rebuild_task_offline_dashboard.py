#!/usr/bin/env python3
"""Rebuild 4-task offline eval dashboard from existing metrics.json files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diag/task_offline"
TASKS = ["stack", "usb", "cleanup", "place"]
CKPT = {
    "stack": "smolvla_rm65_stack_30ep_s1000",
    "usb": "smolvla_rm65_usb_100ep_s40000",
    "cleanup": "smolvla_rm65_cleanup_47ep_s40000",
    "place": "smolvla_rm65_place_47ep_s40000",
}
ATTN = {
    "stack": ROOT / "outputs/diag/smolvla_attn_stack30_f074",
    "usb": ROOT / "outputs/diag/smolvla_attn_usb_f074",
    "cleanup": ROOT / "outputs/diag/smolvla_attn_cleanup_f074",
    "place": ROOT / "outputs/diag/smolvla_attn_place_f074",
}


def main() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#111",
            "axes.facecolor": "#161616",
            "axes.edgecolor": "#555",
            "text.color": "#EEE",
            "xtick.color": "#BBB",
            "ytick.color": "#BBB",
            "axes.labelcolor": "#DDD",
            "grid.color": "#2A2A2A",
            "savefig.facecolor": "#111",
            "font.family": "DejaVu Sans",
        }
    )
    all_metrics = {}
    coverage = {}
    for name in TASKS:
        mets = json.loads((OUT / name / "metrics.json").read_text())
        all_metrics[name] = mets
        traj_ok = all((OUT / name / f"ep{ep:03d}_trajectory.png").is_file() for ep in (0, 1, 2))
        attn_dir = ATTN[name]
        attn_ok = (attn_dir / "lang2img_camera1.png").is_file()
        coverage[name] = {
            "ckpt": CKPT[name],
            "traj": traj_ok,
            "attn": attn_ok,
            "metrics": True,
            "attn_path": str(attn_dir.relative_to(ROOT / "outputs/diag")),
        }

    names = list(all_metrics)
    means = [float(np.mean([m["joint_mae_vs_demo_deg"] for m in all_metrics[k]])) for k in names]
    grips = [float(np.mean([m["gripper_agree"] for m in all_metrics[k]])) * 100 for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    fig.suptitle("RM65 SmolVLA — 4-task offline open-loop summary", fontsize=15, fontweight="bold")
    axes[0].bar(names, means, color=["#50B4FF", "#FFC850", "#B48CFF", "#FF648C"])
    axes[0].set_ylabel("mean joint MAE vs demo (deg)")
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(means):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color="#D2C878")
    axes[1].bar(names, grips, color=["#50FFA0", "#78DCDC", "#50B4FF", "#FFC850"])
    axes[1].set_ylabel("mean gripper agree (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(grips):
        axes[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color="#D2C878")
    fig.savefig(OUT / "cross_task_mae.png", dpi=150, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

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
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    rows = []
    for name, mets in all_metrics.items():
        mean_mae = float(np.mean([m["joint_mae_vs_demo_deg"] for m in mets]))
        mean_grip = float(np.mean([m["gripper_agree"] for m in mets]))
        cov = coverage[name]
        rows.append(
            "<tr>"
            f"<td><b>{name}</b></td><td>{cov['ckpt']}</td>"
            f"<td>{'Y' if cov['traj'] else 'N'}</td>"
            f"<td>{'Y' if cov['attn'] else 'pending'}</td><td>Y</td>"
            f"<td>{mean_mae:.2f}</td><td>{mean_grip:.0%}</td>"
            f"<td><a href='{name}/index.html'>{name}/</a> "
            f"<a href='../{cov['attn_path']}/'>attn</a></td></tr>"
        )
    (OUT / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>RM65 SmolVLA task offline eval</title></head>"
        "<body style='background:#111;color:#eee;font-family:sans-serif;padding:20px;max-width:1100px'>"
        "<h2>RM65 SmolVLA 四任务离线开环评测汇总</h2>"
        "<p>open-loop：n_action_steps=16，cleaned demo ep000–002。</p>"
        "<p><img src='cross_task_mae.png' style='max-width:100%;border:1px solid #333'/></p>"
        "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%'>"
        "<tr><th>任务</th><th>ckpt</th><th>轨迹</th><th>注意力</th><th>指标</th>"
        "<th>mean MAE°</th><th>grip</th><th>链接</th></tr>"
        + "".join(rows)
        + "</table><h3>指标速览</h3><ul>"
        + "".join(
            f"<li><b>{k}</b>: MAE {v['mean_joint_mae_vs_demo_deg']:.2f}° / "
            f"grip {v['mean_gripper_agree']:.0%}</li>"
            for k, v in summary["tasks"].items()
        )
        + "</ul></body></html>\n",
        encoding="utf-8",
    )
    for k, v in summary["tasks"].items():
        print(
            f"{k}: MAE={v['mean_joint_mae_vs_demo_deg']:.2f} "
            f"grip={v['mean_gripper_agree']:.0%} "
            f"traj={coverage[k]['traj']} attn={coverage[k]['attn']}"
        )
    print("OK", OUT / "index.html")


if __name__ == "__main__":
    main()
