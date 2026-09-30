#!/usr/bin/env python3
"""Extract one mid-episode frame pair + run SmolVLA attention for missing tasks.

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      HF_HUB_OFFLINE=1 PYTHONPATH=. python3 scripts/draw_missing_task_attn.py'
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_ROOT = ROOT / "outputs/diag/attn_frames"

TASKS = [
    {
        "name": "usb",
        "ckpt": ROOT / "smolvla_rm65_usb_100ep_s40000",
        "dataset": ROOT / "datasets/cleaned/rm65_usb_insert",
        "cam2_key": "observation.images.mono",
        "out": ROOT / "outputs/diag/smolvla_attn_usb_f074",
        "frame": 74,
        "episode": 0,
    },
    {
        "name": "cleanup",
        "ckpt": ROOT / "smolvla_rm65_cleanup_47ep_s40000",
        "dataset": ROOT / "datasets/cleaned/rm65_cleanup_the_desk",
        "cam2_key": "observation.images.astra",
        "out": ROOT / "outputs/diag/smolvla_attn_cleanup_f074",
        "frame": 74,
        "episode": 0,
    },
    {
        "name": "place",
        "ckpt": ROOT / "smolvla_rm65_place_47ep_s40000",
        "dataset": ROOT / "datasets/cleaned/rm65_place_to_slot",
        "cam2_key": "observation.images.astra",
        "out": ROOT / "outputs/diag/smolvla_attn_place_f074",
        "frame": 74,
        "episode": 0,
    },
]


def extract_frame(mp4: Path, index: int) -> np.ndarray:
    import av

    container = av.open(str(mp4))
    for i, frame in enumerate(container.decode(video=0)):
        if i == index:
            rgb = frame.to_ndarray(format="rgb24")
            container.close()
            return rgb
    container.close()
    raise FileNotFoundError(f"{mp4} has no frame {index}")


def prepare_task(task: dict) -> dict:
    import cv2
    import pandas as pd

    ds = task["dataset"]
    ep = int(task["episode"])
    frame = int(task["frame"])
    info = json.loads((ds / "meta/info.json").read_text())
    tasks = {}
    for line in (ds / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    chunk = ep // int(info.get("chunks_size", 1000))
    pq = ds / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    df = pd.read_parquet(pq)
    n = len(df)
    if frame >= n:
        frame = n // 2
    state = np.asarray(df["observation.state"].iloc[frame], dtype=np.float32)[:7]
    task_str = tasks[int(df["task_index"].iloc[frame])]
    front_p = ds / info["video_path"].format(
        episode_chunk=chunk, episode_index=ep, video_key="observation.images.front"
    )
    cam2_p = ds / info["video_path"].format(
        episode_chunk=chunk, episode_index=ep, video_key=task["cam2_key"]
    )
    front = extract_frame(front_p, frame)
    cam2 = extract_frame(cam2_p, frame)
    out_dir = FRAME_ROOT / task["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cam1_jpg = out_dir / "camera1_front.jpg"
    cam2_jpg = out_dir / "camera2.jpg"
    cv2.imwrite(str(cam1_jpg), cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cam2_jpg), cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR))
    meta = {
        "task": task_str,
        "state": state.tolist(),
        "frame": frame,
        "episode": ep,
        "cam1": str(cam1_jpg),
        "cam2": str(cam2_jpg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("prepared", task["name"], "frame", frame, "task=", task_str[:60])
    return meta


def run_attn(task: dict, meta: dict) -> None:
    out = Path(task["out"])
    if (out / "lang2img_camera1.png").is_file() and (out / "action2img_denoise0.png").is_file():
        print("skip existing attn", out)
        return
    # Use --state=... so leading '-' in joint angles is not parsed as a flag.
    state_csv = ",".join(f"{x:.6f}" for x in meta["state"])
    cmd = [
        sys.executable,
        str(ROOT / "scripts/smolvla_diag_visualize.py"),
        "--ckpt",
        str(task["ckpt"]),
        "--cam1",
        meta["cam1"],
        "--cam2",
        meta["cam2"],
        "--task",
        meta["task"],
        f"--state={state_csv}",
        "--out",
        str(out),
        "--device",
        "cuda",
    ]
    print("RUN", " ".join(cmd[:6]), "...", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    # tiny index
    pngs = sorted(out.glob("*.png"))
    items = "".join(f"<li><a href='{p.name}'>{p.name}</a></li>" for p in pngs)
    (out / "index.html").write_text(
        f"<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
        f"<h3>attn {task['name']} f{meta['frame']:03d}</h3><ul>{items}</ul></body></html>\n",
        encoding="utf-8",
    )


def main() -> None:
    for task in TASKS:
        meta = prepare_task(task)
        run_attn(task, meta)
    print("ALL_ATTN_OK")


if __name__ == "__main__":
    main()
