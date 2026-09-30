#!/usr/bin/env python3
"""LeRobot 数据集收尾：聚合 stats.json + 写入 Sendi clean_report.json。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EMBODIMENT = "rm65_dahuan"
DEFAULT_CAMERA_DEVICE = "/dev/video0"


def finalize_dataset(
    root: Path,
    *,
    embodiment_id: str = DEFAULT_EMBODIMENT,
    camera_device: str = DEFAULT_CAMERA_DEVICE,
    camera_devices: dict[str, str] | None = None,
    recording_script: str = "rm65_record_movep.py",
    notes: str = "",
) -> None:
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.datasets.utils import load_episodes_stats, load_info, write_json, write_stats

    root = root.expanduser().resolve()
    meta_dir = root / "meta"
    if not (meta_dir / "info.json").is_file():
        raise FileNotFoundError(f"缺少 meta/info.json: {root}")

    info = load_info(root)
    episodes_stats = load_episodes_stats(root)
    if episodes_stats:
        stats = aggregate_stats(list(episodes_stats.values()))
        write_stats(stats, root)
        print(f"已写入 meta/stats.json（{len(episodes_stats)} episodes）")
    else:
        print("警告: 无 episodes_stats.jsonl，跳过 stats.json", flush=True)

    cam_keys = [k for k in info["features"] if k.startswith("observation.images.")]
    devices = dict(camera_devices or {})
    cameras = {}
    for full_key in cam_keys:
        short = full_key.removeprefix("observation.images.")
        if short in ("primary", "front"):
            role = "wrist_or_front"
            default_dev = camera_device
        elif short in ("exterior", "side", "third"):
            role = "exterior"
            default_dev = ""
        elif short in ("astra", "astra_pro"):
            role = "astra"
            default_dev = "/dev/video2"
        elif short in ("astra2", "astra_pro2"):
            role = "astra2"
            default_dev = "/dev/video_astra2"
        elif short in ("mono", "cam2"):
            role = "mono_rtsp"
            default_dev = "rtsp://192.168.73.102:554/stream=1"
        else:
            role = short
            default_dev = camera_device if short == "primary" else ""
        cameras[full_key] = {
            "device": devices.get(full_key) or devices.get(short) or default_dev,
            "role": role,
            "width": info["features"][full_key]["shape"][1],
            "height": info["features"][full_key]["shape"][0],
        }

    report = {
        "schema": "sendi/clean_report/v1",
        "embodiment_id": embodiment_id,
        "robot_type": info.get("robot_type", "rm65_movep"),
        "codebase_version": info.get("codebase_version", "v2.1"),
        "fps": info.get("fps"),
        "cameras": cameras,
        "state_names": info["features"].get("observation.state", {}).get("names", []),
        "action_names": info["features"].get("action", {}).get("names", []),
        "action_semantics": "ik_target_from_movep_teleop",
        "observation_semantics": "sdk_measured_joints_deg",
        "recording_script": recording_script,
        "total_episodes": info.get("total_episodes", 0),
        "total_frames": info.get("total_frames", 0),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "notes": notes,
    }
    write_json(report, meta_dir / "clean_report.json")
    print(f"已写入 meta/clean_report.json  embodiment={embodiment_id}")


def main() -> int:
    p = argparse.ArgumentParser(description="Finalize LeRobot dataset (stats + clean_report)")
    p.add_argument(
        "--root",
        default=os.environ.get(
            "RECORD_ROOT",
            "/home/sdznkj/zkx_ws/sendlerobot/datasets/rm65_usb_insert",
        ),
    )
    p.add_argument("--embodiment-id", default=os.environ.get("RECORD_EMBODIMENT_ID", DEFAULT_EMBODIMENT))
    p.add_argument("--camera-device", default=os.environ.get("RECORD_CAMERA", DEFAULT_CAMERA_DEVICE))
    p.add_argument("--notes", default="")
    args = p.parse_args()
    finalize_dataset(
        Path(args.root),
        embodiment_id=args.embodiment_id,
        camera_device=args.camera_device,
        notes=args.notes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
