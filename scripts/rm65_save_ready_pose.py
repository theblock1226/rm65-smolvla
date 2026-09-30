#!/usr/bin/env python3
"""读取当前 RM65 关节角，写入 rm65_ready_pose.json（供 --go-ready 使用）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

READY_POSE_PATH = Path(__file__).resolve().parent / "rm65_ready_pose.json"
RM_IK_PATH = Path(__file__).resolve().parents[1] / "sendrobot" / "send_robot" / "rm_ik.py"
DEFAULT_JOINTS = [0.0, 20.0, 70.0, 10.0, -20.0, 0.0]


def _load_kinematics():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rm_ik", RM_IK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.RM65Kinematics


def load_ready_joints_deg(path: Path | None = None) -> list[float]:
    cfg_path = path or READY_POSE_PATH
    if not cfg_path.is_file():
        return list(DEFAULT_JOINTS)
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        joints = [float(v) for v in data["joints_deg"][:6]]
        if len(joints) != 6:
            raise ValueError("joints_deg 需要 6 个数")
        return joints
    except Exception as exc:
        print(f"警告: 读取 {cfg_path} 失败，用默认就绪位: {exc}", file=sys.stderr)
        return list(DEFAULT_JOINTS)


def save_ready_joints_deg(joints: list[float], path: Path | None = None) -> Path:
    cfg_path = path or READY_POSE_PATH
    payload = {
        "joints_deg": [round(float(v), 4) for v in joints[:6]],
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "note": "遥操/录制 --go-ready 目标。可反复运行 save 覆盖。",
    }
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def main() -> int:
    p = argparse.ArgumentParser(description="把当前关节角存为遥操就绪位")
    p.add_argument("--ip", default=os.environ.get("RM_IP", "192.168.73.18"))
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--joints-deg",
        type=float,
        nargs=6,
        default=None,
        help="不读臂，直接写入这 6 个关节角(度)",
    )
    p.add_argument("--show", action="store_true", help="只打印当前配置，不写入")
    args = p.parse_args()

    if args.show:
        joints = load_ready_joints_deg()
        print(f"配置文件: {READY_POSE_PATH}")
        print(f"就绪关节(deg): {[round(q, 2) for q in joints]}")
        return 0

    if args.joints_deg is not None:
        joints = [float(v) for v in args.joints_deg]
        src = "命令行"
    else:
        RM65Kinematics = _load_kinematics()
        kin = RM65Kinematics(ip=args.ip, port=args.port)
        kin.connect()
        try:
            joints = [float(v) for v in kin.get_current_joints_deg()[:6]]
        finally:
            kin.disconnect()
            time.sleep(0.2)
        src = "SDK 当前关节"

    path = save_ready_joints_deg(joints)
    print(f"来源: {src}")
    print(f"已写入: {path}")
    print(f"joints_deg: {[round(q, 2) for q in joints]}")
    print("下次遥操/录制带 --go-ready 将 movej 到该位姿。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
