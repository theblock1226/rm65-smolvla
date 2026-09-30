#!/usr/bin/env python3
"""评测启动前准备：movej 到录制就绪位 + 夹爪张开到 1.0。

须在 play_action / record_state / robot_client 未运行时调用（避免抢 SDK / 抢运动）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from rm65_ready import load_ready_joints_deg, movej_ready  # noqa: E402


def open_gripper(ip: str, port: int, norm: float = 1.0) -> None:
    from sendrobot.send_robot.dahuan_gripper import DahuanGripperModbus
    from sendrobot.send_robot.rm_ik import RM65Kinematics

    class _Log:
        def info(self, msg: str) -> None:
            print(msg)

        def warning(self, msg: str) -> None:
            print(f"警告: {msg}", file=sys.stderr)

    kin = RM65Kinematics(ip=ip, port=port)
    kin.connect()
    try:
        grip = DahuanGripperModbus(kin.robot)
        grip.setup(_Log())
        ok, hw = grip.set_position_normalized(float(norm), min_delta=0.0)
        if not ok:
            raise RuntimeError(f"夹爪写入失败 norm={norm}")
        print(f"夹爪已命令张开 norm={norm} hw={hw}")
        time.sleep(1.0)
        # 供后续 record_state / SendRobot 对齐：写入环境变量文件不现实，
        # 这里打印明确约定，eval 脚本会 export RM65_INITIAL_GRIPPER_POS
        print(f"RM65_INITIAL_GRIPPER_POS={float(norm)}")
    finally:
        kin.disconnect()
        time.sleep(0.3)


def main() -> int:
    p = argparse.ArgumentParser(description="Eval prep: ready pose + open gripper")
    p.add_argument("--ip", default="192.168.73.18")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--speed", type=int, default=5, help="movej speed")
    p.add_argument("--gripper-norm", type=float, default=1.0, help="1=开 0=合")
    p.add_argument("--pose", type=Path, default=None, help="关节 json，默认录制就绪位")
    p.add_argument("--skip-ready", action="store_true")
    p.add_argument("--skip-gripper", action="store_true")
    args = p.parse_args()

    joints = load_ready_joints_deg(args.pose)
    pose_name = args.pose.name if args.pose is not None else "rm65_ready_pose.json"
    print(f"就绪位目标 ({pose_name}): {[round(q, 2) for q in joints]}")
    if not args.skip_ready:
        print(f">>> movej → 就绪位 (speed={args.speed})")
        import rclpy

        rclpy.init()
        try:
            movej_ready(speed=args.speed, joints_deg=joints, pose_path=args.pose)
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        print("就绪位完成")
    else:
        print("跳过就绪位")

    if not args.skip_gripper:
        print(f">>> 打开夹爪 norm={args.gripper_norm}")
        open_gripper(args.ip, args.port, args.gripper_norm)
    else:
        print("跳过夹爪")

    print("prep 完成，可启动推理栈")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
