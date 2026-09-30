#!/usr/bin/env python3
"""缓慢移动 RM65 到初始/自定义关节位姿（需 rm_driver 已运行）。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Movej

SCRIPT_DIR = Path(__file__).resolve().parent
HOME_CONFIG = SCRIPT_DIR / "rm65_home_joints.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("需要 PyYAML: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_controller_init_joints_deg(ip: str, port: int) -> list[float]:
    from sendrobot.send_robot.rm_ik import RM65Kinematics

    kin = RM65Kinematics(ip=ip, port=port)
    kin.connect()
    try:
        code, joints = kin._robot.rm_get_init_pose()
        if code != 0:
            raise RuntimeError(f"读取控制器初始位姿失败, code={code}")
        return [float(q) for q in joints[:6]]
    finally:
        kin.disconnect()


def _read_current_joints_deg(ip: str, port: int) -> list[float]:
    from sendrobot.send_robot.rm_ik import RM65Kinematics

    kin = RM65Kinematics(ip=ip, port=port)
    kin.connect()
    try:
        return kin.get_current_joints_deg()
    finally:
        kin.disconnect()


def _resolve_target_joints(args: argparse.Namespace) -> tuple[list[float], str]:
    if args.joints_deg:
        return list(args.joints_deg), "命令行 --joints-deg"

    if HOME_CONFIG.exists():
        cfg = _load_yaml(HOME_CONFIG)
        if cfg.get("use_custom_home") and cfg.get("home_joints_deg"):
            joints = [float(q) for q in cfg["home_joints_deg"][:6]]
            return joints, f"配置文件 {HOME_CONFIG.name}"

    joints = _read_controller_init_joints_deg(args.ip, args.port)
    return joints, "控制器 rm_get_init_pose（示教器初始位可能不同）"


def main() -> int:
    parser = argparse.ArgumentParser(description="RM65 缓慢回到初始关节位姿")
    parser.add_argument("--ip", default="192.168.73.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--speed",
        type=int,
        default=2,
        help="movej 速度（越小越慢，建议 1~5）",
    )
    parser.add_argument(
        "--joints-deg",
        type=float,
        nargs=6,
        help="手动指定 6 关节角（度）",
    )
    parser.add_argument(
        "--max-delta-deg",
        type=float,
        default=90.0,
        help="单关节最大允许差值（度）；超限则拒绝下发，防撞桌。可用 --force 强制。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略 --max-delta-deg 保护（危险，仅示教空间确认无遮挡时用）",
    )
    args = parser.parse_args()

    joints_deg, source = _resolve_target_joints(args)
    current_deg = _read_current_joints_deg(args.ip, args.port)
    max_delta = max(abs(a - b) for a, b in zip(current_deg, joints_deg))

    print(f"位姿来源: {source}")
    print(f"当前关节角 (deg): {[round(q, 2) for q in current_deg]}")
    print(f"目标关节角 (deg): {[round(q, 2) for q in joints_deg]}")
    print(f"最大关节差: {max_delta:.2f}°")

    if max_delta < 0.5:
        print("已在目标附近（差值 < 0.5°），几乎不会移动。若姿态不对，请改用自定义 home 配置。")
    if source.startswith("控制器"):
        print(
            "提示: 控制器 init_pose 可能与录制就绪位（rm65_ready_pose.json）不同。\n"
            "      日常回录制起始位请用 rm65_ready.py / 保持 home 与 ready 一致。"
        )

    if max_delta > args.max_delta_deg and not args.force:
        print(
            f"错误: 最大关节差 {max_delta:.1f}° > {args.max_delta_deg:.0f}°，"
            "拒绝自动 movej（关节插值可能扫过桌面）。\n"
            "请示教器手动挪近目标，或确认无遮挡后加 --force。",
            file=sys.stderr,
        )
        return 2

    print(f"movej speed={args.speed}, block=True")

    rclpy.init()
    node = rclpy.create_node("rm65_go_home")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(Movej, "/rm_driver/movej_cmd", qos)

    msg = Movej()
    msg.joint = [math.radians(q) for q in joints_deg]
    msg.speed = int(args.speed)
    msg.block = True
    msg.trajectory_connect = 0
    msg.dof = 6

    for _ in range(50):
        if pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.05)

    if pub.get_subscription_count() == 0:
        print("错误: rm_driver 未订阅 /rm_driver/movej_cmd，请先启动 rm_driver。", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    pub.publish(msg)
    print("已发送回零命令，请等待机械臂到位...")
    time.sleep(1.0)
    rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
