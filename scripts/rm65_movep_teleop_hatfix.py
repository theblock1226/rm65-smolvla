#!/usr/bin/env python3
"""
RM65 笛卡尔遥操 —— 十字键防重叠试验入口。

与 rm65_movep_teleop.py 相同，只是手柄积分改用 rm65_stick_hatfix.py。
原版文件未改；请用 rm65_run_movep_teleop_hatfix.sh 启动。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from rm65_ready import (
    READY_JOINTS_DEG,
    READY_POSE_PATH,
    load_ready_joints_deg,
    movej_ready,
    read_pose,
)
from rm65_stick_hatfix import StickIntegrator, _env_bool, _make_stick
from rm65_teleop_gripper import GRIPPER_CLOSE_POS, GRIPPER_OPEN_POS, TeleopGripper

PRINT_ONLY_SEED = [-0.3, 0.0, 0.5, 0.0, 0.0, 0.0]

__all__ = [
    "GRIPPER_CLOSE_POS",
    "GRIPPER_OPEN_POS",
    "PRINT_ONLY_SEED",
    "READY_JOINTS_DEG",
    "READY_POSE_PATH",
    "StickIntegrator",
    "TeleopGripper",
    "_env_bool",
    "_make_stick",
    "load_ready_joints_deg",
    "movej_ready",
    "read_pose",
]


def _hat_log(stick: StickIntegrator) -> str:
    droll, dpitch, dyaw = stick.last_d_rpy
    return (
        f"hat={stick.last_hat} raw={stick.last_hat_raw} "
        f"lock={stick.last_hat_lock or '-'} "
        f"d_tool=("
        f"{droll * 180 / math.pi:+.2f},"
        f"{dpitch * 180 / math.pi:+.2f},"
        f"{dyaw * 180 / math.pi:+.2f})deg"
    )


def run_print_only(args: argparse.Namespace) -> int:
    stick = _make_stick(PRINT_ONLY_SEED, args)
    stick.start()
    dt = 1.0 / args.rate
    print("print-only（hatfix）：不发指令、不写夹爪。Ctrl+C 退出。")
    print("看 d_tool：上下应只有第一项，左右应只有第三项（工具系 roll/yaw）。")
    last = 0.0
    r0 = (stick.rx, stick.ry, stick.rz)
    try:
        while True:
            t0 = time.time()
            x, y, z, rx, ry, rz = stick.update()
            if stick.gripper_event == "open":
                print("夹爪事件: LB → 开(1000) [print-only 未执行]")
            elif stick.gripper_event == "close":
                print("夹爪事件: RB → 合(0) [print-only 未执行]")
            now = time.time()
            if now - last >= 0.25:
                last = now
                print(
                    f"xyz=({x:.4f},{y:.4f},{z:.4f})  "
                    f"rpy=({rx:.3f},{ry:.3f},{rz:.3f})  "
                    f"Δrpy=("
                    f"{(rx-r0[0])*180/math.pi:+.1f},"
                    f"{(ry-r0[1])*180/math.pi:+.1f},"
                    f"{(rz-r0[2])*180/math.pi:+.1f})deg  "
                    f"{_hat_log(stick)}"
                )
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stick.cleanup()
    return 0


def run_teleop(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rm_ros_interfaces.msg import Cartepos

    if args.go_ready:
        rclpy.init()
        try:
            movej_ready(speed=args.ready_speed)
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        time.sleep(0.5)

    if args.pose:
        parts = [float(v) for v in args.pose.split(",")]
        if len(parts) != 6:
            raise ValueError("--pose 需要 x,y,z,rx,ry,rz")
        seed = parts
        print(f"初始位姿(手动): {[round(v, 4) for v in seed]}")
    else:
        seed = read_pose(args.ip, args.port)

    gripper = TeleopGripper(args.ip, args.port, enabled=not args.no_gripper)
    gripper.setup()

    stick = _make_stick(seed, args)
    stick.start()

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("rm65_movep_teleop_hatfix")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
    )
    pub = node.create_publisher(Cartepos, "/rm_driver/movep_canfd_cmd", qos)
    for _ in range(50):
        if pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        stick.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("rm_driver 未订阅 movep_canfd_cmd")

    dt = 1.0 / args.rate
    print(
        f"控制环 {args.rate:g}Hz  "
        f"平移约 {args.sensitivity * args.rate * 1000:.0f} mm/s  "
        f"姿态约 {args.orient_sensitivity * args.rate * 180 / math.pi:.1f} deg/s"
    )
    print("Ctrl+C 退出。（hatfix 试验版）")
    last_log = 0.0
    x0, y0, z0 = stick.x, stick.y, stick.z
    r0 = (stick.rx, stick.ry, stick.rz)
    try:
        while True:
            t0 = time.time()
            x, y, z, rx, ry, rz = stick.update()

            if stick.gripper_event == "open":
                gripper.set_open(True)
            elif stick.gripper_event == "close":
                gripper.set_open(False)

            quat = stick.as_quat()
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])
            msg = Cartepos()
            msg.pose = pose
            msg.follow = args.follow
            pub.publish(msg)

            now = time.time()
            if now - last_log >= 0.5:
                last_log = now
                print(
                    f"xyz=({x:.4f},{y:.4f},{z:.4f})  "
                    f"rpy=({rx:.3f},{ry:.3f},{rz:.3f})  "
                    f"Δxyz=("
                    f"{(x-x0)*1000:.0f},"
                    f"{(y-y0)*1000:.0f},"
                    f"{(z-z0)*1000:.0f})mm  "
                    f"Δrpy=("
                    f"{(rx-r0[0])*180/math.pi:+.1f},"
                    f"{(ry-r0[1])*180/math.pi:+.1f},"
                    f"{(rz-r0[2])*180/math.pi:+.1f})deg  "
                    f"{_hat_log(stick)}"
                )
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stick.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RM65 movep 侧装遥操（十字键 hatfix）")
    p.add_argument("--ip", default=os.environ.get("RM_IP", "192.168.73.18"))
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument(
        "--sensitivity",
        type=float,
        default=float(os.environ.get("TELEOP_SENSITIVITY", "0.001")),
    )
    p.add_argument(
        "--orient-sensitivity",
        type=float,
        default=float(os.environ.get("TELEOP_ORIENT_SENSITIVITY", "0.003")),
        help="姿态每帧 rad，默认 0.003 ≈ 8.6deg/s @50Hz（偏慢更稳）",
    )
    p.add_argument(
        "--deadzone",
        type=float,
        default=float(os.environ.get("GAMEPAD_DEADZONE", "0.15")),
    )
    p.add_argument("--follow", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--go-ready", action="store_true")
    p.add_argument(
        "--no-gripper",
        action="store_true",
        help="禁用夹爪（不连 SDK Modbus）",
    )
    p.add_argument("--ready-speed", type=int, default=5)
    p.add_argument("--pose", default="")
    p.add_argument("--x-min", type=float, default=-0.8)
    p.add_argument("--x-max", type=float, default=0.0)
    p.add_argument("--y-min", type=float, default=-0.5)
    p.add_argument("--y-max", type=float, default=0.5)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.85)
    args = p.parse_args()
    try:
        if args.print_only:
            return run_print_only(args)
        return run_teleop(args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
