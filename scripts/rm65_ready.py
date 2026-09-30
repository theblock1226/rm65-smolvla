#!/usr/bin/env python3
"""就绪位与启动位姿：读 json、movej、SDK 读当前笛卡尔位姿。"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

READY_JOINTS_DEG = [0.0, 20.0, 70.0, 10.0, -20.0, 0.0]
READY_POSE_PATH = Path(__file__).resolve().parent / "rm65_ready_pose.json"
RM_IK_PATH = Path(__file__).resolve().parents[1] / "sendrobot" / "send_robot" / "rm_ik.py"


def _load_kinematics():
    spec = importlib.util.spec_from_file_location("rm_ik", RM_IK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.RM65Kinematics


def load_ready_joints_deg(path: Path | None = None) -> list[float]:
    """从 json 读关节角；默认 rm65_ready_pose.json，失败则用内置默认。"""
    pose_path = path or READY_POSE_PATH
    try:
        from rm65_save_ready_pose import load_ready_joints_deg as _load

        return _load(pose_path)
    except Exception:
        if not pose_path.is_file():
            return list(READY_JOINTS_DEG)
        try:
            import json

            data = json.loads(pose_path.read_text(encoding="utf-8"))
            joints = [float(v) for v in data["joints_deg"][:6]]
            if len(joints) == 6:
                return joints
        except Exception as exc:
            print(f"警告: 读取就绪位失败，用默认: {exc}", file=sys.stderr)
        return list(READY_JOINTS_DEG)


def read_pose(ip: str, port: int) -> list[float]:
    kin = _load_kinematics()()
    kin.ip, kin.port = ip, port
    kin.connect()
    try:
        pose = kin.get_current_pose()
        print(
            f"初始位姿(SDK): xyz=({pose[0]:.4f},{pose[1]:.4f},{pose[2]:.4f}) "
            f"rpy=({pose[3]:.4f},{pose[4]:.4f},{pose[5]:.4f})"
        )
        return [float(v) for v in pose[:6]]
    finally:
        kin.disconnect()
        time.sleep(0.3)


def movej_ready(
    speed: int = 5,
    timeout_s: float = 120.0,
    joints_deg: list[float] | None = None,
    pose_path: Path | None = None,
) -> None:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rm_ros_interfaces.msg import Movej
    from std_msgs.msg import Bool

    node = rclpy.create_node("rm65_movep_teleop_ready")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(Movej, "/rm_driver/movej_cmd", qos)
    done = {"ok": None}

    def _cb(msg: Bool) -> None:
        done["ok"] = bool(msg.data)

    node.create_subscription(Bool, "/rm_driver/movej_result", _cb, qos)
    for _ in range(50):
        if pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        node.destroy_node()
        raise RuntimeError("rm_driver 未订阅 movej_cmd")

    if joints_deg is None:
        joints_deg = load_ready_joints_deg(pose_path)
    label = (pose_path or READY_POSE_PATH).name
    msg = Movej()
    msg.joint = [math.radians(q) for q in joints_deg]
    msg.speed = speed
    msg.block = True
    msg.trajectory_connect = 0
    msg.dof = 6
    print(f"movej → 就绪位 {[round(q, 2) for q in joints_deg]}  ({label})")
    pub.publish(msg)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if done["ok"] is not None:
            break
    node.destroy_node()
    if done["ok"] is not True:
        raise RuntimeError(f"就绪位失败: {done['ok']}")
    print("就绪位到位")
