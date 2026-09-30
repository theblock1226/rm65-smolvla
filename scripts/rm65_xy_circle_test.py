#!/usr/bin/env python3
"""
RM65 XY 平面画圆测试（半径固定 20cm）。

流程：
1. 离线预检：全圆逆解 + 规划固定 Z 的笛卡尔路点
2. 断开 SDK，仅经 rm_driver 发令（避免与 rm_driver 抢 TCP）
3. movej 移到「画圆就绪」关节位
4. 圆轨迹全部使用 movel（笛卡尔直线），保证 TCP 在固定 Z 平面内运动
5. 每段运动后读 TCP，校验 Z 偏差

注：旧版用 movej_p 逐点到位，路径在关节空间规划，肉眼可见 Z 漂移与三维乱动，不是真圆。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Armstate, Movec, Movej, Movel
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Empty

RM_IK_PATH = Path(__file__).resolve().parents[1] / "sendrobot" / "send_robot" / "rm_ik.py"

CIRCLE_READY_JOINTS_DEG = [0.0, 20.0, 70.0, 10.0, -20.0, 0.0]
CIRCLE_CENTER_OFFSET_X_M = 0.10
CIRCLE_RADIUS_M = 0.20
DEFAULT_Z_TOLERANCE_MM = 5.0


def _load_kinematics():
    spec = importlib.util.spec_from_file_location("rm_ik", RM_IK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.RM65Kinematics


def _pose_from_sdk(sdk_pose: list[float]) -> Pose:
    pose = Pose()
    pose.position.x = float(sdk_pose[0])
    pose.position.y = float(sdk_pose[1])
    pose.position.z = float(sdk_pose[2])
    quat = Rotation.from_euler("xyz", sdk_pose[3:6]).as_quat()
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def _pose_with_xyz_orient(base: Pose, x: float, y: float, z: float) -> Pose:
    out = copy.deepcopy(base)
    out.position.x = float(x)
    out.position.y = float(y)
    out.position.z = float(z)
    return out


def _pos_str(pose: Pose) -> str:
    return f"({pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f})"


def plan_circle_waypoints(
    kin,
    joints_deg: list[float],
    radius_m: float,
    center_offset_x: float,
    num_points: int,
) -> tuple[Pose, list[Pose], tuple[float, float, float]]:
    fk = kin._robot.rm_algo_forward_kinematics(list(joints_deg))
    cx = fk[0] + center_offset_x
    cy, cz = fk[1], fk[2]
    orient_pose = _pose_from_sdk(fk)

    ref = list(joints_deg)
    waypoints: list[Pose] = []
    for i in range(num_points + 1):
        theta = 2.0 * math.pi * i / num_points
        x = cx + radius_m * math.cos(theta)
        y = cy + radius_m * math.sin(theta)
        sdk_pose = [x, y, cz, *fk[3:6]]
        q = kin.try_inverse_kinematics(sdk_pose, ref)
        if q is None:
            raise RuntimeError(
                f"圆轨迹逆解失败 θ={math.degrees(theta):.0f}° pos=({x:.3f},{y:.3f},{cz:.3f})"
            )
        ref = q
        waypoints.append(_pose_with_xyz_orient(orient_pose, x, y, cz))
    return orient_pose, waypoints, (cx, cy, cz)


class XYCircleTester(Node):
    def __init__(self, speed: int, move_timeout_s: float, state_timeout_s: float):
        super().__init__("rm65_xy_circle_test")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._speed = int(speed)
        self._move_timeout_s = move_timeout_s
        self._state_timeout_s = state_timeout_s
        self._last_ml_ok: bool | None = None
        self._last_mc_ok: bool | None = None
        self._last_j_ok: bool | None = None
        self._arm_state: Armstate | None = None
        self._ml_pub = self.create_publisher(Movel, "/rm_driver/movel_cmd", qos)
        self._mc_pub = self.create_publisher(Movec, "/rm_driver/movec_cmd", qos)
        self._j_pub = self.create_publisher(Movej, "/rm_driver/movej_cmd", qos)
        self._state_cmd_pub = self.create_publisher(
            Empty, "/rm_driver/get_current_arm_state_cmd", qos
        )
        self.create_subscription(Bool, "/rm_driver/movel_result", self._ml_cb, qos)
        self.create_subscription(Bool, "/rm_driver/movec_result", self._mc_cb, qos)
        self.create_subscription(Bool, "/rm_driver/movej_result", self._j_cb, qos)
        self.create_subscription(
            Armstate, "/rm_driver/get_current_arm_state_result", self._state_cb, qos
        )

    def _ml_cb(self, msg: Bool) -> None:
        self._last_ml_ok = bool(msg.data)

    def _mc_cb(self, msg: Bool) -> None:
        self._last_mc_ok = bool(msg.data)

    def _j_cb(self, msg: Bool) -> None:
        self._last_j_ok = bool(msg.data)

    def _state_cb(self, msg: Armstate) -> None:
        self._arm_state = msg

    def _wait_subscribers(self) -> None:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if (
                self._ml_pub.get_subscription_count() > 0
                and self._j_pub.get_subscription_count() > 0
            ):
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("rm_driver 未就绪，请先启动 rm_driver")

    def _wait_flag(self, attr: str, label: str) -> None:
        deadline = time.time() + self._move_timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            val = getattr(self, attr)
            if val is not None:
                if not val:
                    raise RuntimeError(f"运动失败: {label}")
                self.get_logger().info(f"到位: {label}")
                return
        raise RuntimeError(f"超时: {label}")

    def read_tcp(self) -> tuple[float, float, float] | None:
        self._arm_state = None
        self._state_cmd_pub.publish(Empty())
        deadline = time.time() + self._state_timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._arm_state is not None:
                p = self._arm_state.pose.position
                return float(p.x), float(p.y), float(p.z)
        self.get_logger().warning("读取 TCP 位姿超时，跳过本步 Z 校验")
        return None

    def movej_deg(self, joints_deg: list[float], label: str) -> None:
        msg = Movej()
        msg.joint = [math.radians(q) for q in joints_deg]
        msg.speed = self._speed
        msg.block = True
        msg.trajectory_connect = 0
        msg.dof = 6
        self._last_j_ok = None
        self.get_logger().info(f"movej -> {label}")
        self._j_pub.publish(msg)
        self._wait_flag("_last_j_ok", label)

    def movel(self, target: Pose, label: str) -> None:
        msg = Movel()
        msg.pose = copy.deepcopy(target)
        msg.speed = self._speed
        msg.trajectory_connect = 0
        msg.block = True
        self._last_ml_ok = None
        self.get_logger().info(f"movel -> {label} {_pos_str(target)}")
        self._ml_pub.publish(msg)
        self._wait_flag("_last_ml_ok", label)

    def movec(self, mid: Pose, end: Pose, label: str) -> None:
        msg = Movec()
        msg.pose_mid = copy.deepcopy(mid)
        msg.pose_end = copy.deepcopy(end)
        msg.speed = self._speed
        msg.trajectory_connect = 0
        msg.block = True
        msg.loop = 0
        self._last_mc_ok = None
        self.get_logger().info(
            f"movec -> {label} mid={_pos_str(mid)} end={_pos_str(end)}"
        )
        self._mc_pub.publish(msg)
        self._wait_flag("_last_mc_ok", label)

    def check_z(self, label: str, target_z: float, tolerance_mm: float) -> float:
        tcp = self.read_tcp()
        if tcp is None:
            return 0.0
        _, _, z = tcp
        dz_mm = abs(z - target_z) * 1000.0
        if dz_mm > tolerance_mm:
            self.get_logger().warning(
                f"{label}: Z 偏差 {dz_mm:.1f}mm (目标 Z={target_z:.4f}, 实际 Z={z:.4f})"
            )
        return dz_mm


def run_xy_circle_movel(
    tester: XYCircleTester,
    ready_joints: list[float],
    waypoints: list[Pose],
    center: tuple[float, float, float],
    radius_m: float,
    num_points: int,
    return_joints: list[float] | None,
    z_tolerance_mm: float,
) -> float:
    cx, cy, cz = center
    print(f"圆心: ({cx:.4f}, {cy:.4f}, {cz:.4f})  半径: {radius_m*100:.0f}cm  共 {len(waypoints)} 点")
    print(f"就绪关节 (deg): {[round(q, 2) for q in ready_joints]}")
    print("轨迹模式: movel（笛卡尔直线，固定 Z）")

    tester.movej_deg(ready_joints, "移到画圆就绪位")
    max_dz_mm = 0.0

    tester.movel(waypoints[0], "移到圆起点 θ=0°")
    max_dz_mm = max(max_dz_mm, tester.check_z("圆起点", cz, z_tolerance_mm))

    for i, wp in enumerate(waypoints[1:], start=1):
        theta = 360.0 * i / num_points
        tester.movel(wp, f"圆轨迹 {i}/{num_points} θ={theta:.0f}°")
        dz_mm = tester.check_z(f"θ={theta:.0f}°", cz, z_tolerance_mm)
        max_dz_mm = max(max_dz_mm, dz_mm)

    tester.movel(waypoints[0], "回圆起点")
    max_dz_mm = max(max_dz_mm, tester.check_z("回起点", cz, z_tolerance_mm))

    if return_joints is not None:
        tester.movej_deg(return_joints, "回初始关节位")

    print(f"XY 平面画圆完成。最大 Z 偏差: {max_dz_mm:.1f}mm（容差 {z_tolerance_mm:.0f}mm）")
    return max_dz_mm


def run_xy_circle_movec(
    tester: XYCircleTester,
    ready_joints: list[float],
    waypoints: list[Pose],
    center: tuple[float, float, float],
    radius_m: float,
    num_points: int,
    return_joints: list[float] | None,
    z_tolerance_mm: float,
) -> float:
    cx, cy, cz = center
    half = num_points // 2
    print(f"圆心: ({cx:.4f}, {cy:.4f}, {cz:.4f})  半径: {radius_m*100:.0f}cm")
    print("轨迹模式: movec 两段半圆")

    tester.movej_deg(ready_joints, "移到画圆就绪位")
    max_dz_mm = 0.0

    tester.movel(waypoints[0], "移到圆起点 θ=0°")
    max_dz_mm = max(max_dz_mm, tester.check_z("圆起点", cz, z_tolerance_mm))

    tester.movec(waypoints[half], waypoints[num_points], "半圆 0°→180°")
    max_dz_mm = max(max_dz_mm, tester.check_z("180°", cz, z_tolerance_mm))

    tester.movec(waypoints[half + half // 2], waypoints[0], "半圆 180°→360°")
    max_dz_mm = max(max_dz_mm, tester.check_z("360°", cz, z_tolerance_mm))

    if return_joints is not None:
        tester.movej_deg(return_joints, "回初始关节位")

    print(f"XY 平面画圆完成。最大 Z 偏差: {max_dz_mm:.1f}mm")
    return max_dz_mm


def main() -> int:
    parser = argparse.ArgumentParser(description="RM65 XY 平面 20cm 画圆（movel）")
    parser.add_argument("--ip", default="192.168.73.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--radius", type=float, default=CIRCLE_RADIUS_M)
    parser.add_argument("--speed", type=int, default=3, help="1~100，越小越慢")
    parser.add_argument("--circle-points", type=int, default=72, help="圆分段数，越多越平滑")
    parser.add_argument("--center-offset-x", type=float, default=CIRCLE_CENTER_OFFSET_X_M)
    parser.add_argument("--move-timeout", type=float, default=180.0)
    parser.add_argument("--z-tolerance-mm", type=float, default=DEFAULT_Z_TOLERANCE_MM)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("movel", "movec"),
        default="movel",
        help="movel=逐段直线; movec=两段半圆（实验）",
    )
    parser.add_argument(
        "--return-to-start-joints",
        action="store_true",
        help="画圆后回到脚本启动时的关节位",
    )
    args = parser.parse_args()

    if abs(args.radius - 0.20) > 1e-6:
        print("警告: 本测试约定半径 20cm，已强制使用 0.20m", file=sys.stderr)
    radius_m = 0.20

    kin = _load_kinematics()()
    kin.connect()
    try:
        start_joints = kin.get_current_joints_deg()
        _, waypoints, center = plan_circle_waypoints(
            kin,
            CIRCLE_READY_JOINTS_DEG,
            radius_m,
            args.center_offset_x,
            args.circle_points,
        )
        seg_len_m = 2.0 * math.pi * radius_m / args.circle_points
        print(
            f"预检: 全圆 {args.circle_points} 点逆解 OK，"
            f"相邻段弧长约 {seg_len_m*100:.1f}cm"
        )
        if args.dry_run:
            return 0
    finally:
        kin.disconnect()
        time.sleep(0.3)

    rclpy.init()
    tester = XYCircleTester(
        speed=args.speed,
        move_timeout_s=args.move_timeout,
        state_timeout_s=5.0,
    )
    try:
        tester._wait_subscribers()
        for sec in range(args.countdown, 0, -1):
            print(f">>> {sec}s 后开始画圆（movel 笛卡尔），请确认半径 20cm 范围内无障碍物...")
            time.sleep(1.0)
        return_joints = start_joints if args.return_to_start_joints else None
        if args.mode == "movec":
            max_dz = run_xy_circle_movec(
                tester,
                CIRCLE_READY_JOINTS_DEG,
                waypoints,
                center,
                radius_m,
                args.circle_points,
                return_joints,
                args.z_tolerance_mm,
            )
        else:
            max_dz = run_xy_circle_movel(
                tester,
                CIRCLE_READY_JOINTS_DEG,
                waypoints,
                center,
                radius_m,
                args.circle_points,
                return_joints,
                args.z_tolerance_mm,
            )
        if max_dz > args.z_tolerance_mm:
            print(
                f"警告: 最大 Z 偏差 {max_dz:.1f}mm 超过容差 {args.z_tolerance_mm:.0f}mm",
                file=sys.stderr,
            )
            return 2
        return 0
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
