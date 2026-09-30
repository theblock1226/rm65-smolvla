#!/usr/bin/env python3
"""发布 rm/recorded_joint_states，并复用同一 SDK 连接驱动大寰夹爪 Modbus。"""

from __future__ import annotations

import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from sendrobot.send_robot.dahuan_gripper import DahuanGripperModbus
from sendrobot.send_robot.rm_ik import RM65Kinematics


class JointStateRecorder(Node):
    GRIPPER_NAME = "left_gripper"

    def __init__(self) -> None:
        super().__init__("full_command_recorder")

        self.last_log_time = 0.0
        self.joint_state: list[float] | None = None
        self._ros_joint_state: list[float] | None = None
        self._sdk_ok = False
        self._gripper: DahuanGripperModbus | None = None
        self._gripper_min_delta = float(os.environ.get("GRIPPER_MIN_DELTA", "0.02"))
        # prep 开爪后由此种子，避免首帧不发布夹爪、SafeActionGate 当成 0
        init_g = os.environ.get("RM65_INITIAL_GRIPPER_POS", "").strip()
        self._last_gripper_norm: float | None = float(init_g) if init_g else None
        if self._last_gripper_norm is not None:
            self.get_logger().info(f"夹爪状态种子 norm={self._last_gripper_norm:.2f}")

        rm_ip = os.environ.get("RM_IP", "192.168.73.18")
        rm_port = int(os.environ.get("RM_PORT", "8080"))
        self._kin = RM65Kinematics(ip=rm_ip, port=rm_port)

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.joint_state_pub = self.create_publisher(JointState, "rm/recorded_joint_states", qos_profile)
        self.create_subscription(JointState, "/joint_states", self._ros_callback, qos_profile)
        self.create_subscription(JointState, "rm/sent_actions", self._action_callback, qos_profile)
        self.create_timer(0.02, self._timer_callback)  # 50 Hz

        try:
            self._kin.connect()
            self._sdk_ok = True
            self.get_logger().info(f"关节反馈：使用 RM SDK ({rm_ip}:{rm_port})")
            if os.environ.get("GRIPPER_MODBUS", "true").lower() in ("1", "true", "yes"):
                if self._kin.robot is not None:
                    self._gripper = DahuanGripperModbus(self._kin.robot)
                    self._gripper.setup(self.get_logger())
                    self.get_logger().info("大寰夹爪 Modbus 已挂载（与关节反馈共用 SDK）")
        except Exception as exc:
            self.get_logger().error(f"RM SDK 连接失败，将尝试 /joint_states: {exc}")

    def _ros_callback(self, msg: JointState) -> None:
        joint_name_to_index = {
            "joint1": 0,
            "joint2": 1,
            "joint3": 2,
            "joint4": 3,
            "joint5": 4,
            "joint6": 5,
        }
        joints = [0.0] * 6
        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                joints[joint_name_to_index[name]] = math.degrees(float(position))
        if max(abs(v) for v in joints) > 0.01:
            self._ros_joint_state = joints

    def _read_joints_deg(self) -> list[float] | None:
        if self._sdk_ok:
            try:
                return self._kin.get_current_joints_deg()
            except Exception as exc:
                self.get_logger().warning(f"SDK 读关节失败: {exc}")
        return self._ros_joint_state

    @staticmethod
    def _lookup_gripper(msg: JointState) -> float | None:
        for name, position in zip(msg.name, msg.position):
            if name == JointStateRecorder.GRIPPER_NAME:
                return float(position)
        return None

    def _action_callback(self, msg: JointState) -> None:
        if self._gripper is None:
            return
        normalized = self._lookup_gripper(msg)
        if normalized is None:
            return
        if (
            self._last_gripper_norm is not None
            and abs(normalized - self._last_gripper_norm) < self._gripper_min_delta
        ):
            return
        ok, hw = self._gripper.set_position_normalized(normalized, self._gripper_min_delta)
        if ok and hw is not None:
            self.get_logger().info(f"夹爪 -> position={hw} (norm={normalized:.2f})")
        elif not ok:
            self.get_logger().warning(f"夹爪 Modbus 写入失败 norm={normalized:.2f}")
        self._last_gripper_norm = normalized

    def _timer_callback(self) -> None:
        joints = self._read_joints_deg()
        if joints is None:
            return
        self.joint_state = joints

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 附带最近命令的夹爪 norm，避免 SendRobot SafeActionGate 把缺失夹爪当成 0
        names = [f"left_joint_{i}" for i in range(1, 7)]
        positions = [float(v) for v in joints]
        if self._last_gripper_norm is not None:
            names.append(self.GRIPPER_NAME)
            positions.append(float(self._last_gripper_norm))
        msg.name = names
        msg.position = positions
        msg.velocity = [0.0] * len(names)
        msg.effort = [0.0] * len(names)
        self.joint_state_pub.publish(msg)

        now = time.time()
        if now - self.last_log_time >= 2.0:
            self.last_log_time = now
            self.get_logger().info(f"recorded joints (deg): {[round(v, 2) for v in joints]}")

    def destroy_node(self) -> None:
        if self._sdk_ok:
            self._kin.disconnect()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = JointStateRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
