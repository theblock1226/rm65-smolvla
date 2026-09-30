#!/usr/bin/env python3
"""RM65 joint follow bridge: rm/sent_actions → /rm_driver/movej_canfd_cmd.

Replaces deleted play_action_test.py. Applies per-tick rate limit matching
``SendRobotEndEffectorConfig.max_joint_step_deg`` (default 0.4°) and an extra
``speed_scale`` (default 0.3) for first real-policy sessions.

Joint values on ``rm/sent_actions`` / ``rm/recorded_joint_states`` are **degrees**.
``Jointpos`` for movej_canfd expects **radians** (rm_driver multiplies by RAD_DEGREE).

Gripper is NOT handled here — keep ``record_state_test.py`` running for Modbus.
"""

from __future__ import annotations

import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import JointState

JOINT_NAMES = [f"left_joint_{i}" for i in range(1, 7)]
DEG2RAD = math.pi / 180.0


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class PlayActionRM65(Node):
    def __init__(self) -> None:
        super().__init__("play_action_rm65")
        self.max_joint_step_deg = _env_float("RM65_MAX_JOINT_STEP_DEG", 0.4)
        self.speed_scale = _env_float("RM65_SPEED_SCALE", 0.3)
        if self.speed_scale <= 0 or self.speed_scale > 1.0:
            raise ValueError(f"RM65_SPEED_SCALE must be in (0, 1], got {self.speed_scale}")
        if self.max_joint_step_deg <= 0:
            raise ValueError("RM65_MAX_JOINT_STEP_DEG must be positive")

        self._current_deg: list[float] | None = None
        self._last_cmd_deg: list[float] | None = None
        self._pub_count = 0
        self._clip_count = 0
        self._last_log = 0.0

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub = self.create_publisher(Jointpos, "/rm_driver/movej_canfd_cmd", qos)
        self.create_subscription(JointState, "rm/recorded_joint_states", self._on_state, qos)
        self.create_subscription(JointState, "rm/sent_actions", self._on_action, qos)

        self.get_logger().info(
            f"play_action_rm65 ready | max_step={self.max_joint_step_deg}° "
            f"speed_scale={self.speed_scale} → movej_canfd"
        )

    def _lookup_joints_deg(self, msg: JointState) -> list[float] | None:
        by_name = {n: float(p) for n, p in zip(msg.name, msg.position)}
        if all(n in by_name for n in JOINT_NAMES):
            return [by_name[n] for n in JOINT_NAMES]
        # Fallback: first 6 positions if names missing
        if len(msg.position) >= 6:
            return [float(v) for v in msg.position[:6]]
        return None

    def _on_state(self, msg: JointState) -> None:
        joints = self._lookup_joints_deg(msg)
        if joints is not None:
            self._current_deg = joints

    def _rate_limit(self, desired: list[float], current: list[float]) -> tuple[list[float], bool]:
        max_step = self.max_joint_step_deg * self.speed_scale
        out: list[float] = []
        clipped = False
        for d, c in zip(desired, current):
            delta = d - c
            if abs(delta) > max_step:
                clipped = True
                delta = math.copysign(max_step, delta)
            out.append(c + delta)
        return out, clipped

    def _on_action(self, msg: JointState) -> None:
        desired = self._lookup_joints_deg(msg)
        if desired is None:
            return
        ref = self._last_cmd_deg if self._last_cmd_deg is not None else self._current_deg
        if ref is None:
            # No feedback yet — hold off rather than jump
            return

        cmd, clipped = self._rate_limit(desired, ref)
        if clipped:
            self._clip_count += 1
        self._last_cmd_deg = cmd

        out = Jointpos()
        out.joint = [float(v) * DEG2RAD for v in cmd]
        out.follow = True
        out.expand = 0.0
        out.dof = 6
        self._pub.publish(out)
        self._pub_count += 1

        now = time.time()
        if now - self._last_log >= 1.0:
            self._last_log = now
            max_delta = max(abs(a - b) for a, b in zip(cmd, ref)) if ref else 0.0
            self.get_logger().info(
                f"follow pubs={self._pub_count} clips={self._clip_count} "
                f"step≈{max_delta:.3f}° scale={self.speed_scale} "
                f"cmd={[round(v, 1) for v in cmd]}"
            )


def main() -> None:
    rclpy.init()
    node = PlayActionRM65()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"shutdown pubs={node._pub_count} clips={node._clip_count}"
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
