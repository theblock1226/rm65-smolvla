#!/usr/bin/env python3
"""
SendRobot 末端笛卡尔遥操：GamepadTeleop 的 delta_x/y/z/gripper -> IK -> 6 关节角 -> rm/sent_actions
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.common.errors import DeviceNotConnectedError
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from .config_send_robot import SendRobotEndEffectorConfig
from .rm_ik import RM65Kinematics
from .send_robot import SendRobot

logger = logging.getLogger(__name__)

GRIPPER_CLOSE = 0
GRIPPER_STAY = 1
GRIPPER_OPEN = 2


class SendRobotEndEffector(SendRobot):
    config_class = SendRobotEndEffectorConfig
    name = "send_robot_end_effector"

    def __init__(self, config: SendRobotEndEffectorConfig):
        super().__init__(config)

        self._kinematics = RM65Kinematics(ip=config.rm_ip, port=config.rm_port)
        self._last_joint_deg: list[float] | None = None
        self._last_published_action: dict[str, float] | None = None

        if self.action_publisher is None:
            qos = getattr(self, "_action_qos", None) or QoSProfile(
                depth=10, reliability=ReliabilityPolicy.RELIABLE
            )
            self.action_publisher = self.create_publisher(JointState, "rm/sent_actions", qos)

    def connect(self) -> None:
        import os

        super().connect()
        # 遥操默认不连 SDK，避免与 record_state_test 抢 TCP 导致臂停/夹爪失效
        use_sdk_ik = os.environ.get("TELEOP_USE_SDK_IK", "false").lower() in ("1", "true", "yes")
        self._sdk_connected = False
        if not getattr(self.config, "joint_space_teleop", False) and use_sdk_ik:
            self._kinematics.connect()
            self._sdk_connected = True
        self._wait_for_joint_feedback()
        self._last_joint_deg = self._current_joint_angles_deg()
        if self._sdk_connected:
            self._warn_if_near_singularity()

    def _wait_for_joint_feedback(self, timeout_s: float = 10.0) -> None:
        import time

        import rclpy

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.joint_states is not None:
                logger.info("已收到 rm/recorded_joint_states 反馈")
                return
            if rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
            else:
                time.sleep(0.05)
        if getattr(self.config, "joint_space_teleop", False):
            raise RuntimeError(
                "未收到 rm/recorded_joint_states。"
                "请先运行: /home/sdznkj/zkx_ws/sendlerobot/scripts/start_rm65_teleop_stack.sh"
            )
        logger.warning("未收到 rm/recorded_joint_states，将使用 SDK 读关节")

    def _warn_if_near_singularity(self) -> None:
        try:
            joints = self._last_joint_deg or self._current_joint_angles_deg()
            code = self._kinematics._robot.rm_algo_universal_singularity_analyse(joints, 0.01)
            if code != 0:
                logger.warning(
                    "当前姿态接近奇异/伸直位（singularity=%s）。"
                    "已默认关节空间遥操；若仍异常请先移到肘部弯曲起始位。",
                    code,
                )
        except Exception as exc:
            logger.debug("奇异点检测跳过: %s", exc)

    def disconnect(self) -> None:
        if getattr(self, "_sdk_connected", False):
            self._kinematics.disconnect()
        self._last_joint_deg = None
        self._last_published_action = None
        super().disconnect()

    def _current_joint_angles_deg(self) -> list[float]:
        sdk_joints: list[float] | None = None
        if getattr(self, "_sdk_connected", False):
            try:
                sdk_joints = self._kinematics.get_current_joints_deg()
            except (RuntimeError, AssertionError) as exc:
                logger.debug("SDK 读关节失败: %s", exc)
                sdk_joints = None

        if self.joint_states is not None:
            try:
                ros_joints = [float(self.joint_states[f"left_joint_{i}.pos"]) for i in range(1, 7)]
                # /joint_states 常全 0（UDP 未通），此时改用 SDK
                if max(abs(v) for v in ros_joints) < 0.01 and sdk_joints is not None:
                    return sdk_joints
                return ros_joints
            except KeyError:
                pass

        if sdk_joints is not None:
            return sdk_joints
        raise RuntimeError(
            "无法读取关节角。请确认 start_rm65_teleop_stack.sh 已启动且 rm/recorded_joint_states 有数据。"
        )

    def _apply_gripper_command(self, gripper_cmd: int | float) -> float:
        """RB/X 按住时直接设目标开口度（非增量），便于夹爪桥接识别。"""
        cmd = int(gripper_cmd)
        if cmd == GRIPPER_CLOSE:
            self._gripper_pos = 0.0
        elif cmd == GRIPPER_OPEN:
            self._gripper_pos = 1.0
        return self._gripper_pos

    def _clip_position(self, pose: list[float]) -> list[float]:
        bounds = self.config.end_effector_bounds
        pose[0] = float(np.clip(pose[0], bounds["min"][0], bounds["max"][0]))
        pose[1] = float(np.clip(pose[1], bounds["min"][1], bounds["max"][1]))
        pose[2] = float(np.clip(pose[2], bounds["min"][2], bounds["max"][2]))
        return pose

    def _clamp_joint_step(self, target_deg: list[float], ref_deg: list[float]) -> list[float]:
        max_step = float(getattr(self.config, "max_joint_step_deg", 0.15))
        deltas = [target - ref for target, ref in zip(target_deg, ref_deg)]
        max_abs = max((abs(d) for d in deltas), default=0.0)
        if max_abs > max_step:
            scale = max_step / max_abs
            deltas = [d * scale for d in deltas]
        return [ref + delta for ref, delta in zip(ref_deg, deltas)]

    def _joint_space_fallback(self, action: dict[str, Any], ref_joints: list[float]) -> list[float]:
        """摇杆 -> 关节增量（度），双向对称。"""
        max_step = float(getattr(self.config, "max_joint_step_deg", 0.15))
        sx = float(action.get("delta_x", 0.0))
        sy = float(action.get("delta_y", 0.0))
        sz = float(action.get("delta_z", 0.0))
        joints = list(ref_joints)
        joints[0] += (-sy) * max_step  # 左摇杆左右 -> J1
        joints[1] += (-sx) * max_step  # 左摇杆前后 -> J2
        joints[2] += (-sz) * max_step  # 右摇杆 -> J3
        return joints

    def _delta_to_joint_action(self, action: dict[str, Any]) -> dict[str, float] | None:
        step = self.config.end_effector_step_sizes
        speed = float(getattr(self.config, "teleop_speed_scale", 1.0))
        dx = float(action["delta_x"]) * step["x"] * speed
        dy = float(action["delta_y"]) * step["y"] * speed
        dz = float(action["delta_z"]) * step["z"] * speed
        gripper_cmd = int(action.get("gripper", GRIPPER_STAY))
        armed = float(action.get("enable", 0.0)) >= 0.5
        stick_eps = float(getattr(self.config, "stick_motion_threshold", 0.12))
        raw_motion = max(
            abs(float(action.get("delta_x", 0.0))),
            abs(float(action.get("delta_y", 0.0))),
            abs(float(action.get("delta_z", 0.0))),
        )
        has_motion = raw_motion > stick_eps
        has_gripper = self.config.use_gripper and gripper_cmd != GRIPPER_STAY
        if not has_motion and not has_gripper:
            return None

        if (
            has_motion
            and getattr(self.config, "require_enable_button", True)
            and not armed
        ):
            if not has_gripper:
                return None
            has_motion = False

        gripper_val: float | None = None
        if has_gripper:
            gripper_val = self._apply_gripper_command(gripper_cmd)

        if not has_motion:
            if gripper_val is not None:
                return {"left_gripper.pos": gripper_val}
            return None

        ref_joints = self._current_joint_angles_deg()

        if getattr(self.config, "joint_space_teleop", False):
            joint_deg = self._joint_space_fallback(action, ref_joints)
            joint_deg = self._clamp_joint_step(joint_deg, ref_joints)
            self._last_joint_deg = joint_deg
        elif has_motion:
            if not getattr(self, "_sdk_connected", False):
                logger.debug("未连接 SDK，笛卡尔逆解不可用，使用关节空间")
                joint_deg = self._joint_space_fallback(action, ref_joints)
            else:
                current_pose = self._kinematics.get_current_pose()
                if getattr(self.config, "cartesian_frame", "work") == "tool":
                    target_pose = self._kinematics.tool_delta_to_pose(ref_joints, dx, dy, dz)
                    ik_base_pose = current_pose
                else:
                    target_pose = self._kinematics.work_delta_to_pose(current_pose, dx, dy, dz)
                    ik_base_pose = current_pose

                try:
                    target_pose = self._clip_position(list(target_pose))
                    joint_deg = self._kinematics.try_inverse_kinematics(
                        target_pose,
                        ref_joints,
                        base_pose=ik_base_pose,
                        deltas=(dx, dy, dz),
                    )
                except Exception as exc:
                    logger.warning("逆解计算失败: %s", exc)
                    joint_deg = None

                if joint_deg is None:
                    logger.debug("逆解失败，使用关节空间回退")
                    joint_deg = self._joint_space_fallback(action, ref_joints)
            joint_deg = self._clamp_joint_step(joint_deg, ref_joints)
            self._last_joint_deg = joint_deg
        else:
            joint_deg = ref_joints

        joint_action = {f"left_joint_{i}.pos": joint_deg[i - 1] for i in range(1, 7)}
        if gripper_val is not None:
            joint_action["left_gripper.pos"] = gripper_val
        return joint_action

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        if self.config.use_gripper:
            obs["left_gripper.pos"] = self._gripper_pos
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if "delta_x" in action and "delta_y" in action and "delta_z" in action:
            raise RuntimeError(
                "send_robot_end_effector 旧遥操路径已禁用 "
                "(关节空间 / IK→关节→sent_actions)。"
                "请使用: /home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_movep_teleop.sh"
            )

        if self.config.use_gripper and "left_gripper.pos" not in action:
            action = dict(action)
            action["left_gripper.pos"] = self._gripper_pos

        result = super().send_action(action)
        self._last_published_action = result
        return result

    def reset(self) -> None:
        self._last_joint_deg = None
        self._last_published_action = None
        self._gripper_pos = self.config.initial_gripper_pos
        self._kalman_initialized = False
