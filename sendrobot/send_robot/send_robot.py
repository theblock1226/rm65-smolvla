#!/usr/bin/env python

import logging
import os
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from sendrobot.send_robot.config_send_robot import SendRobotConfig

from ..robot import Robot

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveKalmanFilter:
    def __init__(self, dim, process_variance, measurement_variance, threshold=5.0, scale_factor=10.0):
        """
        自适应卡尔曼过滤器。
        Args:
            dim (int): 状态变量的维度
            process_variance (float): 初始过程噪声的协方差
            measurement_variance (float): 初始测量噪声的协方差
            threshold (float): 判断变化幅度是否较大的阈值
            scale_factor (float): 自适应变化的缩放因子
        """
        self.dim = dim
        self.x = np.zeros(dim)  # 状态向量初始化
        self.P = np.ones(dim)   # 协方差矩阵初始化
        self.Q = process_variance * np.eye(dim)  # 系统过程噪声
        self.R = measurement_variance * np.eye(1)  # 测量噪声
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.prev_measurement = np.zeros(1)
        self.delta_t = 0.03333

        # 状态转移矩阵（角度和速度的预测公式）
        self.A = np.array([[1, self.delta_t],   # position = position + velocity * delta_t
                           [0, 1]])             # velocity = velocity (保持速度)

    def initialize_state_from_measurement(self, initial_measurement):
        """
        根据初始观测初始化状态。
        Args:
            initial_measurement (np.ndarray): 初始测量值
        """
        self.x[0] = initial_measurement[0]
        self.x[1] = 0.0  # 假设初始速度为零
        self.P = np.array([0.1, 0.1])  # 初始化较小协方差
        self.prev_measurement[0] = initial_measurement[0]

    def predict(self):
        """
        使用运动模型预测当前状态。
        """
        # 预测下一状态（基于状态转移矩阵）
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q  # 更新协方差矩阵

    def update(self, measurement):
        """
        更新状态估计。
        Args:
            measurement (np.ndarray): 测量值（如角度）
        Returns:
            np.ndarray: 更新后的状态 [位置, 速度]
        """
        # 动态调整测量噪声 R
        # 动态调整测量噪声 R
        delta = np.abs(measurement - self.prev_measurement)
        # delta_rate = delta / self.delta_t  # 引入速率变化判断
        if np.any(delta > self.threshold):  # 如果变化速率较大，增大测量噪声
            self.R = self.scale_factor * np.eye(1)
        else:  # 如果变化速率较小，恢复测量噪声
            self.R = 7.5 * np.eye(1)
 
        # 卡尔曼增益计算
        H = np.array([[1, 0]])  # 观测矩阵，仅观测位置
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
 
        # 更新状态
        y = measurement - H @ self.x  # 残差
        self.x += K @ y
        self.P = (np.eye(self.dim) - K @ H) @ self.P
 
        # 保存当前测量值
        self.prev_measurement = measurement
 
        return self.x

class SendRobot(Robot, Node):
    config_class = SendRobotConfig
    name = "send_robot"

    def __init__(self, config: SendRobotConfig):
        Robot.__init__(self, config)
        Node.__init__(self, "send_robot_node")
        self.config = config
        #相机初始化
        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False

        self.joint_states = None
        self.joint_actions = None

        self.motors = list(config.motor_names)
        init_grip = os.environ.get("RM65_INITIAL_GRIPPER_POS", "").strip()
        if init_grip:
            self._gripper_pos = float(init_grip)
        else:
            self._gripper_pos = config.initial_gripper_pos
        logger.info("SendRobot initial gripper pos=%.3f", self._gripper_pos)

        self.action_publisher = None
        self._ros_publish_count = 0

        # kalman滤波初始化标志
        self._kalman_initialized = False

        # 为每个关节创建自适应卡尔曼滤波器实例（夹爪不做滤波）
        self.kalman_filters = {
            motor: AdaptiveKalmanFilter(
                dim=2,
                process_variance=0.5,
                measurement_variance=5,
                threshold=0.075,
                scale_factor=15.0,
            )
            for motor in self.motors
            if motor != "left_gripper"
        }

        self._action_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            JointState, "rm/recorded_joint_states", self._joint_states_callback, self._action_qos
        )

        # teleop=True 时仅被动订阅（录制回放）；遥操必须发布 rm/sent_actions
        if not config.teleop:
            self.action_publisher = self.create_publisher(
                JointState, "rm/sent_actions", self._action_qos
            )

        # Software safety (rate limit); does not replace hardware E-Stop
        self._safety = self._make_safety_gate()
        self._safety_block_logged = False

    def _make_safety_gate(self):
        import os

        from lerobot.common.policies.smolvla_rlt.safety import SafeActionGate, SafetyConfig

        enabled = os.environ.get("RM65_SAFETY", "1").lower() not in ("0", "false", "no")
        max_delta = float(os.environ.get("RM65_MAX_JOINT_STEP_DEG", "0.4"))
        speed_scale = float(os.environ.get("RM65_SPEED_SCALE", "0.3"))
        max_grip = float(os.environ.get("RM65_MAX_DELTA_GRIPPER", "0.08"))
        # Async chunk inference often >1s; default 10s avoids false trips
        watchdog_s = float(os.environ.get("RM65_WATCHDOG_S", "10.0"))
        action_dim = len(self.motors)
        gate = SafeActionGate(
            SafetyConfig(
                enabled=enabled,
                require_deadman=False,
                max_delta_joints=max_delta,
                max_delta_gripper=max_grip,
                speed_scale=speed_scale,
                watchdog_timeout_s=watchdog_s,
                action_dim=action_dim,
            )
        )
        logger.info(
            "SendRobot SafeActionGate enabled=%s max_delta=%.3f speed_scale=%.3f "
            "watchdog=%.1fs dim=%d",
            enabled,
            max_delta,
            speed_scale,
            watchdog_s,
            action_dim,
        )
        return gate

    def _vector_from_joint_dict(self, d: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [float(d.get(f"{m}.pos", 0.0)) for m in self.motors], dtype=np.float32
        )

    def _joint_dict_from_vector(self, v: np.ndarray) -> dict[str, float]:
        return {f"{m}.pos": float(v[i]) for i, m in enumerate(self.motors)}
    
    def _joint_states_callback(self, msg: JointState):
        if not self._is_connected:
            return
        # logger.info(f"Received joint states: {msg.name} with positions {msg.position}")
        # logger.info(f"Received joint velocities: {msg.name} with velocities {msg.velocity}")
        
        self.joint_actions = {f"{name}.pos": position for name, position in zip(msg.name, msg.position)}
        self.joint_states = {f"{name}.pos": position for name, position in zip(msg.name, msg.position)}
        if "left_gripper.pos" in self.joint_states:
            self._gripper_pos = float(self.joint_states["left_gripper.pos"])

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    #这个方法返回机器人观察值的特征描述
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    #这个方法返回机器人动作的特征描述   
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected.")
        logger.info(self._is_connected)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        

        if self.joint_states is None:
            return {}

        obs_dict = self.joint_states.copy()
        if self.config.use_gripper and "left_gripper.pos" not in obs_dict:
            obs_dict["left_gripper.pos"] = self._gripper_pos

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.joint_actions is None:
            return {}

        action_dict = self.joint_actions.copy()

        return action_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        raw_grip = None
        policy_grip = None
        if self.config.use_gripper and "left_gripper.pos" in action:
            policy_grip = float(action["left_gripper.pos"])
            raw_grip = policy_grip
            # Hysteresis + debounce + min-hold: policy often sustains open/close for
            # ~0.5–1s bursts (debounce alone still chatters). Demos hold for seconds.
            if os.environ.get("RM65_GRIPPER_HYSTERESIS", "0").lower() in ("1", "true", "yes"):
                deb = max(1, int(os.environ.get("RM65_GRIPPER_DEBOUNCE", "20")))
                min_hold = float(os.environ.get("RM65_GRIPPER_MIN_HOLD_S", "2.5"))
                now_g = time.time()
                if not hasattr(self, "_grip_bin"):
                    self._grip_bin = float(self._gripper_pos)
                    self._grip_pending = None
                    self._grip_pending_n = 0
                    self._grip_bin_t = now_g
                if policy_grip >= 0.7:
                    cand = 1.0
                elif policy_grip <= 0.3:
                    cand = 0.0
                else:
                    cand = float(self._grip_bin)
                # Hard lock after a switch
                if (now_g - getattr(self, "_grip_bin_t", now_g)) < min_hold:
                    cand = float(self._grip_bin)
                    self._grip_pending = None
                    self._grip_pending_n = 0
                elif cand != float(self._grip_bin):
                    if self._grip_pending == cand:
                        self._grip_pending_n += 1
                    else:
                        self._grip_pending = cand
                        self._grip_pending_n = 1
                    if self._grip_pending_n >= deb:
                        self._grip_bin = float(cand)
                        self._grip_bin_t = now_g
                        self._grip_pending = None
                        self._grip_pending_n = 0
                else:
                    self._grip_pending = None
                    self._grip_pending_n = 0
                action = dict(action)
                action["left_gripper.pos"] = float(self._grip_bin)
                raw_grip = float(self._grip_bin)

        # Rate-limit toward current joint state before Kalman / publish
        if self._safety is not None and self._safety.config.enabled:
            desired = self._vector_from_joint_dict(action)
            if self.joint_states is not None:
                current = self._vector_from_joint_dict(self.joint_states)
            else:
                current = desired.copy()
            # record_state 通常不反馈夹爪；缺省 0 会把「策略要开爪」限速成从 0 爬升，
            # 第一步 ~0.02 会把 prep 已张开的夹爪合死。缺测时用 _gripper_pos。
            if self.config.use_gripper and "left_gripper" in self.motors:
                gi = self.motors.index("left_gripper")
                if self.joint_states is None or "left_gripper.pos" not in self.joint_states:
                    current[gi] = float(self._gripper_pos)
            gated = self._safety.filter(desired, current_state=current, deadman_held=True)
            if gated is None:
                # Async gaps can trip watchdog; clear latch and allow next tick
                reason = self._safety.last_status.blocked_reason
                if reason == "watchdog":
                    self._safety.reset()
                    gated = self._safety.filter(
                        desired, current_state=current, deadman_held=True
                    )
                if gated is None:
                    if not self._safety_block_logged:
                        logger.warning(
                            "SafeActionGate blocked action (%s) — holding",
                            reason or "unknown",
                        )
                        self._safety_block_logged = True
                    return dict(action)
            self._safety_block_logged = False
            action = self._joint_dict_from_vector(gated)
        
        # 如果还未初始化，则用第一个动作来初始化卡尔曼滤波器的状态
        use_kalman = getattr(self.config, "use_kalman_filter", True)
        if use_kalman and not self._kalman_initialized:
            for key, value in action.items():
                if key.endswith(".pos"):
                    motor = key.removesuffix(".pos")
                    if motor in self.kalman_filters:
                        self.kalman_filters[motor].initialize_state_from_measurement(np.array([value]))
            self._kalman_initialized = True

        if use_kalman:
            filtered_action = {}
            for key, value in action.items():
                if key.endswith(".pos"):
                    motor = key.removesuffix(".pos")
                    if motor in self.kalman_filters:
                        kf = self.kalman_filters[motor]
                        kf.predict()
                        filtered_value = float(kf.update(np.array([value]))[0])
                        filtered_action[key] = filtered_value
                    else:
                        filtered_action[key] = value
                else:
                    filtered_action[key] = value
        else:
            filtered_action = dict(action)

        # goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        goal_pos = {key.removesuffix(".pos"): val for key, val in filtered_action.items() if key.endswith(".pos")}

        # Executed gripper: policy raw vs after gate/kalman
        if (
            self.config.use_gripper
            and os.environ.get("RM65_LOG_GRIPPER", "1").lower() not in ("0", "false", "no")
            and "left_gripper" in goal_pos
        ):
            sent_g = float(goal_pos["left_gripper"])
            now = time.time()
            if not hasattr(self, "_grip_log_t"):
                self._grip_log_t = 0.0
                self._grip_log_last = None
                self._grip_edge_count = 0
            prev = self._grip_log_last
            if prev is not None and (prev > 0.5) != (sent_g > 0.5):
                self._grip_edge_count += 1
            if now - self._grip_log_t >= 0.5 or prev is None or abs(sent_g - (prev or sent_g)) > 0.15:
                self._grip_log_t = now
                logger.info(
                    "grip_exec policy=%s bin=%s sent=%.3f edges=%d pubs=%d",
                    f"{policy_grip:.3f}" if policy_grip is not None else "n/a",
                    f"{raw_grip:.3f}" if raw_grip is not None else "n/a",
                    sent_g,
                    self._grip_edge_count,
                    self._ros_publish_count,
                )
                log_path = os.environ.get(
                    "RM65_GRIPPER_EXEC_LOG",
                    "/home/sdznkj/zkx_ws/sendlerobot/logs/gripper_exec.log",
                )
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        pg = f"{policy_grip:.4f}" if policy_grip is not None else "nan"
                        rg = f"{raw_grip:.4f}" if raw_grip is not None else "nan"
                        f.write(
                            f"{now:.3f}\tpolicy={pg}\tbin={rg}\tsent={sent_g:.4f}\tedges={self._grip_edge_count}\n"
                        )
                except OSError:
                    pass
            self._grip_log_last = sent_g

        # 发布 JointState
        joint_state_msg = JointState()
        joint_state_msg.name = list(goal_pos.keys())
        # joint_state_msg.position = list(goal_pos.values())
        joint_state_msg.position = [float(val) for val in goal_pos.values()]
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()

        if self.action_publisher is not None:
            self.action_publisher.publish(joint_state_msg)
            self._ros_publish_count += 1

        if "left_gripper" in goal_pos:
            self._gripper_pos = float(goal_pos["left_gripper"])

        # 返回合并动作
        result = {f"{motor}.pos": val for motor, val in goal_pos.items()}

        return result
    
