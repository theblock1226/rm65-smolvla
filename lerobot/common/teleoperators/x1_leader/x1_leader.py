#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_x1_leader import X1LeaderConfig

logger = logging.getLogger(__name__)


class X1Leader(Teleoperator):

    config_class = X1LeaderConfig
    name = "x1_leader"

    def __init__(self, config: X1LeaderConfig, node: rclpy.node.Node):
        super().__init__(config)
        self.config = config
        self.node = node
        self.joint_commands = {}
        self.left_gripper_command = 0.0
        self.right_gripper_command = 0.0
        self.motors = {
            "la1", "la2", "la3", "la4", "la5", "la6", "left_gripper", "ra1", "ra2", "ra3", "ra4", "ra5", "ra6", "right_gripper" 
        }
        self.is_connected = True
        self.is_calibrated = True

        self.node.create_subscription(
            JointState,
            config.joint_commands_topic,
            self._joint_commands_callback,
            10
        )

        self.node.create_subscription(
            Float32,
            config.left_gripper_command_topic,
            self._left_gripper_command_callback,
            10
        )

        self.node.create_subscription(
            Float32,
            config.right_gripper_command_topic,
            self._right_gripper_command_callback,
            10
        )

    @property
    def _joint_commands_callback(self, msg: JointState):
        self.joint_commands = dict(zip(msg.name, msg.position))
        self.node.get_logger().debug(f"Received joint_commands: {self.joint_commands}")

    @property
    def _left_gripper_command_callback(self, msg: Float32):
        self.left_gripper_command = msg.data
        self.node.get_logger().debug(f"Received left gripper position: {self.left_gripper_command}")

    @property
    def _right_gripper_command_callback(self, msg: Float32):
        self.right_gripper_command = msg.data
        self.node.get_logger().debug(f"Received right gripper position: {self.right_gripper_command}")

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # self.bus.connect()
        # if not self.is_calibrated and calibrate:
        #     self.calibrate()

        # self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.is_calibrated

    # def calibrate(self) -> None:
    #     logger.info(f"\nRunning calibration of {self}")
    #     self.bus.disable_torque()
    #     for motor in self.bus.motors:
    #         self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

    #     self.bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
    #     drive_modes = {motor: 1 if motor == "elbow_flex" else 0 for motor in self.bus.motors}

    #     input(f"Move {self} to the middle of its range of motion and press ENTER....")
    #     homing_offsets = self.bus.set_half_turn_homings()

    #     full_turn_motors = ["shoulder_pan", "wrist_roll"]
    #     unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
    #     print(
    #         f"Move all joints except {full_turn_motors} sequentially through their "
    #         "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
    #     )
    #     range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
    #     for motor in full_turn_motors:
    #         range_mins[motor] = 0
    #         range_maxes[motor] = 4095

    #     self.calibration = {}
    #     for motor, m in self.bus.motors.items():
    #         self.calibration[motor] = MotorCalibration(
    #             id=m.id,
    #             drive_mode=drive_modes[motor],
    #             homing_offset=homing_offsets[motor],
    #             range_min=range_mins[motor],
    #             range_max=range_maxes[motor],
    #         )

    #     self.bus.write_calibration(self.calibration)
    #     self._save_calibration()
    #     logger.info(f"Calibration saved to {self.calibration_fpath}")

    # def configure(self) -> None:
    #     self.bus.disable_torque()
    #     self.bus.configure_motors()
    #     for motor in self.bus.motors:
    #         if motor != "gripper":
    #             # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
    #             # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
    #             # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
    #             # point
    #             self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

    #     # Use 'position control current based' for gripper to be limited by the limit of the current.
    #     # For the follower gripper, it means it can grasp an object without forcing too much even tho,
    #     # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
    #     # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
    #     # to make it move, and it will move back to its original target position when we release the force.
    #     self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
    #     # Set gripper's goal pos in current position mode so that we can use it as a trigger.
    #     self.bus.enable_torque("gripper")
    #     if self.is_calibrated:
    #         self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    # def setup_motors(self) -> None:
    #     for motor in reversed(self.bus.motors):
    #         input(f"Connect the controller board to the '{motor}' motor only and press enter.")
    #         self.bus.setup_motor(motor)
    #         print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.joint_commands.copy()
        action = {f"{motor}.pos": val for motor, val in action.items()}
        action["left_gripper.pos"] = self.left_gripper_command
        action["right_gripper.pos"] = self.right_gripper_command
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.bus.disconnect()
        logger.info(f"{self} disconnected.")
