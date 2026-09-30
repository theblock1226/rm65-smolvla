"""ROS2 运行时辅助：SendRobot 节点需要先 rclpy.init() 并后台 spin。"""

from __future__ import annotations

import threading
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .config import RobotConfig
from .robot import Robot


class RosSpinContext:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.executor: MultiThreadedExecutor | None = None
        self.spin_thread: threading.Thread | None = None

    def start(self) -> None:
        if not isinstance(self.robot, Node):
            return
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.robot)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def stop(self) -> None:
        if isinstance(self.robot, Node):
            self.robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def is_ros_robot_config(config: RobotConfig) -> bool:
    return config.type in ("send_robot", "send_robot_end_effector")


def make_ros_robot(config: RobotConfig) -> tuple[Robot, RosSpinContext | None]:
    from .utils import make_robot_from_config

    if not is_ros_robot_config(config):
        return make_robot_from_config(config), None

    if not rclpy.ok():
        rclpy.init()

    robot = make_robot_from_config(config)
    ctx = RosSpinContext(robot)
    ctx.start()
    return robot, ctx
