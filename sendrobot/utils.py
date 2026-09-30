from .config import RobotConfig
from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "send_robot":
        from .send_robot import SendRobot

        return SendRobot(config)
    if config.type == "send_robot_end_effector":
        from .send_robot.send_robot_end_effector import SendRobotEndEffector

        return SendRobotEndEffector(config)
    raise ValueError(config.type)
