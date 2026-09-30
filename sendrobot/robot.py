import abc
from pathlib import Path
from typing import Any, Type

import draccus

from .config import RobotConfig


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """
    The base abstract class for all LeRobot-compatible robots.

    This class provides a standardized interface for interacting with physical robots.
    Subclasses must implement all abstract methods and properties to be usable.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this robot.
        name (str): The unique robot name used to identify this robot type.
    """

    # Set these in ALL subclasses
    config_class: Type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state. Its structure
                should match :pymeth:`observation_features`.
        """

        pass

    @abc.abstractmethod
    def get_action(self) -> dict[str, Any]:
        """
        Retrieve the last action sent to the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the last action sent to the robot. Its structure
                should match :pymeth:`action_features`.
        """
        pass

    @abc.abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        pass
