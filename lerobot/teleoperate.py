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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up

from lerobot.common.teleoperators import gamepad, koch_leader, so100_leader, so101_leader  # noqa: F401
import sendrobot.send_robot.config_send_robot  # noqa: F401  # register send_robot types
from sendrobot.ros_runtime import make_ros_robot


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    ros_ctx=None,
):
    import rclpy
    from rclpy.node import Node

    joint_keys = [k for k in robot.action_features if k.endswith(".pos")]
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        if ros_ctx is not None and isinstance(robot, Node) and rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0)
        action = teleop.get_action()
        if display_data:
            import rerun as rr

            observation = robot.get_observation()
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation_{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        sent_action = robot.send_action(action)
        # 固定显示顺序：使能/摇杆/关节
        display_items: list[tuple[str, float]] = [
            ("enable", float(action.get("enable", 0))),
        ]
        pressed = action.get("_pressed_bt_list") or []
        display_items.append(("pressed_bt", float(pressed[0]) if len(pressed) == 1 else (-1.0 if not pressed else float(len(pressed)))))
        display_items.extend([
            ("delta_x", float(action.get("delta_x", 0))),
            ("delta_y", float(action.get("delta_y", 0))),
            ("delta_z", float(action.get("delta_z", 0))),
        ])
        gripper_cmd = int(action.get("gripper", 1))
        display_items.append(
            ({0: "gripper(合)", 1: "gripper(停)", 2: "gripper(开)"}.get(gripper_cmd, "gripper"), float(gripper_cmd))
        )
        for key in joint_keys:
            if key in sent_action:
                display_items.append((key, float(sent_action[key])))
        if hasattr(robot, "_ros_publish_count"):
            display_items.append(("ros_pub", float(getattr(robot, "_ros_publish_count", 0))))
        require_enable = bool(
            getattr(getattr(robot, "config", None), "require_enable_button", False)
        )
        stick_eps = float(
            getattr(getattr(robot, "config", None), "stick_motion_threshold", 0.25)
        )
        stick_active = max(
            abs(float(action.get("delta_x", 0))),
            abs(float(action.get("delta_y", 0))),
            abs(float(action.get("delta_z", 0))),
        ) >= stick_eps
        if require_enable and float(action.get("enable", 0)) < 0.5:
            display_items.append(("arm_status", 0.0))
            if stick_active:
                display_items.append(("hint", 1.0))
        elif not stick_active:
            display_items.append(("arm_status", 1.0))
        elif not any(k in sent_action for k in joint_keys):
            display_items.append(("arm_status", 1.5))
        else:
            display_items.append(("arm_status", 2.0))

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start
        display_len = max(len(name) for name, _ in display_items)

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for name, value in display_items:
            label = name
            if name == "arm_status":
                label = {
                    0.0: "arm_status(未使能)",
                    1.0: "arm_status(待推摇杆)" if not require_enable else "arm_status(已使能-请推摇杆)",
                    1.5: "arm_status(无关节输出)",
                    2.0: "arm_status(已下发ROS)",
                }.get(value, name)
            elif name == "hint":
                label = ">>> 按住 LB(6) 才能动臂 <<<"
            elif name == "pressed_bt" and value < 0:
                label = "pressed_bt(无)"
            print(f"{label:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(display_items) + 5)


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        from lerobot.common.utils.visualization_utils import _init_rerun

        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot, ros_ctx = make_ros_robot(cfg.robot)

    teleop.connect()
    if not teleop.is_connected():
        raise RuntimeError(
            "手柄未连接。请确认 G7 Pro USB 已插入 Jetson，并在容器内执行: ls -la /dev/input/js*"
        )
    robot.connect()

    require_enable = bool(getattr(getattr(robot, "config", None), "require_enable_button", False))
    joint_space = bool(getattr(getattr(robot, "config", None), "joint_space_teleop", False))
    if joint_space:
        print("\n>>> 关节模式：左摇杆→J1/J2  右摇杆→J3  |  RB(7)开爪 / X(2)合爪\n")
    else:
        print("\n>>> 笛卡尔模式：左摇杆→水平(XY)  右摇杆上下→竖直(Z)  |  RB(7)开爪 / X(2)合爪\n")
    if require_enable:
        print(">>> 需按住 LB(6) 才能动臂\n")

    try:
        teleop_loop(
            teleop,
            robot,
            cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            ros_ctx=ros_ctx,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            import rerun as rr

            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
        if ros_ctx is not None:
            ros_ctx.stop()


if __name__ == "__main__":
    teleoperate()
