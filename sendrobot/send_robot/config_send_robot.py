from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig

DEFAULT_ARM_MOTORS = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
]

DEFAULT_ARM_MOTORS_WITH_GRIPPER = [*DEFAULT_ARM_MOTORS, "left_gripper"]


@RobotConfig.register_subclass("send_robot")
@dataclass
class SendRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    motor_names: list[str] = field(default_factory=lambda: list(DEFAULT_ARM_MOTORS))
    use_gripper: bool = False
    initial_gripper_pos: float = 1.0
    # teleop=True 时不发布 rm/sent_actions（仅被动读状态）；遥操录制请保持 False
    teleop: bool = False
    use_kalman_filter: bool = True


@RobotConfig.register_subclass("send_robot_end_effector")
@dataclass
class SendRobotEndEffectorConfig(SendRobotConfig):
    rm_ip: str = "192.168.73.18"
    rm_port: int = 8080
    use_gripper: bool = True
    teleop: bool = False
    use_kalman_filter: bool = False
    motor_names: list[str] = field(default_factory=lambda: list(DEFAULT_ARM_MOTORS_WITH_GRIPPER))

    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-0.8, -0.8, 0.05],
            "max": [0.8, 0.8, 0.85],
        }
    )
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.005,
            "y": 0.005,
            "z": 0.005,
        }
    )
    # work=基坐标系（左摇杆 XY / 右摇杆 Z 更符合直觉）；tool=工具坐标系
    cartesian_frame: str = "work"
    # 遥操速度倍率（0~1），与 step_sizes 相乘；可在命令行 --robot.teleop_speed_scale=0.5 调整
    teleop_speed_scale: float = 0.08
    # 每帧相对当前关节角的最大变化（度）；20Hz 下 0.4° ≈ 8°/s
    max_joint_step_deg: float = 0.4
    # 默认关闭：G7 Pro 在 SDL 下 LB 使能 + 摇杆漂移易误触发；改回 true 需设 TELEOP_REQUIRE_ENABLE=true
    require_enable_button: bool = False
# 旧关节空间遥操已禁用；保留字段仅兼容配置解析，不得用于控制臂
    joint_space_teleop: bool = False
    # 摇杆归一化读数超过此阈值才认为有运动（与 GAMEPAD_DEADZONE 对齐）
    stick_motion_threshold: float = 0.25
    gripper_step: float = 0.1
    # 夹爪初始开口度 [0=闭合, 1=张开]；设中间值便于 RB/X 双向测试
    initial_gripper_pos: float = 0.3
