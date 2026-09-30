#!/usr/bin/env python3
"""
RM65 笛卡尔遥操（侧装）——可用基线备份（与当前手感一致；改动请改 rm65_movep_teleop.py）。

逻辑（写死）：
  - 左摇杆左右 → 位置 Z；左摇杆上下 → 位置 Y
  - 右摇杆上下 → 位置 X（方向已按手感反向）；右摇杆左右 → 姿态 ry
  - 十字键上下 → 姿态 rx；十字键左右 → 姿态 rz
  - 启动读当前位姿作种子；无使能键、无松杆停发、无取种

用法：
  /home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_movep_teleop.sh --print-only
  /home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_movep_teleop.sh
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

READY_JOINTS_DEG = [0.0, 20.0, 70.0, 10.0, -20.0, 0.0]
READY_POSE_PATH = Path(__file__).resolve().parent / "rm65_ready_pose.json"
PRINT_ONLY_SEED = [-0.3, 0.0, 0.5, 0.0, 0.0, 0.0]
RM_IK_PATH = Path(__file__).resolve().parents[1] / "sendrobot" / "send_robot" / "rm_ik.py"


def load_ready_joints_deg() -> list[float]:
    """从 rm65_ready_pose.json 读就绪关节；失败则用内置默认。"""
    try:
        from rm65_save_ready_pose import load_ready_joints_deg as _load

        return _load(READY_POSE_PATH)
    except Exception:
        if not READY_POSE_PATH.is_file():
            return list(READY_JOINTS_DEG)
        try:
            import json

            data = json.loads(READY_POSE_PATH.read_text(encoding="utf-8"))
            joints = [float(v) for v in data["joints_deg"][:6]]
            if len(joints) == 6:
                return joints
        except Exception as exc:
            print(f"警告: 读取就绪位失败，用默认: {exc}", file=sys.stderr)
        return list(READY_JOINTS_DEG)


def _load_kinematics():
    spec = importlib.util.spec_from_file_location("rm_ik", RM_IK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.RM65Kinematics


class StickIntegrator:
    """摇杆/十字键 → 绝对 xyz + rpy 积分。"""

    def __init__(
        self,
        x0: float,
        y0: float,
        z0: float,
        rx0: float,
        ry0: float,
        rz0: float,
        *,
        sensitivity: float,
        orient_sensitivity: float,
        deadzone: float,
        bounds: dict[str, tuple[float, float]],
        axis_y: int = 1,  # 左上下 → Y
        axis_z: int = 0,  # 左左右 → Z
        axis_x: int = 3,
        axis_pitch: int = 2,
        invert_x: bool = False,  # 右上下 X 相对原版反向
        invert_y: bool = False,
        invert_z: bool = True,
        invert_pitch: bool = False,
        invert_roll: bool = False,
        invert_yaw: bool = False,
    ):
        self.x, self.y, self.z = float(x0), float(y0), float(z0)
        self.rx, self.ry, self.rz = float(rx0), float(ry0), float(rz0)
        self.sensitivity = float(sensitivity)
        self.orient_sensitivity = float(orient_sensitivity)
        self.deadzone = float(deadzone)
        self.bounds = bounds
        self.axis_y = axis_y
        self.axis_z = axis_z
        self.axis_x = axis_x
        self.axis_pitch = axis_pitch
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        self.invert_pitch = invert_pitch
        self.invert_roll = invert_roll
        self.invert_yaw = invert_yaw
        self.joystick = None
        self._offset: list[float] = []
        self.last_rx_stick = 0.0
        self.last_pitch = 0.0
        self.last_hat = (0, 0)

    def start(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-cursor")
        Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
        js_dev = os.environ.get("SDL_JOYSTICK_DEVICE", "/dev/input/js0")
        if Path(js_dev).exists():
            os.environ["SDL_JOYSTICK_DEVICE"] = js_dev
        else:
            os.environ.pop("SDL_JOYSTICK_DEVICE", None)

        import pygame

        js = None
        last_err: Exception | None = None
        for attempt in range(1, 6):
            if pygame.get_init():
                pygame.quit()
            pygame.init()
            pygame.joystick.init()
            n = pygame.joystick.get_count()
            if n == 0:
                last_err = RuntimeError("no joystick")
                time.sleep(0.4)
                continue
            for i in range(n):
                cand = pygame.joystick.Joystick(i)
                try:
                    cand.init()
                except pygame.error as exc:
                    last_err = exc
                    continue
                if cand.get_numaxes() >= 2:
                    js = cand
                    break
                cand.quit()
            if js is not None:
                break
            print(f"手柄打开失败 ({attempt}/5): {last_err}")
            time.sleep(0.5)
        if js is None:
            raise RuntimeError(f"无法打开手柄: {last_err}")
        self.joystick = js
        print(
            f"手柄: {js.get_name()} axes={js.get_numaxes()} "
            f"hats={js.get_numhats()} buttons={js.get_numbuttons()}"
        )
        if js.get_numhats() < 1:
            print("警告: 未检测到十字键(hat)，rx/rz 将无法用十字键控制", file=sys.stderr)
        self._calibrate()
        print(
            f"映射: 左上下→Y(a{self.axis_y}) 左左右→Z(a{self.axis_z})  "
            f"右上下→X(a{self.axis_x})  右左右→ry(a{self.axis_pitch})  "
            f"十字上下→rx  十字左右→rz  "
            f"deadzone={self.deadzone} sens={self.sensitivity} m/帧  "
            f"orient={self.orient_sensitivity} rad/帧"
        )
        print(
            f"种子: xyz=({self.x:.4f},{self.y:.4f},{self.z:.4f}) "
            f"rpy=({self.rx:.3f},{self.ry:.3f},{self.rz:.3f})"
        )

    def _calibrate(self) -> None:
        import pygame

        assert self.joystick is not None
        n = self.joystick.get_numaxes()
        use = {self.axis_y, self.axis_z, self.axis_x, self.axis_pitch}
        sums = [0.0] * n
        for _ in range(40):
            pygame.event.pump()
            for i in use:
                if i < n:
                    sums[i] += float(self.joystick.get_axis(i))
            time.sleep(0.05)
        self._offset = [0.0] * n
        for i in use:
            if i < n:
                self._offset[i] = sums[i] / 40.0
        print(f"中位校准: {[round(self._offset[i], 3) for i in sorted(use)]}")

    def _read(self, index: int) -> float:
        assert self.joystick is not None
        import pygame

        pygame.event.pump()
        if index >= self.joystick.get_numaxes():
            return 0.0
        v = float(self.joystick.get_axis(index))
        if index < len(self._offset):
            v -= self._offset[index]
        if abs(v) < self.deadzone:
            return 0.0
        return v

    def _hat(self) -> tuple[int, int]:
        assert self.joystick is not None
        import pygame

        pygame.event.pump()
        if self.joystick.get_numhats() < 1:
            return (0, 0)
        hx, hy = self.joystick.get_hat(0)
        return int(hx), int(hy)

    def update(self) -> tuple[float, float, float, float, float, float]:
        assert self.joystick is not None
        import pygame

        pygame.event.pump()
        stick_y = self._read(self.axis_y)
        stick_z = self._read(self.axis_z)
        stick_x = self._read(self.axis_x)
        stick_pitch = self._read(self.axis_pitch)
        hat_x, hat_y = self._hat()
        self.last_rx_stick = stick_x
        self.last_pitch = stick_pitch
        self.last_hat = (hat_x, hat_y)

        # 位置
        dx = (-stick_x if self.invert_x else stick_x) * self.sensitivity
        dy = (-stick_y if self.invert_y else stick_y) * self.sensitivity
        dz = (-stick_z if self.invert_z else stick_z) * self.sensitivity

        # 姿态：右左右 → ry；十字上下 → rx；十字左右 → rz
        # pygame hat: x=-1左/1右，y=1上/-1下
        pitch = (-stick_pitch if self.invert_pitch else stick_pitch) * self.orient_sensitivity
        roll = float(hat_y) * self.orient_sensitivity  # 上=+hy → +rx（可用 invert_roll 翻转）
        if self.invert_roll:
            roll = -roll
        yaw = float(hat_x) * self.orient_sensitivity
        if self.invert_yaw:
            yaw = -yaw

        xb, yb, zb = self.bounds["x"], self.bounds["y"], self.bounds["z"]
        self.x = float(np.clip(self.x + dx, xb[0], xb[1]))
        self.y = float(np.clip(self.y + dy, yb[0], yb[1]))
        self.z = float(np.clip(self.z + dz, zb[0], zb[1]))
        self.rx = float(self.rx + roll)
        self.ry = float(self.ry + pitch)
        self.rz = float(self.rz + yaw)
        return self.x, self.y, self.z, self.rx, self.ry, self.rz

    def cleanup(self) -> None:
        import pygame

        try:
            if self.joystick is not None:
                self.joystick.quit()
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass


def read_pose(ip: str, port: int) -> list[float]:
    kin = _load_kinematics()()
    kin.ip, kin.port = ip, port
    kin.connect()
    try:
        pose = kin.get_current_pose()
        print(
            f"初始位姿(SDK): xyz=({pose[0]:.4f},{pose[1]:.4f},{pose[2]:.4f}) "
            f"rpy=({pose[3]:.4f},{pose[4]:.4f},{pose[5]:.4f})"
        )
        return [float(v) for v in pose[:6]]
    finally:
        kin.disconnect()
        time.sleep(0.3)


def movej_ready(speed: int = 5, timeout_s: float = 120.0) -> None:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rm_ros_interfaces.msg import Movej
    from std_msgs.msg import Bool

    node = rclpy.create_node("rm65_movep_teleop_ready")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(Movej, "/rm_driver/movej_cmd", qos)
    done = {"ok": None}

    def _cb(msg: Bool) -> None:
        done["ok"] = bool(msg.data)

    node.create_subscription(Bool, "/rm_driver/movej_result", _cb, qos)
    for _ in range(50):
        if pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        node.destroy_node()
        raise RuntimeError("rm_driver 未订阅 movej_cmd")

    joints_deg = load_ready_joints_deg()
    msg = Movej()
    msg.joint = [math.radians(q) for q in joints_deg]
    msg.speed = speed
    msg.block = True
    msg.trajectory_connect = 0
    msg.dof = 6
    print(
        f"movej → 就绪位 {[round(q, 2) for q in joints_deg]}  "
        f"({READY_POSE_PATH.name})"
    )
    pub.publish(msg)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if done["ok"] is not None:
            break
    node.destroy_node()
    if done["ok"] is not True:
        raise RuntimeError(f"就绪位失败: {done['ok']}")
    print("就绪位到位")


def _axis_ids() -> tuple[int, int, int, int]:
    # 默认：左上下(a1)→Y，左左右(a0)→Z
    y = int(os.environ.get("GAMEPAD_LEFT_Y_AXIS", "1"))
    z = int(os.environ.get("GAMEPAD_LEFT_Z_AXIS", "0"))
    x = int(os.environ.get("GAMEPAD_X_AXIS", os.environ.get("GAMEPAD_Z_AXIS", "3")))
    pitch = int(os.environ.get("GAMEPAD_PITCH_AXIS", "2"))
    return y, z, x, pitch


def _make_stick(seed: list[float], args: argparse.Namespace) -> StickIntegrator:
    ay, az, ax, apitch = _axis_ids()
    return StickIntegrator(
        seed[0],
        seed[1],
        seed[2],
        seed[3],
        seed[4],
        seed[5],
        sensitivity=args.sensitivity,
        orient_sensitivity=args.orient_sensitivity,
        deadzone=args.deadzone,
        bounds={
            "x": (args.x_min, args.x_max),
            "y": (args.y_min, args.y_max),
            "z": (args.z_min, args.z_max),
        },
        axis_y=ay,
        axis_z=az,
        axis_x=ax,
        axis_pitch=apitch,
    )


def run_print_only(args: argparse.Namespace) -> int:
    stick = _make_stick(PRINT_ONLY_SEED, args)
    stick.start()
    dt = 1.0 / args.rate
    print("print-only：不发指令。Ctrl+C 退出。")
    last = 0.0
    r0 = (stick.rx, stick.ry, stick.rz)
    try:
        while True:
            t0 = time.time()
            x, y, z, rx, ry, rz = stick.update()
            now = time.time()
            if now - last >= 0.25:
                last = now
                print(
                    f"xyz=({x:.4f},{y:.4f},{z:.4f})  "
                    f"rpy=({rx:.3f},{ry:.3f},{rz:.3f})  "
                    f"Δrpy=("
                    f"{(rx-r0[0])*180/math.pi:+.1f},"
                    f"{(ry-r0[1])*180/math.pi:+.1f},"
                    f"{(rz-r0[2])*180/math.pi:+.1f})deg  "
                    f"hat={stick.last_hat} pitch={stick.last_pitch:+.2f}"
                )
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stick.cleanup()
    return 0


def run_teleop(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rm_ros_interfaces.msg import Cartepos
    from scipy.spatial.transform import Rotation

    if args.go_ready:
        rclpy.init()
        try:
            movej_ready(speed=args.ready_speed)
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        time.sleep(0.5)

    if args.pose:
        parts = [float(v) for v in args.pose.split(",")]
        if len(parts) != 6:
            raise ValueError("--pose 需要 x,y,z,rx,ry,rz")
        seed = parts
        print(f"初始位姿(手动): {[round(v, 4) for v in seed]}")
    else:
        seed = read_pose(args.ip, args.port)

    stick = _make_stick(seed, args)
    stick.start()

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("rm65_movep_teleop")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
    )
    pub = node.create_publisher(Cartepos, "/rm_driver/movep_canfd_cmd", qos)
    for _ in range(50):
        if pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        stick.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("rm_driver 未订阅 movep_canfd_cmd")

    dt = 1.0 / args.rate
    print(
        f"控制环 {args.rate:g}Hz  "
        f"平移约 {args.sensitivity * args.rate * 1000:.0f} mm/s  "
        f"姿态约 {args.orient_sensitivity * args.rate * 180 / math.pi:.1f} deg/s"
    )
    print("Ctrl+C 退出。")
    last_log = 0.0
    x0, y0, z0 = stick.x, stick.y, stick.z
    r0 = (stick.rx, stick.ry, stick.rz)
    try:
        while True:
            t0 = time.time()
            x, y, z, rx, ry, rz = stick.update()

            quat = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])
            msg = Cartepos()
            msg.pose = pose
            msg.follow = args.follow
            pub.publish(msg)

            now = time.time()
            if now - last_log >= 0.5:
                last_log = now
                print(
                    f"xyz=({x:.4f},{y:.4f},{z:.4f})  "
                    f"rpy=({rx:.3f},{ry:.3f},{rz:.3f})  "
                    f"Δxyz=("
                    f"{(x-x0)*1000:.0f},"
                    f"{(y-y0)*1000:.0f},"
                    f"{(z-z0)*1000:.0f})mm  "
                    f"Δrpy=("
                    f"{(rx-r0[0])*180/math.pi:+.1f},"
                    f"{(ry-r0[1])*180/math.pi:+.1f},"
                    f"{(rz-r0[2])*180/math.pi:+.1f})deg  "
                    f"hat={stick.last_hat}"
                )
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        stick.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RM65 movep 侧装遥操")
    p.add_argument("--ip", default=os.environ.get("RM_IP", "192.168.73.18"))
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument(
        "--sensitivity",
        type=float,
        default=float(os.environ.get("TELEOP_SENSITIVITY", "0.001")),
    )
    p.add_argument(
        "--orient-sensitivity",
        type=float,
        default=float(os.environ.get("TELEOP_ORIENT_SENSITIVITY", "0.003")),
        help="姿态每帧 rad，默认 0.003 ≈ 8.6deg/s @50Hz（偏慢更稳）",
    )
    p.add_argument(
        "--deadzone",
        type=float,
        default=float(os.environ.get("GAMEPAD_DEADZONE", "0.15")),
    )
    p.add_argument("--follow", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--go-ready", action="store_true")
    p.add_argument("--ready-speed", type=int, default=5)
    p.add_argument("--pose", default="")
    p.add_argument("--x-min", type=float, default=-0.8)
    p.add_argument("--x-max", type=float, default=0.0)
    p.add_argument("--y-min", type=float, default=-0.5)
    p.add_argument("--y-max", type=float, default=0.5)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.85)
    args = p.parse_args()
    try:
        if args.print_only:
            return run_print_only(args)
        return run_teleop(args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
