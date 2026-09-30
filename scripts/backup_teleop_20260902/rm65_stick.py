#!/usr/bin/env python3
"""手柄积分：摇杆/十字键 → 绝对 xyz + rpy。遥操与录制共用。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _axis_ids() -> tuple[int, int, int, int]:
    # 默认：左上下(a1)→Z，左左右(a0)→Y
    y = int(os.environ.get("GAMEPAD_LEFT_Y_AXIS", "0"))
    z = int(os.environ.get("GAMEPAD_LEFT_Z_AXIS", "1"))
    x = int(os.environ.get("GAMEPAD_X_AXIS", os.environ.get("GAMEPAD_Z_AXIS", "3")))
    pitch = int(os.environ.get("GAMEPAD_PITCH_AXIS", "2"))
    return y, z, x, pitch


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
        axis_y: int = 0,  # 左左右 → Y
        axis_z: int = 1,  # 左上下 → Z
        axis_x: int = 3,
        axis_pitch: int = 2,
        invert_x: bool = False,  # 右上下 X 相对原版反向
        invert_y: bool = True,
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
        # G7 Pro (pygame): LB=6 开, RB=7 合；A=0 开始录 / B=1 结束 / X=2 重录
        self.btn_gripper_open = int(os.environ.get("GAMEPAD_GRIPPER_OPEN", "6"))
        self.btn_gripper_close = int(os.environ.get("GAMEPAD_GRIPPER_CLOSE", "7"))
        self.btn_rec_start = int(os.environ.get("GAMEPAD_REC_START", "0"))
        self.btn_rec_stop = int(os.environ.get("GAMEPAD_REC_STOP", "1"))
        self.btn_rec_rerecord = int(os.environ.get("GAMEPAD_REC_RERECORD", "2"))
        self._prev_open = False
        self._prev_close = False
        self._prev_rec_start = False
        self._prev_rec_stop = False
        self._prev_rec_rerecord = False
        self.gripper_event: str | None = None  # "open" | "close"
        self.record_event: str | None = None  # "start" | "stop" | "rerecord"

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
            f"映射: 左上下→Z(a{self.axis_z}) 左左右→Y(a{self.axis_y})  "
            f"右上下→X(a{self.axis_x})  右左右→ry(a{self.axis_pitch})  "
            f"十字上下→rx  十字左右→rz  "
            f"LB(b{self.btn_gripper_open})→夹爪开  RB(b{self.btn_gripper_close})→夹爪合  "
            f"A(b{self.btn_rec_start})→开录  B(b{self.btn_rec_stop})→结束  "
            f"X(b{self.btn_rec_rerecord})→重录  "
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

        nbtn = self.joystick.get_numbuttons()
        open_now = bool(
            self.btn_gripper_open < nbtn
            and self.joystick.get_button(self.btn_gripper_open)
        )
        close_now = bool(
            self.btn_gripper_close < nbtn
            and self.joystick.get_button(self.btn_gripper_close)
        )
        self.gripper_event = None
        if open_now and not self._prev_open:
            self.gripper_event = "open"
        elif close_now and not self._prev_close:
            self.gripper_event = "close"
        self._prev_open = open_now
        self._prev_close = close_now

        start_now = bool(
            self.btn_rec_start < nbtn and self.joystick.get_button(self.btn_rec_start)
        )
        stop_now = bool(
            self.btn_rec_stop < nbtn and self.joystick.get_button(self.btn_rec_stop)
        )
        rerecord_now = bool(
            self.btn_rec_rerecord < nbtn and self.joystick.get_button(self.btn_rec_rerecord)
        )
        self.record_event = None
        if start_now and not self._prev_rec_start:
            self.record_event = "start"
        elif stop_now and not self._prev_rec_stop:
            self.record_event = "stop"
        elif rerecord_now and not self._prev_rec_rerecord:
            self.record_event = "rerecord"
        self._prev_rec_start = start_now
        self._prev_rec_stop = stop_now
        self._prev_rec_rerecord = rerecord_now

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
