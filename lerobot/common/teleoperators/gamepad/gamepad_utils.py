#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import os
import time
from pathlib import Path


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        pass

    def stop(self):
        pass

    def get_deltas(self):
        return 0.0, 0.0, 0.0

    def should_quit(self):
        return not self.running

    def update(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    def should_intervene(self):
        return self.intervention_flag

    def gripper_command(self):
        if self.open_gripper_command and not self.close_gripper_command:
            return "open"
        if self.close_gripper_command and not self.open_gripper_command:
            return "close"
        return "stay"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def stop(self):
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        delta_x = delta_y = delta_z = 0.0
        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size
        return delta_x, delta_y, delta_z


class GamepadController(InputController):
    """G7 Pro 手柄：简单稳定的按键/摇杆映射。"""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.25):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = float(os.environ.get("GAMEPAD_DEADZONE", str(deadzone)))
        self.joystick = None
        self._axis_offset: list[float] = []
        self._drift_warned = False

        self.enable_button_ids = self._parse_button_ids(
            os.environ.get("GAMEPAD_ENABLE_BUTTONS", "6")
        )
        # 仅当 TELEOP_REQUIRE_ENABLE=true 时，摇杆输入才需要按住 LB
        self.require_enable_for_sticks = os.environ.get(
            "TELEOP_REQUIRE_ENABLE", "false"
        ).lower() in ("1", "true", "yes")
        self.gripper_open_button = int(os.environ.get("GAMEPAD_GRIPPER_OPEN", "7"))
        self.gripper_close_button = int(os.environ.get("GAMEPAD_GRIPPER_CLOSE", "2"))
        self.left_x_axis = int(os.environ.get("GAMEPAD_LEFT_X_AXIS", "0"))
        self.left_y_axis = int(os.environ.get("GAMEPAD_LEFT_Y_AXIS", "1"))
        self.z_axis_index = int(os.environ.get("GAMEPAD_Z_AXIS", "3"))

    @staticmethod
    def _parse_button_ids(raw: str) -> tuple[int, ...]:
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())

    def start(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-cursor")
        Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
        # 不设置 SDL_JOYSTICK_DEVICE，避免 js0 被占用时 pygame 直接崩溃
        os.environ.pop("SDL_JOYSTICK_DEVICE", None)

        if not self._open_joystick():
            self.running = False
            return

        print("校准摇杆中位（3 秒，请勿碰摇杆）…")
        self._calibrate_offsets()
        cal_axes = {self.left_x_axis, self.left_y_axis, self.z_axis_index}
        centered = [round(self._axis_centered(i), 3) for i in sorted(cal_axes)]
        if any(abs(v) > 0.12 for v in centered):
            print(f"警告：校准后中位仍偏离 {centered}，遥操可能漂移。请松开摇杆后重启。")
        print(
            f"GameSir 轴: 左({self.left_x_axis},{self.left_y_axis}) "
            f"右Z({self.z_axis_index}) | RB开爪={self.gripper_open_button} X合爪={self.gripper_close_button}"
        )
        if self.require_enable_for_sticks:
            print(f"  使能模式: 按住 LB{self.enable_button_ids} + 摇杆")
        print("  注意: 旧关节/IK 遥操已禁用；请用 rm65_run_movep_teleop.sh（左XY 右Z）")
        print(f"校准后: {centered}")
        print("  RB→夹爪开(1.0)  X→夹爪合(0.0)")

    def _open_joystick(self) -> bool:
        import pygame

        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()

            if pygame.joystick.get_count() == 0:
                logging.error("pygame 未检测到任何手柄")
                return False

            last_error: Exception | None = None
            for index in range(pygame.joystick.get_count()):
                candidate = pygame.joystick.Joystick(index)
                try:
                    candidate.init()
                except pygame.error as exc:
                    last_error = exc
                    logging.warning("无法打开手柄 index=%s: %s", index, exc)
                    continue
                if candidate.get_numaxes() < 2:
                    candidate.quit()
                    continue
                self.joystick = candidate
                logging.info("已打开手柄 index=%s name=%s", index, candidate.get_name())
                return True

            logging.error(f"所有手柄 index 均打开失败{': ' + str(last_error) if last_error else ''}")
            return False
        except Exception as exc:
            logging.error(f"打开手柄失败: {exc}", exc_info=True)
            return False

    def stop(self):
        import pygame

        try:
            if self.joystick is not None:
                self.joystick.quit()
                self.joystick = None
            if pygame.joystick.get_init():
                pygame.joystick.quit()
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass

    def _calibrate_offsets(self) -> None:
        import pygame

        if not self.joystick:
            return
        n = self.joystick.get_numaxes()
        cal_axes = {self.left_x_axis, self.left_y_axis, self.z_axis_index}
        sums = [0.0] * n
        samples = 60
        for _ in range(samples):
            pygame.event.pump()
            for i in range(n):
                if i in cal_axes:
                    sums[i] += float(self.joystick.get_axis(i))
            time.sleep(0.05)
        self._axis_offset = [0.0] * n
        for i in cal_axes:
            if i < n:
                self._axis_offset[i] = sums[i] / samples

    def _axis_raw(self, index: int) -> float:
        if self.joystick is None or index >= self.joystick.get_numaxes():
            return 0.0
        return float(self.joystick.get_axis(index))

    def _axis_centered(self, index: int) -> float:
        raw = self._axis_raw(index)
        if index < len(self._axis_offset):
            raw -= self._axis_offset[index]
        return raw

    def _button(self, index: int) -> bool:
        if self.joystick is None or index >= self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(index))

    def get_pressed_button_ids(self) -> list[int]:
        if not self.joystick:
            return []
        return [i for i in range(self.joystick.get_numbuttons()) if self.joystick.get_button(i)]

    def update(self):
        if not self.joystick:
            return

        import pygame

        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = "success"
                elif event.button == 1:
                    self.episode_end_status = "failure"
                elif event.button == 0:
                    self.episode_end_status = "rerecord_episode"
            elif event.type == pygame.JOYBUTTONUP and event.button in (0, 2, 3):
                self.episode_end_status = None

        pressed = self.get_pressed_button_ids()
        self.intervention_flag = any(i in pressed for i in self.enable_button_ids)
        self.open_gripper_command = self._button(self.gripper_open_button)
        self.close_gripper_command = self._button(self.gripper_close_button)

    def get_deltas(self):
        if not self.joystick:
            return 0.0, 0.0, 0.0

        if self.require_enable_for_sticks and not self.intervention_flag:
            return 0.0, 0.0, 0.0

        import pygame

        try:
            pygame.event.pump()
            x = self._axis_centered(self.left_x_axis)
            y = self._axis_centered(self.left_y_axis)
            z = self._axis_centered(self.z_axis_index)

            # G7 Pro + SDL 常见误读：三轴同时饱和，并非真实摇杆输入
            if abs(x) > 0.95 and abs(y) > 0.95 and abs(z) > 0.95:
                if not self._drift_warned:
                    self._drift_warned = True
                    logging.warning("检测到摇杆三轴同时饱和，已忽略（SDL 漂移）。请重插手柄或调大 GAMEPAD_DEADZONE。")
                return 0.0, 0.0, 0.0

            if abs(x) < self.deadzone:
                x = 0.0
            if abs(y) < self.deadzone:
                y = 0.0
            if abs(z) < self.deadzone:
                z = 0.0

            return -y * self.y_step_size, -x * self.x_step_size, -z * self.z_step_size
        except pygame.error as exc:
            logging.error(f"读取手柄失败: {exc}")
            return 0.0, 0.0, 0.0


def pick_joystick_device() -> str | None:
    """保留兼容；不再设置 SDL_JOYSTICK_DEVICE。"""
    if os.path.exists("/dev/input/js0"):
        return "/dev/input/js0"
    return None


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

    def start(self):
        import hid

        devices = hid.enumerate()
        for dev in devices:
            if dev.get("usage_page") == 1 and dev.get("usage") == 5:
                self.device_info = dev
                break
        if self.device_info is None:
            logging.error("No HID gamepad found")
            self.running = False
            return
        self.device = hid.device()
        self.device.open_path(self.device_info["path"])

    def stop(self):
        if self.device:
            self.device.close()

    def update(self):
        pass

    def get_deltas(self):
        return 0.0, 0.0, 0.0
