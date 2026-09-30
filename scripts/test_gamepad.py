#!/usr/bin/env python3
"""G7 Pro / Xbox 手柄检测脚本（不依赖 LeRobot/draccus）。"""

import glob
import os
import sys
import time

import pygame


def _pick_joystick_device() -> None:
    """Jetson 上 SDL 有时无法自动发现 js0，显式指定设备路径。"""
    if os.environ.get("SDL_JOYSTICK_DEVICE"):
        return
    for pattern in ("/dev/input/js*", "/dev/input/by-id/*-joystick"):
        for path in sorted(glob.glob(pattern)):
            os.environ["SDL_JOYSTICK_DEVICE"] = path
            return


def main() -> int:
    _pick_joystick_device()
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    print(f"pygame 识别到手柄数量: {count}")

    if count == 0:
        print("\n未检测到手柄。请先确认：")
        print("  1. lsusb 中是否出现 3537:xxxx (GameSir) 或 045e:xxxx (Xbox)")
        print("  2. 2.4G 模式：接收器已插在本机 USB，手柄已开机并完成配对")
        print("  3. 有线模式：使用数据线（非仅充电线）直连本机")
        print("  4. 蓝牙模式：需先 bluetoothctl pair/connect")
        print("\n本机内核 CONFIG_INPUT_JOYSTICK 未启用，/dev/input/js* 可能不会出现，")
        print("但 USB 枚举成功后 pygame 仍可能通过 evdev 识别。")
        return 1

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"手柄名称: {joy.get_name()}")
    print(f"轴数量: {joy.get_numaxes()}, 按钮数量: {joy.get_numbuttons()}, 帽键: {joy.get_numhats()}")
    print("\n操作摇杆/按键 10 秒，观察数值变化 (Ctrl+C 退出):\n")

    deadline = time.time() + 10
    while time.time() < deadline:
        pygame.event.pump()
        axes = [round(joy.get_axis(i), 3) for i in range(joy.get_numaxes())]
        buttons = [joy.get_button(i) for i in range(joy.get_numbuttons())]
        active_axes = [i for i, v in enumerate(axes) if abs(v) > 0.05]
        active_buttons = [i for i, v in enumerate(buttons) if v]
        if active_axes or active_buttons:
            print(f"axes={axes}  active_axes={active_axes}  active_buttons={active_buttons}")
        time.sleep(0.05)

    joy.quit()
    pygame.joystick.quit()
    pygame.quit()
    print("\n检测完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
