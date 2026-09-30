#!/usr/bin/env python3
"""G7 Pro 按键/摇杆诊断：找出 LB 使能键对应的 button id。"""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-cursor")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)

import pygame  # noqa: E402

from lerobot.common.teleoperators.gamepad.gamepad_utils import pick_joystick_device  # noqa: E402


def main() -> None:
    pick_joystick_device()
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄")
        raise SystemExit(1)

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"手柄: {joy.get_name()}")
    print(f"按钮数: {joy.get_numbuttons()}, 轴数: {joy.get_numaxes()}")
    print("请依次按 LB、RB、A/B/X/Y 并推动摇杆，10 秒内显示变化…\n")

    last_bt: list[int] = []
    last_axes: list[float] = []
    end = time.time() + 10
    while time.time() < end:
        pygame.event.pump()
        bt = [i for i in range(joy.get_numbuttons()) if joy.get_button(i)]
        axes = [round(joy.get_axis(i), 3) for i in range(joy.get_numaxes())]
        if bt != last_bt or any(abs(a - b) > 0.05 for a, b in zip(axes, last_axes)):
            print(f"buttons={bt}  axes={axes}")
            last_bt, last_axes = bt, axes
        time.sleep(0.05)

    print("\nG7 Pro 实测: LB=6, RB=7, A=0, B=1（遥操: LB开 / RB合）")
    print("启动遥操:")
    print("  /home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_movep_teleop.sh")
    joy.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
