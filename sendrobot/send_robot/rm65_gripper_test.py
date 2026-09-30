#!/usr/bin/env python3
"""大寰夹爪 Modbus 开合测试（不依赖遥操）。"""

from __future__ import annotations

import argparse
import time

REG_INIT = 0x0100
REG_FORCE = 0x0101
REG_SPEED = 0x0102
REG_POS = 0x0103
REG_INIT_STATE = 0x0200
MODBUS_PORT = 1
MODBUS_DEVICE = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="大寰夹爪 Modbus 开合测试")
    parser.add_argument("--ip", default="192.168.73.18")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.ip, args.port)
    if handle.id < 0:
        raise SystemExit(f"连接失败 {args.ip}:{args.port}")

    def wr(addr: int, val: int) -> int:
        p = rm_peripheral_read_write_params_t(port=MODBUS_PORT, address=addr, device=MODBUS_DEVICE, num=1)
        return arm.rm_write_single_register(p, int(val))

    def rd(addr: int) -> int | None:
        p = rm_peripheral_read_write_params_t(port=MODBUS_PORT, address=addr, device=MODBUS_DEVICE, num=1)
        code, value = arm.rm_read_holding_registers(p)
        return int(value) if code == 0 else None

    print(">>> 24V + Modbus RTU")
    print("  tool_voltage", arm.rm_set_tool_voltage(3))
    time.sleep(0.3)
    print("  modbus", arm.rm_set_modbus_mode(MODBUS_PORT, 115200, 2))
    time.sleep(0.3)

    print(f">>> init_state={rd(REG_INIT_STATE)}")
    if rd(REG_INIT_STATE) != 1:
        print(">>> 初始化...")
        print("  code", wr(REG_INIT, 1))
        for _ in range(20):
            st = rd(REG_INIT_STATE)
            print(f"  init_state={st}")
            if st == 1:
                break
            time.sleep(0.4)

    wr(REG_FORCE, 50)
    wr(REG_SPEED, 50)

    for label, pos in [("全开", 1000), ("全合", 0), ("再开", 1000)]:
        print(f"\n>>> {label} position={pos}")
        print("  write code", wr(REG_POS, pos))
        time.sleep(3)
        print(f"  读回 position={rd(REG_POS)}")

    arm.rm_delete_robot_arm()
    print("\n完成。若夹爪未动，请检查末端 RS485 接线与夹爪 ID/波特率(115200)。")


if __name__ == "__main__":
    main()
