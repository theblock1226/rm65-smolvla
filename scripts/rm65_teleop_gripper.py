#!/usr/bin/env python3
"""遥操夹爪：短时 SDK TCP 写大寰 Modbus；位姿仍走 ROS movep_canfd。"""

from __future__ import annotations

import os
import sys
import time

GRIPPER_OPEN_POS = 1000
GRIPPER_CLOSE_POS = 0


class _PrintLogger:
    def info(self, msg: str) -> None:
        print(msg)

    def warning(self, msg: str) -> None:
        print(f"警告: {msg}", file=sys.stderr)


class TeleopGripper:
    """短时 SDK TCP 写大寰 Modbus；遥操位姿仍走 ROS movep_canfd。"""

    def __init__(self, ip: str, port: int, *, enabled: bool = True):
        self.ip = ip
        self.port = int(port)
        self.enabled = bool(enabled)
        self.ready = False
        self._force = int(os.environ.get("DAHUAN_FORCE", "50"))
        self._speed = int(os.environ.get("DAHUAN_SPEED", "50"))

    def _with_arm(self, fn):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(self.ip, self.port)
        if handle.id < 0:
            raise RuntimeError(f"SDK 连接失败 {self.ip}:{self.port}")
        try:
            return fn(arm)
        finally:
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                pass
            time.sleep(0.2)

    def setup(self) -> bool:
        if not self.enabled:
            print("夹爪: 已禁用 (--no-gripper)")
            return False
        try:
            from sendrobot.send_robot.dahuan_gripper import (
                REG_FORCE,
                REG_SPEED,
                DahuanGripperModbus,
            )

            def _do(arm):
                grip = DahuanGripperModbus(arm)
                grip.setup(_PrintLogger())
                grip._write_reg(REG_FORCE, self._force)
                grip._write_reg(REG_SPEED, self._speed)
                return True

            self._with_arm(_do)
            self.ready = True
            print(
                f"夹爪就绪: LB→开({GRIPPER_OPEN_POS}) RB→合({GRIPPER_CLOSE_POS}) "
                f"force={self._force} speed={self._speed}"
            )
            return True
        except Exception as exc:
            self.ready = False
            print(f"夹爪初始化失败（禁用夹爪，遥操可继续）: {exc}", file=sys.stderr)
            return False

    def set_open(self, open_: bool) -> None:
        if not self.ready:
            return
        pos = GRIPPER_OPEN_POS if open_ else GRIPPER_CLOSE_POS
        label = "开" if open_ else "合"
        try:
            from sendrobot.send_robot.dahuan_gripper import (
                MODBUS_DEVICE,
                MODBUS_PORT,
                REG_POS,
            )
            from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t

            def _do(arm):
                p = rm_peripheral_read_write_params_t(
                    port=MODBUS_PORT, address=REG_POS, device=MODBUS_DEVICE, num=1
                )
                return arm.rm_write_single_register(p, int(pos))

            code = self._with_arm(_do)
            print(f"夹爪{label}: pos={pos} code={code}")
        except Exception as exc:
            print(f"夹爪{label}失败: {exc}", file=sys.stderr)
