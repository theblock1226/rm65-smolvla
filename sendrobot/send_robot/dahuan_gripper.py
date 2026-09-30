"""大寰夹爪 Modbus RTU（经 RM 末端 RS485）。"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

REG_INIT = int(os.environ.get("DAHUAN_REG_INIT", "0x0100"), 0)
REG_FORCE = int(os.environ.get("DAHUAN_REG_FORCE", "0x0101"), 0)
REG_SPEED = int(os.environ.get("DAHUAN_REG_SPEED", "0x0102"), 0)
REG_POS = int(os.environ.get("DAHUAN_REG_POS", "0x0103"), 0)
REG_INIT_STATE = int(os.environ.get("DAHUAN_REG_INIT_STATE", "0x0200"), 0)
MODBUS_PORT = int(os.environ.get("DAHUAN_MODBUS_PORT", "1"))
MODBUS_DEVICE = int(os.environ.get("DAHUAN_DEVICE_ID", "1"))
MODBUS_BAUD = int(os.environ.get("DAHUAN_BAUD", "115200"))


if TYPE_CHECKING:
    from Robotic_Arm.rm_robot_interface import RoboticArm


class DahuanGripperModbus:
    """复用已有 RoboticArm 连接，避免多 TCP 抢线。"""

    def __init__(self, arm: RoboticArm):
        from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t

        self._arm = arm
        self._params_cls = rm_peripheral_read_write_params_t
        self._ready = False
        self._last_hw: int | None = None

    def setup(self, logger) -> None:
        code = self._arm.rm_set_tool_voltage(3)
        logger.info(f"工具端 24V: code={code}")
        time.sleep(0.3)
        code = self._arm.rm_set_modbus_mode(MODBUS_PORT, MODBUS_BAUD, 2)
        logger.info(f"Modbus RTU port={MODBUS_PORT} baud={MODBUS_BAUD}: code={code}")
        time.sleep(0.3)
        self._ensure_initialized(logger)
        self._write_reg(REG_FORCE, 50)
        self._write_reg(REG_SPEED, 50)
        self._ready = True

    def _read_reg(self, address: int) -> int | None:
        p = self._params_cls(port=MODBUS_PORT, address=address, device=MODBUS_DEVICE, num=1)
        code, value = self._arm.rm_read_holding_registers(p)
        return int(value) if code == 0 else None

    def _write_reg(self, address: int, value: int) -> bool:
        p = self._params_cls(port=MODBUS_PORT, address=address, device=MODBUS_DEVICE, num=1)
        return self._arm.rm_write_single_register(p, int(value)) == 0

    def _ensure_initialized(self, logger) -> None:
        state = self._read_reg(REG_INIT_STATE)
        logger.info(f"夹爪 init_state(0x{REG_INIT_STATE:X})={state}")
        if state == 1:
            return
        logger.info("执行大寰夹爪初始化 (0x0100=1)...")
        if not self._write_reg(REG_INIT, 1):
            logger.warning("初始化命令发送失败")
            return
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._read_reg(REG_INIT_STATE) == 1:
                logger.info("夹爪初始化完成")
                return
            time.sleep(0.3)
        logger.warning("夹爪初始化超时")

    def set_position_normalized(self, normalized: float, min_delta: float = 0.02) -> tuple[bool, int | None]:
        if not self._ready:
            return False, None
        hw = int(max(0, min(1000, round(normalized * 1000))))
        if self._last_hw is not None and abs(hw - self._last_hw) < int(min_delta * 1000):
            return True, None
        ok = self._write_reg(REG_POS, hw)
        if ok:
            self._last_hw = hw
        return ok, hw if ok else None
