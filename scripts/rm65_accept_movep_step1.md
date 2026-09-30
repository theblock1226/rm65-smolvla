# Step1 验收：movep_canfd 笛卡尔遥操

参考 SO100 `joystick_so100.py`：主机积分绝对 xyz，左杆 XY，右杆 Z，姿态锁定。

## 前置

1. 终端 1：`rm_driver` 已启动  
2. **不要**开旧 `run_teleop_docker.sh` / `play_action_test`  
3. 确认 20cm 工作区内无障碍  

## A. 手柄读数（不动臂）

**不依赖 rm_driver / movej**。终端打印带 `<< 主要XY` / `<< 主要Z` 标签。

```bash
/home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_movep_teleop.sh --print-only
```

| 操作 | 通过标准 |
|------|----------|
| 松手 | 持续出现 `[静]` |
| 只推左摇杆 | 持续出现 `[XY]`，本段 `Δz≈0` |
| 只推右摇杆 | 持续出现 `[Z]`，本段 `Δxy≈0` |
| 长时间出现 `[XY+Z]` 且你确认只推一根杆 | FAIL（轴串扰） |

## B. 实机笛卡尔（发 movep_canfd）

```bash
/home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_movep_teleop.sh
```

脚本会先 `movej` 到就绪位，再 50Hz 发 `/rm_driver/movep_canfd_cmd`。

| ID | 任务 | 通过标准 |
|----|------|----------|
| L1-1 | 只推左杆 10s | 终端 `Δz_from_start` 绝对值 **&lt; 5mm**（肉眼看高度基本不变） |
| L1-2 | 只推右杆 10s | 终端 `Δxy_from_start` **&lt; 5mm** |
| L1-3 | 双杆回中 5s | 臂基本静止，无持续爬行 |
| L1-4 | 指令通道 | 另开终端：`ros2 topic hz /rm_driver/movep_canfd_cmd` ≈ 50；`movej_canfd_cmd` 应为 0 |

## 失败时

- 轴方向反了：改环境变量 `GAMEPAD_LEFT_X_AXIS` / `GAMEPAD_LEFT_Y_AXIS` / `GAMEPAD_Z_AXIS`，或后续加 invert 开关  
- 臂不动：确认已停 `play_action_test`，且 `rm_driver` 在跑  
- 乱动/报警：Ctrl+C，回就绪位后再试；勿同时开旧关节遥操  

## 本步明确不做

夹爪 Modbus、相机录制、LeRobot `send_robot_end_effector` 改造（Step2+）。
