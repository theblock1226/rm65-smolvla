# RM65 遥操安全入口（唯一生效路径）

## 允许

| 用途 | 命令 |
|------|------|
| 手柄验收（不动臂） | `scripts/rm65_run_movep_teleop.sh --print-only` |
| 实机笛卡尔遥操 | `scripts/rm65_run_movep_teleop.sh` |
| 关节空间录制（movep+相机） | `scripts/rm65_run_record.sh`（相机 key=primary，退出 finalize） |
| 数据集 finalize | `scripts/rm65_finalize_dataset.sh` |
| 驱动 | `ros2 launch rm_driver rm_65_driver.launch.py`（容器内） |
| 清错/上使能 | `scripts/rm65_recover_estop.sh` |
| XY 画圆 (movel) | `scripts/rm65_test_xy_circle.sh` |

语义以 `rm65_movep_teleop.py` / 备份 `scripts/base.py` 为准。默认约 **50 mm/s**。

## 第一批已删除（勿再引用）

- `scripts/run_teleop_docker.sh`
- `scripts/start_rm65_teleop_stack.sh`
- `scripts/rm65_test_cartesian.sh`
- `scripts/rm65_cartesian_path_test.py`
- `sendrobot/send_robot/play_action_test.py`
- `sendrobot/send_robot/gripper_bridge.py`

仍存在但勿作臂控主路径：`send_robot_end_effector` 的 delta 遥操（调用即抛错）。

启动脚本仍会 `pkill` 上述旧进程名（无进程则忽略）。
