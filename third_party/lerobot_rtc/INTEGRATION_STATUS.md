# RTC 集成状态（本 fork）

已合入本地路径（旧布局 `lerobot.common.policies`）：

| 目标 | 状态 |
|------|------|
| `lerobot/common/policies/rtc/` | 已拷贝并改 import |
| `lerobot/common/policies/common/flow_matching.py` | 已加入（`euler_integrate`） |
| `lerobot/configs/types.py` → `RTCAttentionSchedule` | 已合并 |
| `smolvla/configuration_smolvla.py` → `rtc_config` | 已加 |
| `smolvla/modeling_smolvla.py` RTC 钩子 | 已移植 |
| leftover 物理空间→模型空间 | `_prepare_rtc_prev_chunk`（含 CPU→GPU） |
| 上游 `rollout/` | **未合入**（依赖新版 processor/API，保留在本目录参考） |
| `relative.py` 完整实现 | **桩**：需 `lerobot.processor`；绝对动作 SmolVLA 不需要 |

## L1 离线验证

```bash
PYTHONPATH=. python3 scripts/l1_rtc_smolvla_check.py --device cuda
```

实测：前缀 MAE 1.37 → 0.53（约 -61%），PASS。

## L2 模拟闭环（无需真机）

```bash
PYTHONPATH=. python3 scripts/l2_rtc_action_queue_check.py --device cuda
```

通过标准：
1. ActionQueue unit：leftover 长度 / merge 跳过 delay 正确
2. RTC 闭环 underruns == 0
3. 交接处对 leftover 的 MAE：RTC < 无 RTC

实测（2026-09-20）：queue unit OK；underruns=0；leftover_head_mae **1.21 vs 21.75**，PASS。

## 推理环 profiling + 插值（等 ckpt 期间）

```bash
PYTHONPATH=. python3 scripts/profile_inference_loop.py --policy_mode mock
# client: --interpolation_multiplier=1|2
PYTHONPATH=. python3 scripts/test_waiting_tools_offline.py
```

`RobotClientConfig.interpolation_multiplier`（默认 1）已接入 `robot_client.py`。

## 用法要点

- 默认 `rtc_config=None`，现有 Async gRPC / `select_action` 行为不变。
- 启用 RTC：`RTCConfig` + `predict_action_chunk(..., prev_chunk_left_over, inference_delay)` + `ActionQueue.merge(..., delay)`。
- `prev_chunk_left_over` 可直接用 `ActionQueue.get_left_over()`（物理动作，自动归一化）。

## 参考源

- 快照：`third_party/lerobot_rtc/`（commit 见 `UPSTREAM_COMMIT.txt`）
