# RLT（RL Token）集成状态 — 本 fork

上游参考：https://github.com/RajatDandekar/RL-Token-SmolVLA  
快照：`third_party/RL-Token-SmolVLA/`

## 已合入

| 项 | 路径 |
|----|------|
| `extract_embeddings()` | `lerobot/common/policies/smolvla/modeling_smolvla.py` |
| RLT 模块 | `lerobot/common/policies/smolvla_rlt/` |
| factory 注册 | `smolvla_rlt` → `SmolVLARLTPolicy` |
| Stage1 脚本（本地） | `scripts/train_rlt_stage1_local.py` |
| 离线数据集加载 | `smolvla_rlt/local_dataset.py`（禁止 Hub） |
| 观测对齐 | `smolvla_rlt/obs_batch.py` |
| 默认 action/state dim | **7**（6 关节 + 夹爪） |

## Stage1

- 数据：只用 `datasets/cleaned/rm65_*`（7 维、640×480）
- `--mock_vla`：云端 ckpt 未到时的管线 dry-run
- 真训：`--smolvla_path <RM65_SmolVLA>`（需与 cleaned 同分布）

```bash
PYTHONPATH=. python3 scripts/train_rlt_stage1_local.py \
  --mock_vla --dataset_root datasets/cleaned/rm65_stacking_blocks \
  --episodes 0 --steps 8 --batch_size 2 --device cpu
```

## Stage2（RM65）

| 阶段 | 状态 | 怎么跑 |
|------|------|--------|
| S2-0..S2-2 | 已过 | `--backend mock --policy_mode mock` |
| S2-3 观测对齐 | 已过（plumbing） | `--policy_mode plumbing` |
| S2-4 SendRobot | dry_run 已过 | `--backend send_robot --send_robot_dry_run` |
| S2-5 真机开训 | 部分就绪 | 云端 ckpt 已到：`usb_smolvla` / `astra_smolvla` |

### 相机与冻结 VLA 族（RLT 配置）

| 任务 | 相机 | SmolVLA |
|------|------|---------|
| USB 插入 | **front + mono** | `usb_smolvla` |
| desk / slot / stack | **front + astra** | `astra_smolvla` |

- `SmolVLARLTConfig`: `vlm_hidden_dim=960`, `expert_hidden_dim=720`（与本机 extract_embeddings 一致）
- `--task_family usb|desk|slot|stack|astra` 会设好 `camera_set` + 默认 ckpt 路径
- **推理侧 camera1/2/3 remap 未做**（按约定稍后改）

```bash
# Stage2 mock，USB 相机组
PYTHONPATH=. python3 scripts/train_rlt_stage2_rm65.py \
  --backend mock --policy_mode plumbing --task_family usb --auto_reward 1 \
  --max_episodes 2 --steps_per_episode 8 --device cpu
```

一键离线验收：

```bash
PYTHONPATH=. python3 scripts/test_rlt_remaining_offline.py
```

## 未合入 / 仍阻塞

| 项 | 原因 |
|----|------|
| Stage1 真 `extract_embeddings` | 等云端 RM65 SmolVLA |
| `--policy_mode real` | 同上 + Stage1 ckpt |
| SendRobot `dry_run=False` | 要 ROS / 真机 |
| `processor_smolvla_rlt` | 本 fork 无 `lerobot.processor` → 桩 |
| 上游 `install.sh` | **不要跑** |

## 冒烟

```bash
PYTHONPATH=. python3 -c "from lerobot.common.policies.smolvla_rlt import SmolVLARLTConfig; print(SmolVLARLTConfig())"
```
