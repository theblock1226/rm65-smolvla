# RM65 DAgger-lite（路径 B）

官方 `lerobot-rollout --strategy.type=dagger` **未合入**（依赖新版 rollout/processor）。  
本目录实现的是 **HG-DAgger 语义的本机版**，与 RLT Stage2 共用「按住键接管」。

## 文件

| 路径 | 说明 |
|------|------|
| `upstream_ref/dagger.py` | 上游官方策略参考（只读对照） |
| `upstream_ref/rollout_configs.py` | 上游 DAgger 配置参考 |
| `../../lerobot/common/policies/dagger_rm65/` | 本机状态机 + episode 落盘 |
| `../../scripts/dagger_rm65_collect.py` | 采集入口（现支持 `--backend mock`） |

## 与 RLT Stage2 对照（一起看时用）

| | **DAgger-lite** | **RLT Stage2** |
|--|-----------------|----------------|
| 目的 | 采专家纠错数据 → 再 SFT 底座 | 在线 RL 精修小 actor |
| 按住键 | HUMAN 动作写入数据集 | HUMAN transition 进 replay |
| 松开 | 回 VLA/策略 | 回 VLA（warmup）或 Actor |
| 学习 | 离线模仿（另开 train） | 当场 TD + BC |
| 输出 | `episode_*/trajectory.npz` + `intervention` | `actor/critic.pt` |
| 默认介入键 | 同 RLT：button **6**（真机 S2-4）；mock 用 `--intervene_steps` | 同左 |

## 状态机（简化）

```
AUTONOMOUS --(hold)--> CORRECTING --(release)--> AUTONOMOUS
```

默认 `--corrections_only`：只保留含介入帧的 episode（经典 DAgger 纠错窗）。

## Mock 验收

```bash
cd /home/sdznkj/zkx_ws/sendlerobot
PYTHONPATH=. python3 scripts/test_dagger_rm65_e2e.py
```

流水线：采集 → `dagger_rm65_to_lerobot.py` → 校验 parquet/`intervention`。

单独转换：

```bash
PYTHONPATH=. python3 scripts/dagger_rm65_to_lerobot.py \
  --bundle_dir datasets/dagger_rm65_mock_e2e \
  --out_root datasets/dagger_lerobot \
  --repo_id rm65_dagger_e2e
```

## 后续（未做）

- 真机 SendRobot + 手柄按住键（与 RLT S2-4 共用 adapter）
- 与 `datasets/cleaned` 合并后再 SFT
- 多轮 DAgger：采集 → SFT → 再采集
- `use_videos=True` 的 mp4 导出（当前默认 image 嵌入 parquet，离线更稳）
