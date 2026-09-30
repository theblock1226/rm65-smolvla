# DAgger-lite vs RLT Stage2（本机）

两套都已 mock 可跑；真机都还没接 SendRobot。

## 怎么跑

```bash
# DAgger-lite
PYTHONPATH=. python3 scripts/dagger_rm65_collect.py \
  --backend mock --max_episodes 3 --steps_per_episode 12 \
  --intervene_steps 4,5,6 --output_dir datasets/dagger_rm65_mock

# RLT Stage2
PYTHONPATH=. python3 scripts/train_rlt_stage2_rm65.py \
  --backend mock --auto_reward 1 --max_episodes 3 --steps_per_episode 12 \
  --warmup_episodes 1 --batch_size 8 --device cpu --intervene_steps 3,4,5
```

## 一眼差异

1. **DAgger**：产出纠错数据集，下一步是 **SFT SmolVLA**。  
2. **RLT**：产出 **actor 权重**，下一步是部署推理 / 继续在线训。  
3. **同一套按住介入 UX**（mock 用 step 列表；真机计划 button=6）。

## 建议配合顺序（已定）

底座 SFT →（可选 DAgger 多轮）→ RLT Stage1 → RLT Stage2 精修。
