# rm65-smolvla

RM65 + SmolVLA 推理/评测 **代码与小资产**（GitHub 瘦身包）。

大文件（权重、清洗训练数据、完整离线评测产物、VLM、RLT checkpoint）在 Hugging Face，命名与本仓对齐。

## 命名对照

| 角色 | 平台 | 名称 |
|------|------|------|
| 代码 + 配置 + 图表 | GitHub | **`theblock1226/rm65-smolvla-code`**（本仓） |
| 四任务策略权重 | HF Models | `FelishLeon/smolvla_rm65_stack_30ep_s1000` 等 |
| 清洗数据集 | HF Datasets | `FelishLeon/rm65_stacking_blocks` / `rm65_usb_insert` / `rm65_cleanup_the_desk` / `rm65_place_to_slot` |
| 离线评测大产物 | HF Datasets | `FelishLeon/rm65-smolvla-offline-eval` |
| 额外离岗资产（VLM / RLT / fp16） | HF Datasets | `FelishLeon/rm65-smolvla-extras` |

本地 ckpt 目录名 = HF model repo 名（去掉 `FelishLeon/` 前缀），例如：

```text
./smolvla_rm65_usb_100ep_s40000  ↔  FelishLeon/smolvla_rm65_usb_100ep_s40000
```

本仓每个 `smolvla_rm65_*` 只含 `config.json` / tokenizer / `edge_patch` 等元数据；**不含** `model.safetensors`。

## 从 HF 拉齐大文件

```bash
# 权重（示例：USB）
hf download FelishLeon/smolvla_rm65_usb_100ep_s40000 --local-dir ./smolvla_rm65_usb_100ep_s40000

# 清洗数据
hf download FelishLeon/rm65_usb_insert --repo-type dataset --local-dir ./datasets/cleaned/rm65_usb_insert

# 离线评测大产物 + 额外资产
hf download FelishLeon/rm65-smolvla-offline-eval --repo-type dataset --local-dir ./outputs/diag/_from_hf_eval
hf download FelishLeon/rm65-smolvla-extras --repo-type dataset --local-dir ./_from_hf_extras
```

（国内直连不通时可设 `HF_ENDPOINT=https://hf-mirror.com`。）

## 本仓包含

- `lerobot/` `sendrobot/` `scripts/`：Async server/client、RTC、DAgger、夹爪滞回、离线评测脚本
- `third_party/{lerobot_rtc,dagger_rm65,RL-Token-SmolVLA}`
- `smolvla_rm65_*/`：元数据 + `edge_patch`（无权重）
- `outputs/diag/`：注意力图、四任务离线轨迹图、judge/dataset plots（无权重备份、无 live 大视频）
- `docs/INFER_ASSET_MANIFEST.md`：完整离岗体积清单

## 换机最小跑通

见 `docs/INFER_ASSET_MANIFEST.md`。概要：装依赖 → 拉 HF 权重+VLM → ROS `rm_driver` → `scripts/rm65_run_astra_eval.sh`。

## License

继承上游 LeRobot / 本仓既有许可证文件。
