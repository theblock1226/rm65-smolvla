# RM65 SmolVLA 推理相关资产清单（离岗备份 / GitHub）

> 统计时间：2026-09-30  
> 根目录：`/home/sdznkj/zkx_ws/sendlerobot`  
> 原则：代码+配置+图表必带；权重/VLM/vendor 二进制建议网盘或 HF，勿塞进公开 GitHub。

---

## 体积总览（修正后、无重复计）

| 档 | 内容 | 约计 |
|----|------|------|
| **① GitHub 瘦身包** | 核心代码 + RTC/DAgger/RLT 参考 + 滞回 edge_patch + 评测图表 + ckpt 的 json/tokenizer（**不含**真机 dagger 视频、vla-perf 大目录、权重、VLM、vendor） | **~180 MB** |
| **② GitHub+产物** | ① + `third_party/vla-perf` + 真机 dagger bundle | **~6.2 GB** |
| **③ 离岗不含 VLM** | ② + vendor + 4 任务权重 + fp16/backup 权重 + RLT checkpoints | **~13.3 GB** |
| **④ 完整离岗** | ③ + `base_models/SmolVLM2-500M-Video-Instruct` | **~15.2 GB** |
| 另：清洗训练数据 `datasets/cleaned` | 离线评测/复现用，非必须推理 | **~6.5 GB** |

---

## A. 核心推理代码（~3.7 MB）— 必传

| 路径 | 说明 |
|------|------|
| `lerobot/` | fork：SmolVLA、Async server/client、RTC、DAgger、safety、**夹爪滞回**（已在 `configuration_smolvla.py` / `modeling_smolvla.py`） |
| `lerobot/scripts/server/` | `policy_server.py` `robot_client.py` `rm65_ckpt.py` `rm65_rtc.py` `helpers.py` |
| `lerobot/common/policies/smolvla/` | 策略实现 + 滞回 |
| `lerobot/common/policies/rtc/` | RTC |
| `lerobot/common/policies/dagger_rm65/` | DAgger-lite + gamepad clutch + SmolVLAActor |
| `lerobot/common/policies/smolvla_rlt/` | RLT Stage2 adapter / safety gate |
| `sendrobot/` | SendRobot、`play_action_rm65`、`record_state_test`、夹爪 Modbus、IK |
| `scripts/` | 评测/遥操/DAgger/RTC 检查/延迟 bench/可视化（见下） |
| `start_server.sh` `start_client.sh` | 旧入口（路径需改） |
| `pyproject.toml` `Makefile` `docs/` | 依赖与重训规格 |

### scripts 中与推理强相关（勿漏）

- 评测栈：`rm65_run_astra_eval.sh` `rm65_eval_prep.py` `rm65_ready.py` `rm65_ready_pose.json`
- 开环/策略：`rm65_play_policy_openloop.py` `rm65_run_play_action.sh`
- DAgger：`rm65_run_dagger_collect.sh` `dagger_rm65_*.py` `test_dagger_rm65_e2e.py` `rm65_probe_gamepad.sh`
- RTC：`l1_rtc_smolvla_check.py` `l2_rtc_action_queue_check.py` `rm65_rtc_enable_check.py`
- 夹爪：`rm65_teleop_gripper.py` `rm65_test_gripper.sh` `rm65_summarize_gripper_log.sh`
- 诊断图：`smolvla_diag_visualize.py` `eval_judge_plots.py` `eval_dataset_coverage_plots.py` `eval_stack_30ep_offline.py` `eval_astra_deltaq_offline.py`
- 延迟：`rm65_latency_breakdown.py` `profile_inference_loop.py` `rm65_fp16_*.py` `rm65_torch_compile_bench.py`
- 恢复：`rm65_recover_estop.sh` `rm65_go_home.*` `TELEOP_SAFE.md`

---

## B. 第三方扩展

| 路径 | 大小 | 建议 |
|------|------|------|
| `third_party/lerobot_rtc/` | ~0.7 MB | **必传** |
| `third_party/dagger_rm65/` | ~58 KB | **必传** |
| `third_party/RL-Token-SmolVLA/` | ~0.2 MB | 必传（RLT 参考） |
| `third_party/vla-perf/` | **~746 MB** | 工具+产物；可网盘，公开 GH 可只留代码子目录 |

---

## C. 夹爪滞回 edge_patch（~76 KB）— 必传

每个任务 ckpt 下：

- `smolvla_rm65_*/edge_patch/configuration_smolvla.py`
- `smolvla_rm65_*/edge_patch/processor_smolvla.py`

部署前覆盖本机 `lerobot/.../smolvla/` 同名文件（README 已写）。主树里也已有 hysteresis 字段，**两边都要留**。

运行时另有：`sendrobot/.../send_robot.py` 里 `RM65_GRIPPER_HYSTERESIS` / debounce / min_hold。

---

## D. 评测图表与诊断（~157 MB，已剔除权重备份）— 必传

`outputs/diag/`：

| 子目录/文件 | 约计 | 内容 |
|-------------|------|------|
| `judge_plots/` | 1.1 MB | 轨迹/延迟瀑布图 HTML |
| `dataset_plots/` | 1.5 MB | 数据覆盖图 |
| `smolvla_attn*` | ~13 MB | 注意力可视化（含 dagger/deltaq/stack30） |
| `astra_deltaq_offline/` | 1.1 MB | 差分动作离线评 |
| `stack_30ep_offline/` | ~104 MB | 30ep 离线评测产物 |
| `latency_summary/` | 0.2 MB | 降延迟尝试图+README |
| `ep000_videos/` `ep000_schedule_dryrun/` | ~30 MB | 样例视频/调度 |
| `vla_perf/` | 5 MB | 性能摘录 |
| `fp16_*.json` `torch_compile_bench.json` | 小 | bench |
| `RM65_SmolVLA_retrain_spec.*` | 小 | 规格副本 |
| `outputs/usb_eval/` | 小 | USB 评测协议样例 |
| `logs/astra_stack_*` `logs/gripper_*.log` `logs/rtc_live` `logs/ood_live_probe` | ~1 MB | 真机栈/夹爪/RTC/OOD 日志 |

**权重备份（另计，离岗必带）：**

- `outputs/diag/smolvla_rm65_stack_30ep_s1000_backup/` ~868 MB  
- `outputs/diag/smolvla_rm65_stack_30ep_s1000_fp16/` ~862 MB  

---

## E. DAgger 样本

| 路径 | 大小 | 说明 |
|------|------|------|
| `datasets/dagger_rm65_mock*` `dagger_lerobot/` | ~32 MB | 离线 e2e，可进 GH |
| `datasets/dagger_rm65_20260923_175937/` | **~5.3 GB** | 真机集；其中 `record_state.log` **约 4.5 GB** 疑似过量日志，可只留 `episode_*`（~0.8 GB） |

---

## F. Forge

本机为 **pip 包** `forge-robotics==0.2.0`（~1.3 MB）：

- `/home/sdznkj/.local/lib/python3.10/site-packages/forge/`
- 上游：https://github.com/arpitg1304/forge  

**仓库内无 forge 源码树**；离岗请 `pip download forge-robotics` 或 `pip freeze` 记下，换机重装。

---

## G. 任务权重（网盘 / HF，勿公开塞满 GH）

| 目录 | 非权重 | 权重 |
|------|--------|------|
| `smolvla_rm65_stack_30ep_s1000` | ~3.4 MB | ~865 MB |
| `smolvla_rm65_usb_100ep_s40000` | ~3.4 MB | ~865 MB |
| `smolvla_rm65_cleanup_47ep_s40000` | ~3.4 MB | ~865 MB |
| `smolvla_rm65_place_47ep_s40000` | ~3.4 MB | ~865 MB |
| **4 任务合计** | **~14 MB** | **~3.5 GB** |

每个目录至少保留：`config.json` `train_config.json` `policy_*.json` `tokenizer/` `edge_patch/` `README.md` + **`model.safetensors` + normalizer `*.safetensors`**。

`checkpoints/`（RLT）合计 **~1.8 GB**。

`base_models/SmolVLM2-500M-Video-Instruct` **~1.9 GB**（推理加载依赖）。

---

## H. Vendor / Docker

| 路径 | 大小 | 建议 |
|------|------|------|
| `vendor/Robotic_Arm` | ~9 MB | 离岗网盘（厂商 SDK） |
| `vendor/pygame*` | ~41 MB | 离岗或换机 pip 装 |
| `docker/` | 小 | 可进 GH |

换机还需：**ROS2 Humble + rm_driver**、相机号（当前常为 front=`/dev/video2`，astra=`/dev/video0`）。

---

## 推荐打包命令（示例）

```bash
# ① 瘦身：代码+图表+配置（无权重）
# 把下列路径打进 git 或 tar，并 gitignore *.safetensors / datasets/cleaned / base_models

# ③ 离岗网盘（含权重，可排除 dagger 里 4.5G 的 record_state.log）
cd /home/sdznkj/zkx_ws/sendlerobot
tar --exclude='datasets/dagger_rm65_20260923_175937/record_state.log' \
    --exclude='**/__pycache__' --exclude='**/.cache' \
    -cvf /tmp/rm65_smolvla_infer_full.tar \
    lerobot sendrobot scripts docs docker third_party vendor \
    start_*.sh pyproject.toml Makefile \
    smolvla_rm65_* outputs/diag outputs/usb_eval outputs/profiling \
    checkpoints base_models \
    datasets/dagger_rm65_mock datasets/dagger_rm65_mock_e2e datasets/dagger_lerobot \
    datasets/dagger_rm65_20260923_175937 \
    logs/astra_stack_* logs/gripper_*.log logs/rtc_live logs/ood_live_probe
```

Forge：另存 `pip download forge-robotics==0.2.0 -d /tmp/forge_wheels`

---

## 换机最小跑通

1. 装 `pip install -e ".[smolvla]"` + forge + pygame  
2. 放 ckpt + VLM；`edge_patch` 覆盖（若主树未同步）  
3. ROS `rm_driver` + vendor SDK  
4. `FRONT_DEV=2 ASTRA_DEV=0 ASTRA_CKPT=... scripts/rm65_run_astra_eval.sh`
