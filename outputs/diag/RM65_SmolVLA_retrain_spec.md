# RM65 SmolVLA 重训规格（给执行 AI / 云端训练用）

- **文档 ID**: `RM65_SmolVLA_retrain_spec_v1`
- **日期**: 2026-09-24
- **读者**: 云端改代码并重新训练的 AI 或工程师
- **不要做**: 不要在本文件里发明新架构；不要打开未实现的 Aloha delta 开关；不要从旧 checkpoint 继续 finetune
- **必须做**: 按「约定 A」把关节改成相对当前状态的 Δq；状态 7 维含夹爪；只保留两路真实相机

本文件足够单独执行。下面标识符、路径、公式以代码为准，叙述性文字只作解释。

---

## 0. 一句话目标

下一版 checkpoint 必须同时满足：

1. 模型输出的前 6 维是 **相对当前编码器的关节增量 Δq（度）**，第 7 维夹爪仍是 **绝对开合 [0,1]**。
2. `observation.state` 是 **7 维**：`[j1..j6, gripper]`，训练与推理都不截断。
3. 图像输入 **只有两路**：`observation.images.camera1`、`observation.images.camera2`。`empty_cameras=0`，删除 `camera3` 和 `empty_camera_0`。
4. 不能靠改旧权重 `astra_smolvla` 的 config 凑合。输入维度变了，必须从 `smolvla_base` **重新 SFT**。

验收通过的定义不是 loss 下降，而是：离线把预测 chunk **加回当前状态** 叠到示教轨迹上，橙峰贴人手，夹爪呈台阶。

---

## 1. 硬约束（违反即做错）

### 1.1 禁止

| 禁止 | 原因 |
|---|---|
| 把 `use_delta_joint_actions_aloha=true` | 当前 LeRobot SmolVLA 直接 `NotImplementedError`，见 `lerobot/common/policies/smolvla/configuration_smolvla.py` |
| 从 `astra_smolvla` / `usb_smolvla` 继续训 | 旧权重 state=6、含 camera3/empty_camera，对不齐新输入 |
| 把 parquet 里每行改成 `action[:6]-state[:6]` 就当作 Δq | 那是「逐步速度」（约定 B）。本规格要的是「相对 chunk 起始状态」（约定 A） |
| 推理时把 Δq 直接当绝对角发给 RM65 | 臂会乱甩。必须 `q_cmd = q_now + Δq` 之后再下发 |
| 训练时 MEAN_STD 仍用绝对关节的 mean/std | 增量尺度完全不同，必须在 Δq 变换 **之后** 重算 action 统计 |
| 给缺失相机填全零图且 mask=1 | 模型会当真实黑图来看 |
| 解冻整颗 VLM / 换更大模型 | 本轮不需要 |
| 加第三路相机或深度 | 本轮不需要 |

### 1.2 允许保持不变

- 骨架：SmolVLA，`attention_mode=cross_attn`，`freeze_vision_encoder=true`，`train_expert_only=true`，`train_state_proj=true`
- 预训练：`lerobot/smolvla_base`（或本地同等 `smolvla_base`）
- 视觉归一化：`VISUAL=IDENTITY`；状态/动作：`MEAN_STD`（但 action 统计必须基于 Δq）
- 任务语言、`rename_map`（`front→camera1`，`astra→camera2`）
- 优化器：AdamW `lr=1e-4`，cosine warmup 1000 / decay 35000，`batch_size=8`，`steps=35000`（可按数据量略调，不要低于 2 万）
- 机器人：RM65，20 Hz，动作仍 7 维下发到控制器（推理侧加回状态后变成绝对角）

---

## 2. 旧权重错在哪（对照，不要再复制）

旧 checkpoint 路径（边缘机，仅作对照，不要当 init）：

- `/home/sdznkj/zkx_ws/sendlerobot/astra_smolvla/`
- 训练配置副本：`astra_smolvla/train_config.json`
- 训练数据：`FelishLeon/rm65_astra_multitask`，本地 root 曾为 `/root/hf_datasets_download/rm65_astra_multitask`

旧配置事实：

| 项 | 旧值 | 问题 |
|---|---|---|
| `observation.state.shape` | `[6]` | 数据集和真机是 7 维，夹爪被 `RM65_POLICY_STATE_DIM=6` 截掉 |
| `action.shape` | `[7]` | 绝对关节角 + 夹爪 |
| `empty_cameras` | `1` | 多一路黑图 token |
| `observation.images.camera3` | 存在 | 真机没有这路相机 |
| `observation.images.empty_camera_0` | 存在 | Aloha 模板占位 |
| `chunk_size` / `n_action_steps` | `50` / `50` | 一次猜 2.5 s，推理还要 ~1.3 s |
| `n_obs_steps` | `1` | 可先保持 |
| `use_delta_joint_actions_aloha` | `false` | 保持 false，自己做 Δq |
| `rename_map` | front→camera1, astra→camera2 | **保留** |

离线诊断（DAgger ep000，人手 100%）：人手 `mean|Δq|≈0.05°`，旧策略 chunk `mean|Δq|≈0.42°`，约 40% 步超过 0.4° 限幅，夹爪同一 chunk 内 0/1 对打。真机评估 `speed_scale=0.3` + `max_step=0.4°`，执行每步约 0.02°。

`astra_smolvla/README.md` 写「7-DoF action/state」，与 `config.json` 的 state=6 **不一致**。以 `config.json` 为准：旧权重是 6 维状态。

堆叠子集统计（绝对角，仅说明尺度，不要当新 normalizer）：

- `datasets/cleaned/rm65_stacking_blocks`：52 ep / 32749 frames / 20 Hz
- `action` 前 6 维 std 约 `[9.7, 4.7, 14.0, 12.2, 6.2, 9.9]` 度（工作空间尺度）
- 人手逐步 |Δq| 约 0.05°。MEAN_STD 在绝对角上，归一化空间 0.1σ ≈ 1° 量级，所以橙峰会飞。

---

## 3. Δq 约定（必须用约定 A）

### 3.1 约定 A（要做）

对每个训练样本，观测时刻为 `t`，chunk 长度 `T=chunk_size`：

```text
state[t] ∈ R^7     # [q1..q6, grip]，单位：关节=度，夹爪∈[0,1]
action_abs[t : t+T] ∈ R^{T×7}   # 数据集里原来的绝对目标

action_train[k, 0:6] = action_abs[t+k, 0:6] - state[t, 0:6]   # Δq 相对「这一帧观测」
action_train[k, 6]   = action_abs[t+k, 6]                     # 夹爪绝对，不减
```

推理（unnormalize 之后、发给机器人之前）：

```text
q_cmd[k, 0:6] = q_now[0:6] + action_pred[k, 0:6]
q_cmd[k, 6]   = action_pred[k, 6]          # 再 clip 到 [0,1]
```

`q_now` 必须是 **本次推理用的那帧编码器**（与 state 输入一致），不要用过期的 chunk 起始位姿去加后续重叠 chunk。

数值例子（仅帮助理解）：

```text
t 时刻编码器 j1 = -148.5°
未来第 0 步示教绝对角 = -148.3°  →  Δq = +0.2°
未来第 49 步示教绝对角 = -147.0° →  Δq = +1.5°
夹爪示教 = 1.0  →  仍写 1.0
```

### 3.2 约定 B（不要做）

```text
action_train[k, 0:6] = action_abs[t+k, 0:6] - action_abs[t+k-1, 0:6]
```

这是逐步速度，推理必须 `cumsum`。与 SmolVLA 注释「deltas with respect to the current state」不一致。禁止把 parquet 整列改成逐步差再拿去当 chunk 标签。

### 3.3 实现位置

**保留数据集里的绝对 `action` 和 7 维 `state`，不要改 parquet 数值。**

在 **采样出 (state[t], action_abs[t:t+T]) 之后、Normalizer 拟合/变换之前** 做约定 A。推荐单独 processor，例如 `DeltaJointVsCurrentStateProcessor`：

- `joint_dims = 6`
- `gripper_dim = 6`（最后一维）
- `forward`（训练 / 算 stats）：`action[..., :6] -= state[..., :6]`（注意 state 要 broadcast 到时间维）
- `inverse`（推理 unnormalize 后）：`action[..., :6] += state[..., :6]`

必须保证：

1. 拟合 `ACTION` 的 mean/std 时，用的已经是 Δq。
2. `STATE` 的 mean/std 仍用绝对 7 维状态（不要对状态做 Δ）。
3. processor 写入 checkpoint，推理 `from_pretrained` 自动 inverse。不要只在训练脚本里偷偷减、推理忘了加。

若暂时不能做 processor：可以离线生成 **平行数据集**，每个 episode 仍存绝对角，另存 `meta/delta_mode.json` 声明约定 A，并在训练 collate 里减。禁止只改 stats.json 而不改训练 tensor。

### 3.4 变换后 sanity（训练前必跑）

在真实训练集上随机抽 N≥1000 个 chunk：

- `action_train[:, :6]` 的 **mean 绝对值应接近 0**（相对当前姿态，不是工作空间中心）
- `action_train[:, :6]` 的 std 应远小于旧绝对角 std（堆叠任务粗估逐步 ~0.05°，一个 24 步 chunk 相对起点通常是 **几度以内**，不是 10° 量级 std）
- `action_train[:, 6]` 仍大致在 {0,1} 或 [0,1]
- `inverse(forward(action_abs))` 应还原到绝对角，误差 < 1e-4

若 Δq std 仍像 `[9,5,14,...]`，说明还在用绝对角，变换没接上。

---

## 4. 状态 7 维

数据集本来就是 7 维，名字：

```text
left_joint_1.pos
left_joint_2.pos
left_joint_3.pos
left_joint_4.pos
left_joint_5.pos
left_joint_6.pos
left_gripper.pos
```

必须改：

```json
"observation.state": { "type": "STATE", "shape": [7] }
"action":            { "type": "ACTION", "shape": [7] }
```

preprocessor / postprocessor / `policy_preprocessor.json` 里的 `observation.state.shape` 一律 `[7]`。

边缘机旧环境变量（训练侧不要复制）：

```bash
RM65_POLICY_STATE_DIM=6   # 旧。新权重禁止再截断
```

新权重配套：

```bash
RM65_POLICY_STATE_DIM=7
```

所有 `state[..., :6]` 截断逻辑，对新 checkpoint 必须关闭。`max_state_dim=32` 的 pad 保持即可，有效维是前 7。

夹爪：作为 **输入状态第 7 维** 和 **输出动作第 7 维**。不要把夹爪从 state 拿掉。

可选（推荐）：训练标签把夹爪二值化 `grip = 1.0 if grip >= 0.5 else 0.0`，减少 0.4/0.6 这种中间值。推理后再 `>0.5` 视为开。这不替代 Δq。

---

## 5. 相机：只留两路，去掉 empty_camera

真机与数据只有：

| 数据 key | 策略 key | 含义 |
|---|---|---|
| `observation.images.front` | `observation.images.camera1` | 第三人称 / 胸前 |
| `observation.images.astra` | `observation.images.camera2` | Astra Pro |

`rename_map` **保留**：

```json
"rename_map": {
  "observation.images.front": "observation.images.camera1",
  "observation.images.astra": "observation.images.camera2"
}
```

必须：

```text
empty_cameras = 0
```

并从 `input_features`、preprocessor features、eval 相机 yaml **删除**：

- `observation.images.camera3`
- `observation.images.empty_camera_0`

不要为了「对齐预训练 Aloha 槽位」再垫黑图。本轮两路真实相机即可。

`verify_robot_cameras`：新 checkpoint 的 image features 必须能被 robot 两路相机满足，不要再在 eval 里声明 dummy `camera3` / `empty_camera_0`。

若训练代码 `validate_features()` 在 `empty_cameras>0` 时自动插入 empty key：设 0 后确认 `config.json` 里没有这两个 key。

---

## 6. 建议一并改的训练超参

不是失败主因，但和 1.3 s 推理、50 步乱猜有关。新实验采用：

| 项 | 旧 | 新 |
|---|---|---|
| `chunk_size` | 50 | **24** |
| `n_action_steps` | 50 | **16** |
| `num_steps`（flow 采样） | 10 | 训练保持 10；**评估**可试 4–5 |
| `n_obs_steps` | 1 | 先保持 1 |
| `resize_imgs_with_padding` | [512,512] | 保持 |
| `tokenizer_max_length` | 48 | 保持 |
| `empty_cameras` | 1 | **0** |

`n_action_steps <= chunk_size`。执行只用前 16 步，剩下 8 步当重叠缓冲。

---

## 7. 目标 `policy` 配置片段（训练用）

把旧 `train_config.json` 的 `policy` 段改成下面语义（JSON 仅列必须改动的字段，其余可从旧文件复制）：

```json
{
  "type": "smolvla",
  "n_obs_steps": 1,
  "chunk_size": 24,
  "n_action_steps": 16,
  "empty_cameras": 0,
  "adapt_to_pi_aloha": false,
  "use_delta_joint_actions_aloha": false,
  "freeze_vision_encoder": true,
  "train_expert_only": true,
  "train_state_proj": true,
  "attention_mode": "cross_attn",
  "num_steps": 10,
  "pretrained_path": "<PATH_OR_HUB_TO_smolvla_base>",
  "input_features": {
    "observation.state": { "type": "STATE", "shape": [7] },
    "observation.images.camera1": { "type": "VISUAL", "shape": [3, 256, 256] },
    "observation.images.camera2": { "type": "VISUAL", "shape": [3, 256, 256] }
  },
  "output_features": {
    "action": { "type": "ACTION", "shape": [7] }
  },
  "normalization_mapping": {
    "VISUAL": "IDENTITY",
    "STATE": "MEAN_STD",
    "ACTION": "MEAN_STD"
  }
}
```

`dataset.rename_map` 保持 front/astra 映射。不要在 `input_features` 里写 front/astra 同时又写 camera1/camera2。

job 名建议：`smolvla_rm65_astra_deltaq_2cam_s7`，避免和旧 `smolvla_rm65_astra_multitask` 混淆。

输出 checkpoint 的 `config.json` 必须能 grep 出：

- `"empty_cameras": 0`
- 没有 `camera3`、没有 `empty_camera_0`
- `"shape": [7]` 同时出现在 state 和 action

---

## 8. 训练数据

沿用 Astra 多任务合并集（place-to-slot / cleanup-the-desk / stacking-blocks），20 Hz，两路 RGB。

- Hub：`FelishLeon/rm65_astra_multitask`（若云端已有同样 root 则用本地）
- 单任务样例字段见 `datasets/cleaned/rm65_stacking_blocks/meta/info.json`：state 7、action 7、`observation.images.front`、`observation.images.astra`

不要为了 Δq 重编码视频。不要丢掉夹爪维。

新数据（可选，非阻塞）：20–30 条干净堆叠，夹爪开合干脆、少来回修正、任务字符串与评估一致。没有新数据也可以先用现有集合重训。

---

## 9. 建议的云端工作顺序

1. 复制旧 `train_config.json` 为新实验配置，按第 7 节改字段。
2. 实现约定 A processor，单测 `inverse(forward(x))=x`。
3. 用现有数据集跑一遍 stats：打印 Δq 的 mean/std；确认不是绝对角尺度。
4. 从 `smolvla_base` 启动 SFT，`steps=35000`（或等价 epoch）。
5. 导出完整 LeRobot 目录：`config.json`、权重、tokenizer、`policy_preprocessor*`、`policy_postprocessor*`、`train_config.json`。
6. 离线开环叠加（见第 10 节）。不通过不要当正式权重。
7. 把 checkpoint 交回边缘机。边缘机再改 `RM65_POLICY_STATE_DIM=7`、去掉 dummy 相机、确认 inverse 在 `predict_action_chunk` 路径上。

---

## 10. 验收（训练方在云端就能做）

### 10.1 配置体检

新目录 `config.json`：

```text
empty_cameras == 0
"camera3" not in file
"empty_camera_0" not in file
input_features["observation.state"].shape == [7]
output_features["action"].shape == [7]
chunk_size == 24
use_delta_joint_actions_aloha == false
```

preprocessor 的 state shape 也是 7，且没有 camera3。

### 10.2 动作尺度体检

对验证集或训练集若干帧 `predict_action_chunk`：

- 未加回状态的 `action_pred[:6]`：幅度应是 **度级增量**，mean 近 0，单步/单 chunk 不应出现 10°+ 的系统性偏置（除非示教本身大动）
- 加回状态后的绝对轨迹：应贴近该帧之后的示教绝对角
- 夹爪：一个 chunk 内不应长期同时出现 `<0.3` 和 `>0.7`（允许在开合沿附近短过渡）

对照旧权重（同一帧）：旧 chunk `j1` 可从约 -148 甩到 -159；新权重不应复现这种「相对示教的大弯」。

### 10.3 离线叠加图（通过线）

横轴时间，纵轴各关节绝对角：

- 实线：编码器 / 示教状态
- 虚线：人手 `action_abs`
- 叠加：`q_now + Δq_chunk`（24 步）

**通过**：叠加曲线贴虚线，形状像平滑示教，而不是一截截外戳的峰。  
**失败**：加回状态后仍飞出十几度，或没加回状态就把 Δq 当绝对角画（会画在 0 附近一条平线）。

### 10.4 不要用这些当通过

- 训练 loss 单独下降
- 真机「好像动了一点」
- attention 热力图好看

---

## 11. 交回边缘机后必须改的运行时（云端 AI 可写进 README，边缘再施）

旧脚本默认值（错误，对新权重）：

```bash
# scripts/rm65_run_astra_eval.sh
RM65_POLICY_STATE_DIM=6
actions_per_chunk=50
cameras 含 camera3 dummy 和 empty_camera_0 dummy
speed_scale=0.3
```

新权重配套：

```bash
RM65_POLICY_STATE_DIM=7
actions_per_chunk=16          # 与 n_action_steps 一致
# cameras 只留 camera1、camera2
# 不要 dummy camera3 / empty_camera_0
```

涉及文件（边缘仓库）：

- `scripts/rm65_run_astra_eval.sh`
- `scripts/rm65_run_dagger_collect.sh`
- `lerobot/common/policies/dagger_rm65/smolvla_actor.py`（`state_dim` 默认 6；不要给缺失相机填 zeros 当真实图）
- `lerobot/scripts/server/helpers.py`（`RM65_POLICY_STATE_DIM` 截断）
- `scripts/dagger_rm65_collect.py`（dummy `camera3`）

`send_robot` 继续收 **绝对角**。Δq→绝对角必须发生在 policy postprocessor / actor，不要让底层驱动去猜。

安全限幅 `RM65_MAX_JOINT_STEP_DEG=0.4` 可保留。`RM65_SPEED_SCALE=0.3` 在 Δq 版通过离线验收前不要当作「让策略变好」的旋钮；验收通过后可先 1.0，靠 0.4° 限幅保安全。

夹爪滞后（debounce=20，min_hold=2.5s）在新策略夹爪已经呈台阶之前，会锁死错误的第一次开合。离线夹爪干净之前，评估可关 `RM65_GRIPPER_HYSTERESIS=0` 做对比。

评估 `num_steps` 可试 4–5 降延迟；与训练时 10 步不完全相同，属推理加速，不替代本规格的三项输入改动。

---

## 12. 给执行 AI 的检查清单

完成时必须能逐条勾：

- [ ] 未设置 `use_delta_joint_actions_aloha=true`
- [ ] 未从 `astra_smolvla` 继续训，init 为 `smolvla_base`
- [ ] 约定 A：`action_chunk[:,:6] -= state[:6]`，夹爪不减
- [ ] ACTION 的 MEAN_STD 在 Δq 上重算；STATE 的 MEAN_STD 在绝对 7 维上
- [ ] 推理 inverse：`q_cmd[:6] = q_now[:6] + pred[:6]`
- [ ] `observation.state` shape `[7]`，无截断
- [ ] `empty_cameras=0`，无 `camera3`，无 `empty_camera_0`
- [ ] `rename_map` 仍是 front→camera1、astra→camera2
- [ ] `chunk_size=24`，`n_action_steps=16`
- [ ] 训练前打印过 Δq mean/std，确认不是 ~10° 的绝对角 std
- [ ] 离线叠加通过第 10.3 节
- [ ] 交付物含完整 preprocessor/postprocessor，边缘 `from_pretrained` 能复现 inverse

---

## 13. 机器可读摘要

```yaml
spec_id: RM65_SmolVLA_retrain_spec_v1
init_from: smolvla_base
do_not_init_from: [astra_smolvla, usb_smolvla]
delta_q:
  convention: A_minus_current_state
  formula: action[:, :6] -= state[:6]
  gripper: absolute
  aloha_flag: false
  forbidden_convention: B_per_step_diff
state_dim: 7
action_dim: 7
cameras:
  keep: [observation.images.camera1, observation.images.camera2]
  rename_map:
    observation.images.front: observation.images.camera1
    observation.images.astra: observation.images.camera2
  delete: [observation.images.camera3, observation.images.empty_camera_0]
  empty_cameras: 0
chunk_size: 24
n_action_steps: 16
n_obs_steps: 1
freeze_vision_encoder: true
train_expert_only: true
use_delta_joint_actions_aloha: false
normalization:
  visual: IDENTITY
  state: MEAN_STD          # on absolute 7-D state
  action: MEAN_STD         # on delta-q joints + abs gripper
eval_env_on_edge:
  RM65_POLICY_STATE_DIM: 7
  actions_per_chunk: 16
  dummy_cameras: false
pass_fail: offline_absolute_overlay_tracks_human
```
