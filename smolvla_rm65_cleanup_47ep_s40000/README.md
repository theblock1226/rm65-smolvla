---
license: apache-2.0
tags:
  - lerobot
  - smolvla
  - robotics
---

# SmolVLA RM65 收拾桌面（47 条训练，40000 步，夹爪滞回）

从 `smolvla_base` 微调，第 40000 步检查点。状态 7 维，相机 `camera1`（front）和 `camera2`（astra），关节动作为相对当前状态的差分，夹爪为绝对值。

夹爪输出使用滞回：连续值降到 0.35 及以下输出 0（合上），升到 0.65 及以上输出 1（张开），中间保持上一状态。第一帧按 0.5 判定。

放置阶段的语言仍是同一句：`Put three objects from the desk into the box. Place it in the box and release.` 边缘端按数据里的原句发送，不要改成 left / middle / right。

边缘端加载前，用下面两个文件覆盖本机 `lerobot/policies/smolvla/` 里的同名文件，否则 `gripper_hysteresis_processor` 无法注册：

- `edge_patch/configuration_smolvla.py`
- `edge_patch/processor_smolvla.py`

推理时 `n_action_steps=16`，`empty_cameras=0`。关节指令为 `q_cmd[:6] = q_now[:6] + pred[:6]`，夹爪使用滞回后的绝对值。
