# LeRobot RTC backport kit

Upstream: https://github.com/huggingface/lerobot
Commit: 5aa74557f84c54d4b458f8b9643c5aa2982acfed (main tip at download time)

## Purpose
Standalone copy of Real-Time Chunking (RTC) sources so you can merge into the
existing local fork under `src/lerobot/` without a full LeRobot upgrade.

## Layout
- `src/lerobot/policies/rtc/` — RTC core package (required)
- `src/lerobot/rollout/` — rollout controller + inference modes incl. `inference/rtc.py`
- `src/lerobot/configs/types.py` — contains `RTCAttentionSchedule` (merge carefully)
- `examples/rtc/eval_dataset.py` — offline RTC eval example
- `docs/source/rtc.mdx`, `docs/source/policy_rtc_README.md`
- `tests/policies/rtc/` + SmolVLA/π0 RTC unit tests
- `upstream_policy_hooks/` — upstream modeling/config files with RTC hooks (reference only)

## How to integrate (high level)
1. Copy `src/lerobot/policies/rtc/` into your fork's `src/lerobot/policies/rtc/`.
2. Copy or selectively merge `src/lerobot/rollout/` (your fork may not have rollout yet).
3. Diff-merge `RTCAttentionSchedule` / RTC-related types from `configs/types.py`.
4. Port RTC hooks from `upstream_policy_hooks/smolvla_modeling_smolvla.py` into
   your local `modeling_smolvla.py` (search for `rtc`, `RTCConfig`, `prefix_attention`).
5. Same for π0/π0.5 only if you need those policies.
6. Keep Async Inference (gRPC PolicyServer) as-is; RTC is a separate inference path.

## Not included / still local work
- Dependency bumps (if any new packages appear in upstream pyproject)
- Wiring into your RM65 / sendrobot client scripts
- Full LeRobot version upgrade

