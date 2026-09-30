"""Turn guided RTC on for a loaded RM65 SmolVLA policy.

Does not rewrite the checkpoint. ``num_steps`` and ``rtc_config`` live only on
the in-memory config. Relative-action flags are copied from ``config.hub.json``
when the on-disk ``config.json`` lost them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def enable_smolvla_rtc(
    policy,
    *,
    num_steps: int = 4,
    execution_horizon: int = 13,
    max_guidance_weight: float = 10.0,
    ckpt_dir: str | Path | None = None,
):
    """Enable guided RTC and the training-time relative-action postprocess."""
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.configs.types import RTCAttentionSchedule

    config = policy.config
    chunk = int(config.chunk_size)
    if execution_horizon >= chunk:
        raise ValueError(f"execution_horizon ({execution_horizon}) must be < chunk_size ({chunk})")
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    hub_path = Path(ckpt_dir) / "config.hub.json" if ckpt_dir else None
    if hub_path is not None and hub_path.is_file():
        hub = json.loads(hub_path.read_text())
        if hub.get("use_relative_actions"):
            config.use_relative_actions = True
            config.relative_exclude_joints = list(hub.get("relative_exclude_joints") or ["gripper"])

    if int(config.n_action_steps) < chunk:
        # Hysteresis otherwise freezes the gripper on the unused tail.
        config.n_action_steps = chunk
        policy.reset()

    config.num_steps = int(num_steps)
    config.rtc_config = RTCConfig(
        enabled=True,
        mode="guided",
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=float(max_guidance_weight),
        execution_horizon=int(execution_horizon),
        debug=False,
    )
    policy.init_rtc_processor()
    logger.info(
        "RTC on: num_steps=%s horizon=%s chunk=%s relative=%s",
        config.num_steps,
        config.rtc_config.execution_horizon,
        chunk,
        bool(getattr(config, "use_relative_actions", False)),
    )
    return policy
