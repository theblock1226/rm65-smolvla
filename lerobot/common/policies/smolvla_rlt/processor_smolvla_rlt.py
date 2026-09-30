"""Processor stub for local fork (no lerobot.processor package).

Upstream RLT uses LeRobot v0.4.4 processor pipelines. Stage1/2 scripts in this
fork call SmolVLA prepare_* helpers directly, so a full processor is optional.
"""

from __future__ import annotations

from typing import Any


def make_smolvla_rlt_pre_post_processors(config, dataset_stats=None) -> tuple[Any, Any]:
    raise NotImplementedError(
        "smolvla_rlt processors need lerobot.processor (LeRobot >=0.4). "
        "Use scripts/train_rlt_stage1_local.py which bypasses processors."
    )
