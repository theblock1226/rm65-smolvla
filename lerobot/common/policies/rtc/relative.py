#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
"""Relative-action helpers for RTC.

Upstream uses ``lerobot.processor``. This fork re-anchors with the same rule
SmolVLA applies at postprocess time: the first ``n_relative_joints`` are deltas
from the current state, and the remaining dimensions (gripper) stay absolute.
"""

from __future__ import annotations

import torch


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    n_relative_joints: int = 6,
) -> torch.Tensor:
    """Turn absolute leftover commands back into model-space deltas.

    ``prev_actions_absolute`` is what ``predict_action_chunk`` already returned
    (joints = q + Δq). Guidance runs on the normalized deltas the flow matcher
    was trained on, so subtract the current joint position before normalizing.

    Args:
        prev_actions_absolute: ``(T, A)`` or ``(B, T, A)`` unnormalized commands.
        current_state: ``(A,)``, ``(B, A)`` or ``(B, 1, A)`` unnormalized state.
        n_relative_joints: Leading dims expressed as deltas. Gripper stays put.
    """
    if n_relative_joints <= 0:
        return prev_actions_absolute

    state = current_state.detach()
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim == 3:
        state = state[:, -1, :]
    if state.ndim != 2:
        raise ValueError(f"current_state must be (A,), (B, A) or (B, 1, A), got {tuple(current_state.shape)}")

    squeeze = prev_actions_absolute.ndim == 2
    prev = prev_actions_absolute.unsqueeze(0) if squeeze else prev_actions_absolute
    if prev.ndim != 3:
        raise ValueError(
            f"prev_actions_absolute must be (T, A) or (B, T, A), got {tuple(prev_actions_absolute.shape)}"
        )

    n = min(int(n_relative_joints), prev.shape[-1], state.shape[-1])
    out = prev.clone()
    delta = state[:, None, :n].to(device=out.device, dtype=out.dtype)
    out[..., :n] = out[..., :n] - delta
    return out.squeeze(0) if squeeze else out
