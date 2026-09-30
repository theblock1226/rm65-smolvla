#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NewLineTaskProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_smolvla import SmolVLAConfig


def gripper_action_index(config: SmolVLAConfig) -> int:
    """Index of the gripper in the action vector. Falls back to the last RM65 joint dim."""
    names = list(getattr(config, "action_feature_names", None) or [])
    for index, name in enumerate(names):
        if "gripper" in str(name).lower():
            return index
    return 6


@ProcessorStepRegistry.register("gripper_threshold_processor")
@dataclass
class GripperThresholdProcessorStep(ProcessorStep):
    """Snap one action dimension to {0, 1} using a threshold in the action's own units.

    Applied on the unnormalized absolute action. Values at or above the threshold become 1.
    """

    gripper_dim: int = 6
    threshold: float = 0.5

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        updated = action.clone()
        updated[..., self.gripper_dim] = (updated[..., self.gripper_dim] >= self.threshold).to(dtype=updated.dtype)
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = updated
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}

    def transform_features(
        self, features: dict
    ) -> dict:
        return features


@ProcessorStepRegistry.register("gripper_hysteresis_processor")
@dataclass
class GripperHysteresisProcessorStep(ProcessorStep):
    """Hold the gripper at 0 or 1 until the continuous value crosses the opposite bound.

    Closed (0) is entered at or below `low`. Open (1) is entered at or above `high`.
    Values between the bounds keep the previous output. The first sample uses
    `initial_threshold`. State persists across calls until `reset()`.
    """

    gripper_dim: int = 6
    low: float = 0.35
    high: float = 0.65
    initial_threshold: float = 0.5
    _state: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._state = None

    def _next(self, value: float) -> float:
        if self._state is None:
            self._state = 1.0 if value >= self.initial_threshold else 0.0
        elif self._state >= 0.5:
            if value <= self.low:
                self._state = 0.0
        elif value >= self.high:
            self._state = 1.0
        return self._state

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        updated = action.clone()
        dim = self.gripper_dim
        if updated.ndim == 1:
            updated[dim] = self._next(float(updated[dim]))
        elif updated.ndim == 2:
            # One timestep, shape (batch, dim). The robot server calls this once per executed step.
            state = self._next(float(updated[0, dim]))
            updated[:, dim] = state
        else:
            # Chunk, shape (batch, time, dim). Walk time in execution order.
            for t in range(updated.shape[-2]):
                state = self._next(float(updated[0, t, dim]))
                updated[:, t, dim] = state
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = updated
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "gripper_dim": self.gripper_dim,
            "low": self.low,
            "high": self.high,
            "initial_threshold": self.initial_threshold,
        }

    def transform_features(self, features: dict) -> dict:
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)
    relative_enabled = bool(getattr(config, "use_relative_actions", False))
    relative_step = RelativeActionsProcessorStep(
        enabled=relative_enabled,
        exclude_joints=list(getattr(config, "relative_exclude_joints", None) or ["gripper"]),
        action_names=getattr(config, "action_feature_names", None),
    )

    # Convention A runs on raw actions, before MEAN_STD. Inverse runs after unnormalize.
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
    ]
    if relative_enabled:
        input_steps.append(relative_step)
    input_steps.append(steps.normalize)
    output_steps: list[ProcessorStep] = [steps.unnormalize]
    if relative_enabled:
        output_steps.append(
            AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
        )
    gripper_threshold = getattr(config, "gripper_output_threshold", None)
    hysteresis_low = getattr(config, "gripper_hysteresis_low", None)
    hysteresis_high = getattr(config, "gripper_hysteresis_high", None)
    if hysteresis_low is not None and hysteresis_high is not None:
        output_steps.append(
            GripperHysteresisProcessorStep(
                gripper_dim=gripper_action_index(config),
                low=float(hysteresis_low),
                high=float(hysteresis_high),
                initial_threshold=float(gripper_threshold if gripper_threshold is not None else 0.5),
            )
        )
    elif gripper_threshold is not None:
        output_steps.append(
            GripperThresholdProcessorStep(
                gripper_dim=gripper_action_index(config),
                threshold=float(gripper_threshold),
            )
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
