#!/usr/bin/env python3
"""RLT Stage 2 for RM65 — online RL with hold-to-intervene (gamepad).

Phases:
  S2-0..S2-2  ``--backend mock --policy_mode mock``
  S2-3        ``--policy_mode plumbing`` (obs→batch→z_rl path, no real VLA)
  S2-4        ``--backend send_robot`` (dry_run, no ROS) or real SendRobot later
  S2-5        real SmolVLA + Stage1 (``--policy_mode real``) when cloud ckpt ready

Example dry-run (S2-0..2)::

    PYTHONPATH=. python3 scripts/train_rlt_stage2_rm65.py \\
      --backend mock --policy_mode mock --auto_reward 1 \\
      --max_episodes 3 --steps_per_episode 12 --warmup_episodes 1 --device cpu

S2-3 + S2-4 offline::

    PYTHONPATH=. python3 scripts/train_rlt_stage2_rm65.py \\
      --backend send_robot --policy_mode plumbing --auto_reward 1 \\
      --max_episodes 2 --steps_per_episode 10 --warmup_episodes 1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHUTDOWN = False


def _signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown requested — finishing current episode...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Replay ──────────────────────────────────────────────────────────────────


@dataclass
class Transition:
    z_rl: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    ref_actions: np.ndarray
    reward: float
    next_z_rl: np.ndarray
    next_state: np.ndarray
    next_ref_actions: np.ndarray
    done: float
    is_intervention: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "z_rl": torch.as_tensor(np.stack([t.z_rl for t in batch]), dtype=torch.float32),
            "state": torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32),
            "actions": torch.as_tensor(np.stack([t.actions for t in batch]), dtype=torch.float32),
            "ref_actions": torch.as_tensor(
                np.stack([t.ref_actions for t in batch]), dtype=torch.float32
            ),
            "rewards": torch.as_tensor([[t.reward] for t in batch], dtype=torch.float32),
            "next_z_rl": torch.as_tensor(
                np.stack([t.next_z_rl for t in batch]), dtype=torch.float32
            ),
            "next_state": torch.as_tensor(
                np.stack([t.next_state for t in batch]), dtype=torch.float32
            ),
            "next_ref_actions": torch.as_tensor(
                np.stack([t.next_ref_actions for t in batch]), dtype=torch.float32
            ),
            "dones": torch.as_tensor([[t.done] for t in batch], dtype=torch.float32),
            "is_intervention": torch.as_tensor(
                [[float(t.is_intervention)] for t in batch], dtype=torch.float32
            ),
        }

    @property
    def intervention_count(self) -> int:
        return sum(1 for t in self.buffer if t.is_intervention)

    def __len__(self) -> int:
        return len(self.buffer)


def get_human_reward(*, had_intervention: bool, auto_reward: bool | None) -> tuple[float, bool]:
    """Return (reward, should_continue). auto_reward: True/False skip stdin; None ask."""
    if auto_reward is not None:
        if had_intervention:
            return 1.0, True
        return (1.0 if auto_reward else 0.0), True
    if had_intervention:
        logger.info("  Episode had intervention → auto SUCCESS (reward=1)")
        return 1.0, True
    while True:
        response = input("\n>>> Episode successful? [y/n/q]: ").strip().lower()
        if response in ("y", "yes", "1"):
            return 1.0, True
        if response in ("n", "no", "0"):
            return 0.0, True
        if response in ("q", "quit"):
            return 0.0, False
        print("  enter y / n / q")


# USB insert: wrist is joint 6; joint 2 rises after the grasp in cleaned demos.
WRIST_INDEX = 5
GRIPPER_INDEX = 6
LIFT_JOINT_INDEX = 1


def alignment_reward(angle_deg: float, target_deg: float, sigma_deg: float) -> float:
    """Map wrist error to (0, 1]. sigma_deg is the decay scale in degrees."""
    sigma = max(float(sigma_deg), 1e-6)
    return math.exp(-abs(float(angle_deg) - float(target_deg)) / sigma)


def usb_align_active_mask(
    states: list[np.ndarray],
    *,
    gripper_closed_max: float,
    lift_delta_deg: float,
    lift_joint_index: int = LIFT_JOINT_INDEX,
) -> list[bool]:
    """Steps after the gripper has closed and the lift joint has risen.

    Closing alone is the grasp. The align term stays off until ``lift_joint``
    has increased by ``lift_delta_deg`` from its angle at that grasp, then stays
    on while the gripper remains closed. Opening the gripper resets the phase.
    """
    active = [False] * len(states)
    grasp_lift: float | None = None
    lifted = False
    for i, st in enumerate(states):
        if st.shape[0] <= max(GRIPPER_INDEX, lift_joint_index, WRIST_INDEX):
            continue
        closed = float(st[GRIPPER_INDEX]) <= gripper_closed_max
        if not closed:
            grasp_lift = None
            lifted = False
            continue
        if grasp_lift is None:
            grasp_lift = float(st[lift_joint_index])
        if float(st[lift_joint_index]) - grasp_lift >= lift_delta_deg:
            lifted = True
        active[i] = lifted
    return active


def assign_episode_rewards(
    transitions: list[Transition],
    *,
    success: float,
    intervention_bonus: float,
    align_target_j6: float | None,
    align_weight: float,
    align_sigma_deg: float,
    gripper_closed_max: float,
    lift_delta_deg: float,
) -> dict[str, float]:
    """Sparse success on the last step, intervention bonus, optional USB wrist term.

    Without ``align_target_j6`` the rewards match the original Stage2 labels.
    With a target, steps in the post-lift closed-gripper phase add
    ``align_weight * exp(-|j6 - target| / sigma)``.
    """
    n = len(transitions)
    use_align = align_target_j6 is not None and n > 0
    mask = (
        usb_align_active_mask(
            [tr.state for tr in transitions],
            gripper_closed_max=gripper_closed_max,
            lift_delta_deg=lift_delta_deg,
        )
        if use_align
        else [False] * n
    )
    align_sum = 0.0
    align_steps = 0
    for i, tr in enumerate(transitions):
        terminal = i == n - 1
        tr.done = 1.0 if terminal else 0.0
        reward = float(success) if terminal else 0.0
        if (not terminal) and tr.is_intervention:
            reward += intervention_bonus
        if mask[i]:
            bonus = align_weight * alignment_reward(float(tr.state[WRIST_INDEX]), align_target_j6, align_sigma_deg)
            reward += bonus
            align_sum += bonus
            align_steps += 1
        tr.reward = reward
    return {
        "align_steps": float(align_steps),
        "align_reward_mean": (align_sum / align_steps) if align_steps else 0.0,
    }


# ── Policy helpers (mock or real) ────────────────────────────────────────────


class MockPolicyBundle:
    """Stand-in when SmolVLA / Stage1 are not available yet (S2-0..S2-2)."""

    def __init__(self, cfg, device: torch.device, seed: int = 0):
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.last_batch_image_shapes: dict[str, tuple[int, ...]] = {}

    def ref_and_z(self, state: np.ndarray, images=None, task: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        ref = torch.as_tensor(
            self.rng.normal(0, 0.05, size=(1, self.cfg.n_action_steps_rl, self.cfg.action_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        z = torch.as_tensor(
            np.concatenate(
                [
                    state.astype(np.float32),
                    self.rng.normal(0, 1, size=self.cfg.rlt_hidden_dim - state.shape[0]).astype(
                        np.float32
                    ),
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return ref, z


class PlumbingPolicyBundle:
    """S2-3: exercise obs→batch→encoder path without a real SmolVLA ckpt.

    Builds the same batch keys as RealBundle, then feeds synthetic embeddings
    through a live ``RLTokenEncoder`` so shapes match Stage2 actor/critic.
    """

    def __init__(self, cfg, device: torch.device, seed: int = 0, expected_image_keys=None):
        from lerobot.common.policies.smolvla_rlt.modeling_smolvla_rlt import RLTokenEncoder

        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.expected_image_keys = expected_image_keys
        self.encoder = RLTokenEncoder(cfg).to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.last_batch_image_shapes: dict[str, tuple[int, ...]] = {}
        self.n_calls = 0

    def ref_and_z(self, state: np.ndarray, images=None, task: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        from lerobot.common.policies.smolvla_rlt.obs_batch import build_vla_batch, describe_batch_images

        images = images or {}
        batch = build_vla_batch(
            state=state,
            images=images,
            task=task or "task",
            device=self.device,
            expected_image_keys=self.expected_image_keys,
        )
        self.last_batch_image_shapes = describe_batch_images(batch)
        if not self.last_batch_image_shapes:
            raise RuntimeError("S2-3 plumbing: no observation.images.* in batch")

        bsz = 1
        # Synthetic VLM/expert tokens conditioned on state + image energy
        img_energy = 0.0
        for t in batch.values():
            if isinstance(t, torch.Tensor) and t.ndim == 4:
                img_energy += float(t.mean().detach().cpu())
        state_t = batch["observation.state"]
        base = state_t.mean(dim=-1, keepdim=True) + 0.01 * img_energy
        vlm = torch.randn(bsz, 32, self.cfg.vlm_hidden_dim, device=self.device) + base.unsqueeze(-1)
        expert = torch.randn(bsz, 16, self.cfg.expert_hidden_dim, device=self.device) + base.unsqueeze(
            -1
        )
        with torch.no_grad():
            z_rl = self.encoder(vlm, expert)
        ref = torch.as_tensor(
            self.rng.normal(0, 0.05, size=(1, self.cfg.n_action_steps_rl, self.cfg.action_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        # Bias first action toward current state so executed cmds stay in range
        ref = ref + state_t.unsqueeze(1) * 0.05
        self.n_calls += 1
        return ref, z_rl


def load_real_policy_bundle(args, rlt_config, device: torch.device):
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
    from lerobot.common.policies.smolvla_rlt.modeling_smolvla_rlt import RLTokenEncoder
    from lerobot.common.policies.smolvla_rlt.obs_batch import build_vla_batch, describe_batch_images

    from lerobot.scripts.server.rm65_ckpt import (
        load_dataset_stats_from_ckpt,
        load_offline_smolvla_config,
        trim_stats_to_config,
    )

    logger.info("Loading SmolVLA from %s", args.smolvla_path)
    vla_config = load_offline_smolvla_config(args.smolvla_path)
    dataset_stats = trim_stats_to_config(load_dataset_stats_from_ckpt(args.smolvla_path), vla_config)
    if not dataset_stats:
        raise SystemExit(f"No normalizer stats in {args.smolvla_path}")
    smolvla_policy = SmolVLAPolicy.from_pretrained(
        args.smolvla_path, config=vla_config, dataset_stats=dataset_stats
    )
    smolvla_policy.to(device)
    smolvla_policy.eval()
    for p in smolvla_policy.parameters():
        p.requires_grad = False
    vla = smolvla_policy.model

    logger.info("Loading Stage1 encoder from %s", args.rlt_checkpoint)
    ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=False)
    encoder = RLTokenEncoder(rlt_config).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    expected_keys = list(getattr(smolvla_policy.config, "image_features", {}) or {})

    class RealBundle:
        def __init__(self):
            self.last_batch_image_shapes: dict[str, tuple[int, ...]] = {}
            self.n_calls = 0

        def ref_and_z(self, state: np.ndarray, images: dict[str, np.ndarray], task: str):
            batch = build_vla_batch(
                state=state,
                images=images,
                task=task,
                device=device,
                expected_image_keys=expected_keys or None,
            )
            self.last_batch_image_shapes = describe_batch_images(batch)
            batch = smolvla_policy.normalize_inputs(batch)
            imgs, masks = smolvla_policy.prepare_images(batch)
            state_t = smolvla_policy.prepare_state(batch)
            lang_tokens, lang_masks = smolvla_policy.prepare_language(batch)
            with torch.no_grad():
                ref_full = vla.sample_actions(imgs, masks, lang_tokens, lang_masks, state_t)
                ref_full = ref_full[:, :, : rlt_config.action_dim]
                stride = rlt_config.action_stride
                ref_sub = ref_full[:, ::stride, :][:, : rlt_config.n_action_steps_rl, :]
                dummy = pad_vector(ref_full, rlt_config.max_action_dim)
                vlm_emb, expert_emb = vla.extract_embeddings(
                    imgs, masks, lang_tokens, lang_masks, state_t, dummy
                )
                z_rl = encoder(vlm_emb.float(), expert_emb.float())
            self.n_calls += 1
            if self.n_calls == 1:
                logger.info(
                    "real bundle z_rl=%s ref=%s images=%s",
                    tuple(z_rl.shape),
                    tuple(ref_sub.shape),
                    self.last_batch_image_shapes,
                )
            return ref_sub, z_rl

    return RealBundle()


def parse_args():
    p = argparse.ArgumentParser(description="RLT Stage2 RM65 (mock/real)")
    p.add_argument(
        "--backend",
        choices=["mock", "send_robot"],
        default="mock",
        help="mock = in-process fake arm; send_robot = SendRobot adapter (dry_run by default)",
    )
    p.add_argument(
        "--send_robot_dry_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When backend=send_robot: skip ROS (default True for offline)",
    )
    p.add_argument(
        "--task_family",
        choices=["usb", "astra", "desk", "slot", "stack"],
        default="",
        help="Shortcut: sets camera_set + default SmolVLA family "
        "(usb→front+mono+usb_smolvla; desk/slot/stack/astra→front+astra+astra_smolvla)",
    )
    p.add_argument("--camera_set", choices=["desk", "slot", "stack", "usb", "default"], default="stack")
    p.add_argument("--smolvla_path", type=str, default="")
    p.add_argument("--rlt_checkpoint", type=str, default="")
    p.add_argument("--task", type=str, default="Stack three cylinder blocks.")
    p.add_argument(
        "--policy_mode",
        choices=["mock", "plumbing", "real"],
        default="mock",
        help="mock=S2-0..2; plumbing=S2-3 obs path; real=SmolVLA+Stage1 (needs ckpt)",
    )
    p.add_argument("--auto_reward", type=str, default="1", help="1/0/ask — skip stdin in dry-run")
    p.add_argument("--max_episodes", type=int, default=3)
    p.add_argument("--steps_per_episode", type=int, default=16)
    p.add_argument("--warmup_episodes", type=int, default=1)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--output_dir", type=str, default="checkpoints/rlt_stage2_rm65")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--buffer_capacity", type=int, default=10_000)
    p.add_argument("--utd_ratio", type=int, default=2)
    p.add_argument("--critic_per_actor", type=int, default=2)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--bc_weight", type=float, default=0.1)
    p.add_argument("--target_tau", type=float, default=0.005)
    p.add_argument("--intervention_reward_bonus", type=float, default=0.5)
    p.add_argument(
        "--align_target_j6",
        type=float,
        default=None,
        help="USB only: desired left_joint_6 in degrees for this base orientation. "
        "Unset keeps the original sparse reward.",
    )
    p.add_argument(
        "--align_reward_weight",
        type=float,
        default=0.3,
        help="Weight on exp(-|j6-target|/sigma) after grasp-and-lift",
    )
    p.add_argument(
        "--align_sigma_deg",
        type=float,
        default=6.0,
        help="Wrist-error scale in degrees; a few degrees of miss decays the align term",
    )
    p.add_argument(
        "--gripper_closed_max",
        type=float,
        default=0.5,
        help="Gripper at or below this counts as closed (0=closed, 1=open)",
    )
    p.add_argument(
        "--lift_delta_deg",
        type=float,
        default=5.0,
        help="After grasp, left_joint_2 must rise by this many degrees before the align term turns on",
    )
    p.add_argument(
        "--intervene_steps",
        type=str,
        default="3,4,5",
        help="Comma-separated step indices with forced hold intervention (mock/dry_run)",
    )
    p.add_argument("--intervene_button", type=int, default=6)
    p.add_argument(
        "--deadman_button",
        type=int,
        default=5,
        help="Hold-to-arm button (separate from intervene; default 5)",
    )
    p.add_argument(
        "--safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SafeActionGate (rate limit + optional deadman/watchdog)",
    )
    p.add_argument(
        "--require_deadman",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require deadman held to move (default: on for real send_robot, off for mock/dry_run)",
    )
    p.add_argument(
        "--speed_scale",
        type=float,
        default=0.3,
        help="Shrink commanded joint step after Δq clip (0–1; first real sessions)",
    )
    p.add_argument("--max_delta_joints", type=float, default=2.0)
    p.add_argument("--max_delta_gripper", type=float, default=0.08)
    p.add_argument("--watchdog_timeout_s", type=float, default=0.25)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--state_dim", type=int, default=7)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from lerobot.common.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.common.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP, RLCriticMLP
    from lerobot.common.policies.smolvla_rlt.robot_adapter import (
        CAMERA_SETS,
        MockRM65Adapter,
        SendRobotRM65Adapter,
        resolve_camera_set,
        resolve_smolvla_family,
    )
    from lerobot.common.policies.smolvla_rlt.safety import SafetyConfig

    camera_set = args.camera_set
    smolvla_family = "astra"
    if args.task_family:
        smolvla_family = resolve_smolvla_family(args.task_family)
        camera_set = resolve_camera_set(args.task_family)
        logger.info(
            "task_family=%s → smolvla_family=%s camera_set=%s cams=%s",
            args.task_family,
            smolvla_family,
            camera_set,
            CAMERA_SETS[camera_set],
        )

    rlt_config = SmolVLARLTConfig(
        mode="online_rl",
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        discount=args.discount,
        bc_weight=args.bc_weight,
        target_tau=args.target_tau,
        camera_set=camera_set,
        smolvla_family=smolvla_family,
        smolvla_pretrained_path=args.smolvla_path or "",
    )
    # Re-read after config post_init may normalize camera_set / default path
    camera_set = rlt_config.camera_set
    if not args.smolvla_path:
        args.smolvla_path = rlt_config.smolvla_pretrained_path

    # Software safety: mock/dry_run default deadman off; real send_robot on
    if args.require_deadman is None:
        require_deadman = args.backend == "send_robot" and not args.send_robot_dry_run
    else:
        require_deadman = bool(args.require_deadman)
    safety_cfg = SafetyConfig(
        enabled=bool(args.safety),
        require_deadman=require_deadman,
        deadman_button=args.deadman_button,
        max_delta_joints=args.max_delta_joints,
        max_delta_gripper=args.max_delta_gripper,
        speed_scale=args.speed_scale,
        watchdog_timeout_s=args.watchdog_timeout_s,
        action_dim=args.action_dim,
    )

    intervene_steps = {int(x) for x in args.intervene_steps.split(",") if x.strip() != ""}
    if args.backend == "mock":
        robot = MockRM65Adapter(
            camera_set=camera_set,
            intervene_schedule=intervene_steps,
            seed=args.seed,
            safety=safety_cfg,
        )
    elif args.backend == "send_robot":
        robot = SendRobotRM65Adapter(
            camera_set=camera_set,
            intervene_button=args.intervene_button,
            deadman_button=args.deadman_button,
            dry_run=args.send_robot_dry_run,
            seed=args.seed,
            safety=safety_cfg,
            require_deadman=require_deadman,
        )
    else:
        raise NotImplementedError(f"backend={args.backend}")

    expected_img_keys = [
        f"observation.images.{n}" for n in CAMERA_SETS.get(camera_set, CAMERA_SETS["default"])
    ]
    if args.policy_mode == "mock":
        logger.info("Using MockPolicyBundle")
        policy_bundle = MockPolicyBundle(rlt_config, device, seed=args.seed)
    elif args.policy_mode == "plumbing":
        logger.info("Using PlumbingPolicyBundle (S2-3 obs alignment, no SmolVLA)")
        policy_bundle = PlumbingPolicyBundle(
            rlt_config, device, seed=args.seed, expected_image_keys=expected_img_keys
        )
    elif args.policy_mode == "real":
        if not (args.smolvla_path and args.rlt_checkpoint):
            raise SystemExit("--policy_mode real requires --smolvla_path and --rlt_checkpoint")
        policy_bundle = load_real_policy_bundle(args, rlt_config, device)
    else:
        raise SystemExit(f"unknown policy_mode={args.policy_mode}")

    actor = RLActorMLP(rlt_config).to(device)
    critic = RLCriticMLP(rlt_config).to(device)
    critic_target = RLCriticMLP(rlt_config).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    buffer = ReplayBuffer(args.buffer_capacity)
    robot.connect()

    if args.auto_reward == "ask":
        auto_reward: bool | None = None
    else:
        auto_reward = args.auto_reward.strip() not in ("0", "false", "False", "n")

    episode_rewards: list[float] = []
    episode_interventions: list[int] = []
    total_updates = 0
    sources_seen: set[str] = set()
    target_dt = 1.0 / args.fps

    try:
        for episode in range(1, args.max_episodes + 1):
            if SHUTDOWN:
                break
            logger.info("=" * 60)
            logger.info("Episode %d/%d", episode, args.max_episodes)
            if hasattr(robot, "reset_episode"):
                robot.reset_episode()
            elif hasattr(robot, "intervention"):
                robot.intervention.reset()

            episode_transitions: list[Transition] = []
            episode_had_intervention = False
            intervention_steps = 0
            prev_z = prev_state = prev_ref = prev_exec = None
            prev_was_interv = False

            for step in range(args.steps_per_episode):
                if SHUTDOWN:
                    break
                t0 = time.time()
                # Scripted hold windows for mock schedule / send_robot dry_run
                if args.backend == "send_robot" and args.send_robot_dry_run:
                    robot.set_force_intervention(step in intervene_steps)
                obs = robot.get_observation()
                is_interv = robot.poll_intervention()
                if is_interv:
                    episode_had_intervention = True
                    intervention_steps += 1

                ref_actions, z_rl = policy_bundle.ref_and_z(obs.state, obs.images, args.task)

                state_t = torch.as_tensor(obs.state[: rlt_config.state_dim], device=device)

                if is_interv:
                    human = torch.as_tensor(robot.human_action(), device=device)
                    executed = human.view(1, 1, -1).expand(1, rlt_config.n_action_steps_rl, -1)
                    source = "HUMAN"
                elif episode <= args.warmup_episodes:
                    executed = ref_actions
                    source = "VLA"
                else:
                    with torch.no_grad():
                        mean, _ = actor(
                            z_rl, state_t.unsqueeze(0), ref_actions, training=False
                        )
                    executed = mean
                    source = "ACTOR"
                sources_seen.add(source)

                first = executed[0, 0].detach().cpu().numpy()
                robot.send_action(first)

                if prev_z is not None:
                    episode_transitions.append(
                        Transition(
                            z_rl=prev_z.detach().cpu().numpy().squeeze(),
                            state=prev_state.detach().cpu().numpy(),
                            actions=prev_exec.detach().cpu().numpy().squeeze(),
                            ref_actions=prev_ref.detach().cpu().numpy().squeeze(),
                            reward=0.0,
                            next_z_rl=z_rl.detach().cpu().numpy().squeeze(),
                            next_state=state_t.detach().cpu().numpy(),
                            next_ref_actions=ref_actions.detach().cpu().numpy().squeeze(),
                            done=0.0,
                            is_intervention=prev_was_interv,
                        )
                    )

                prev_z, prev_state = z_rl, state_t
                prev_ref, prev_exec = ref_actions, executed
                prev_was_interv = is_interv

                if step % 5 == 0:
                    logger.info(
                        "  step %d | %s | interventions=%d",
                        step,
                        source,
                        intervention_steps,
                    )

                sleep = target_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(min(sleep, 0.05 if args.backend == "mock" else sleep))

            reward, should_continue = get_human_reward(
                had_intervention=episode_had_intervention, auto_reward=auto_reward
            )
            align_stats = assign_episode_rewards(
                episode_transitions,
                success=reward,
                intervention_bonus=args.intervention_reward_bonus,
                align_target_j6=args.align_target_j6,
                align_weight=args.align_reward_weight,
                align_sigma_deg=args.align_sigma_deg,
                gripper_closed_max=args.gripper_closed_max,
                lift_delta_deg=args.lift_delta_deg,
            )
            for tr in episode_transitions:
                buffer.add(tr)
            if args.align_target_j6 is not None:
                logger.info(
                    "  align steps=%d mean_bonus=%.3f target_j6=%.2f",
                    int(align_stats["align_steps"]),
                    align_stats["align_reward_mean"],
                    args.align_target_j6,
                )

            episode_rewards.append(reward)
            episode_interventions.append(intervention_steps)

            # RL updates
            if episode > args.warmup_episodes and len(buffer) >= args.batch_size:
                n_updates = max(1, args.utd_ratio * len(episode_transitions))
                critic_loss_v = 0.0
                for update_idx in range(n_updates):
                    batch = {k: v.to(device) for k, v in buffer.sample(args.batch_size).items()}
                    with torch.no_grad():
                        next_mean, _ = actor(
                            batch["next_z_rl"], batch["next_state"], batch["next_ref_actions"]
                        )
                        target_q = critic_target.q_min(
                            batch["next_z_rl"], batch["next_state"], next_mean
                        )
                        td_target = batch["rewards"] + args.discount * (1.0 - batch["dones"]) * target_q
                    q1, q2 = critic(batch["z_rl"], batch["state"], batch["actions"])
                    critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()
                    critic_loss_v = float(critic_loss.detach().item())

                    if update_idx % args.critic_per_actor == 0:
                        mean, _ = actor(
                            batch["z_rl"], batch["state"], batch["ref_actions"], training=True
                        )
                        q_loss = -critic.q_min(batch["z_rl"].detach(), batch["state"], mean).mean()
                        bc_target = batch["ref_actions"].clone()
                        mask = batch["is_intervention"].bool().squeeze(-1)
                        if mask.any():
                            bc_target[mask] = batch["actions"][mask]
                        actor_loss = q_loss + args.bc_weight * F.mse_loss(mean, bc_target)
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_opt.step()

                    tau = args.target_tau
                    for p, tp in zip(critic.parameters(), critic_target.parameters()):
                        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
                    total_updates += 1
                logger.info("  RL updates=%d critic_loss=%.4f", n_updates, critic_loss_v)

            logger.info(
                "  ep %d reward=%.0f interv_steps=%d buffer=%d (human=%d) updates=%d",
                episode,
                reward,
                intervention_steps,
                len(buffer),
                buffer.intervention_count,
                total_updates,
            )

            if episode % args.save_every == 0 or episode == args.max_episodes:
                ckpt = {
                    "episode": episode,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "critic_target_state_dict": critic_target.state_dict(),
                    "episode_rewards": episode_rewards,
                    "episode_interventions": episode_interventions,
                    "total_updates": total_updates,
                    "sources_seen": sorted(sources_seen),
                    "action_dim": args.action_dim,
                    "state_dim": args.state_dim,
                }
                path = output_dir / f"checkpoint_ep{episode}.pt"
                torch.save(ckpt, path)
                logger.info("  saved %s", path)

            if not should_continue:
                break
    finally:
        robot.disconnect()

    batch_shapes = getattr(policy_bundle, "last_batch_image_shapes", {}) or {}
    plumbing_calls = int(getattr(policy_bundle, "n_calls", 0) or 0)
    sent_joint_ok = True
    if args.backend == "send_robot":
        sent_joint_ok = hasattr(robot, "sent_joint_dicts") and len(robot.sent_joint_dicts) > 0
        if sent_joint_ok:
            jd = robot.sent_joint_dicts[0]
            from lerobot.common.policies.smolvla_rlt.robot_adapter import RM65_JOINT_KEYS

            sent_joint_ok = all(k in jd for k in RM65_JOINT_KEYS)

    report = {
        "episodes": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_interventions": episode_interventions,
        "total_updates": total_updates,
        "buffer_size": len(buffer),
        "intervention_transitions": buffer.intervention_count,
        "sources_seen": sorted(sources_seen),
        "backend": args.backend,
        "policy_mode": args.policy_mode,
        "batch_image_shapes": {k: list(v) for k, v in batch_shapes.items()},
        "plumbing_calls": plumbing_calls,
        "pass_s2_0": True,
        "pass_s2_1": total_updates > 0 and len(buffer) > 0,
        "pass_s2_2": buffer.intervention_count > 0 and "HUMAN" in sources_seen,
        "pass_s2_3": (
            args.policy_mode in ("plumbing", "real")
            and plumbing_calls > 0
            and len(batch_shapes) > 0
        ),
        "pass_s2_4": args.backend == "send_robot" and sent_joint_ok and buffer.intervention_count > 0,
    }
    report["pass"] = report["pass_s2_0"] and report["pass_s2_1"] and report["pass_s2_2"]
    if args.policy_mode in ("plumbing", "real"):
        report["pass"] = report["pass"] and report["pass_s2_3"]
    if args.backend == "send_robot":
        report["pass"] = report["pass"] and report["pass_s2_4"]
    with open(output_dir / "dryrun_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "episode_rewards": episode_rewards,
        },
        output_dir / "final_checkpoint.pt",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["pass"]:
        logger.info("PASS Stage2 dry-run (modes=%s/%s)", args.backend, args.policy_mode)
        return 0
    logger.error("FAIL dry-run criteria: %s", report)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
