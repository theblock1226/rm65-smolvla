"""RM65 robot adapters for RLT Stage 2.

S2-0: abstract interface + Mock backend (no real robot).
S2-2: hold-button intervention (mock schedule or pygame gamepad).
Real SendRobot backend can be added later (S2-4) behind the same interface.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from lerobot.common.policies.smolvla_rlt.safety import SafeActionGate, SafetyConfig

logger = logging.getLogger(__name__)

# Matches datasets/cleaned/*/meta/clean_report.json
RM65_JOINT_KEYS = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
]

# RM65 cameras (cleaned datasets + Stage2 adapters)
# - USB insert: front + mono  (mono is USB-only)
# - desk / slot / stack: front + astra  (no mono)
CAMERA_SETS = {
    "desk": ["front", "astra"],
    "slot": ["front", "astra"],
    "stack": ["front", "astra"],
    # Eval / SendRobot keys used by astra_smolvla + rm65_run_astra_eval
    "eval": ["camera1", "camera2"],
    "stack_eval": ["camera1", "camera2"],
    "usb": ["front", "mono"],
    "default": ["front", "astra"],
}

# Alternate names accepted when reading SendRobot / dataset observations
CAMERA_KEY_ALIASES = {
    "front": ("front", "camera1"),
    "astra": ("astra", "camera2"),
    "mono": ("mono", "camera2"),
    "camera1": ("camera1", "front"),
    "camera2": ("camera2", "astra", "mono"),
    "camera3": ("camera3",),
}

# Map cleaned dataset folder / policy family → camera_set key
TASK_CAMERA_SET = {
    "usb": "usb",
    "usb_insert": "usb",
    "rm65_usb_insert": "usb",
    "desk": "desk",
    "cleanup_the_desk": "desk",
    "rm65_cleanup_the_desk": "desk",
    "slot": "slot",
    "place_to_slot": "slot",
    "rm65_place_to_slot": "slot",
    "stack": "stack",
    "stacking_blocks": "stack",
    "rm65_stacking_blocks": "stack",
    "astra": "stack",
    "astra_multitask": "stack",
}

# Which frozen SmolVLA family to use (local dirs after HF download)
TASK_SMOLVLA_FAMILY = {
    "usb": "usb",
    "usb_insert": "usb",
    "rm65_usb_insert": "usb",
    "desk": "astra",
    "cleanup_the_desk": "astra",
    "rm65_cleanup_the_desk": "astra",
    "slot": "astra",
    "place_to_slot": "astra",
    "rm65_place_to_slot": "astra",
    "stack": "astra",
    "stacking_blocks": "astra",
    "rm65_stacking_blocks": "astra",
    "astra": "astra",
    "astra_multitask": "astra",
}


def resolve_camera_set(task_or_set: str) -> str:
    """Return a CAMERA_SETS key for a task name or pass-through set name."""
    key = task_or_set.strip()
    if key in CAMERA_SETS:
        return key
    return TASK_CAMERA_SET.get(key, TASK_CAMERA_SET.get(key.replace("-", "_"), "default"))


def resolve_smolvla_family(task_or_family: str) -> str:
    """Return ``usb`` or ``astra`` frozen-VLA family for a task name."""
    key = task_or_family.strip()
    if key in ("usb", "astra"):
        return key
    return TASK_SMOLVLA_FAMILY.get(key, TASK_SMOLVLA_FAMILY.get(key.replace("-", "_"), "astra"))


def cameras_for_task(task_or_set: str) -> list[str]:
    return list(CAMERA_SETS[resolve_camera_set(task_or_set)])


def joint_dict_to_vector(obs: dict, dim: int = 7) -> np.ndarray:
    vals = [float(obs.get(k, 0.0)) for k in RM65_JOINT_KEYS[:dim]]
    return np.asarray(vals, dtype=np.float32)


def vector_to_joint_dict(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    return {RM65_JOINT_KEYS[i]: float(action[i]) for i in range(min(len(RM65_JOINT_KEYS), action.shape[0]))}


@dataclass
class RobotObservation:
    """Normalized observation for the Stage-2 loop."""

    state: np.ndarray  # (7,)
    images: dict[str, np.ndarray]  # name -> (C,H,W) float32 in [0,1] or uint8
    raw: dict = field(default_factory=dict)


class RobotAdapter(ABC):
    """Hardware / mock interface used by train_rlt_stage2_rm65.py."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_observation(self) -> RobotObservation: ...

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """Send one 7-D joint+gripper command."""

    @abstractmethod
    def poll_intervention(self) -> bool:
        """True while human hold-to-intervene is active."""

    @abstractmethod
    def human_action(self) -> np.ndarray:
        """Current human joint command (7,) while intervening."""


class HoldButtonIntervention:
    """Hold-to-intervene: pressed = human takeover, released = policy.

    Mock mode: ``force`` can be set by the training loop for scripted tests.
    Gamepad mode: polls pygame joystick button index (default avoids LB/RB gripper).
    """

    def __init__(self, button_index: int = 6, backend: str = "mock"):
        self.button_index = int(os.environ.get("RLT_INTERVENE_BUTTON", button_index))
        self.backend = backend
        self._force: bool | None = None
        self._joystick = None
        self._was = False

    def set_force(self, value: bool | None) -> None:
        """Override for mock/scripted intervention (None = use hardware/backend)."""
        self._force = value

    def _ensure_gamepad(self) -> None:
        if self._joystick is not None:
            return
        import pygame

        if not pygame.get_init():
            pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No gamepad found for hold-to-intervene")
        js = pygame.joystick.Joystick(0)
        js.init()
        self._joystick = js
        logger.info(
            "Intervene button index=%d on %s (%d buttons)",
            self.button_index,
            js.get_name(),
            js.get_numbuttons(),
        )

    def poll(self) -> bool:
        if self._force is not None:
            active = bool(self._force)
        elif self.backend == "mock":
            active = False
        else:
            import pygame

            self._ensure_gamepad()
            pygame.event.pump()
            n = self._joystick.get_numbuttons()
            active = self.button_index < n and bool(self._joystick.get_button(self.button_index))

        if active and not self._was:
            logger.info("  >> INTERVENTION ON (hold) — human takeover")
        if not active and self._was:
            logger.info("  >> INTERVENTION OFF (release) — back to policy")
        self._was = active
        return active

    def reset(self) -> None:
        self._was = False
        self._force = None


class DeadmanHold:
    """Hold-to-arm: pressed = motion allowed; released = SafeActionGate blocks.

    Separate from intervene (human takeover). Default button index 5
    (``RLT_DEADMAN_BUTTON``), leaving intervene on 6.
    """

    def __init__(self, button_index: int = 5, backend: str = "mock"):
        self.button_index = int(os.environ.get("RLT_DEADMAN_BUTTON", button_index))
        self.backend = backend
        self._force: bool | None = None
        self._joystick = None
        self._was = False

    def set_force(self, value: bool | None) -> None:
        self._force = value

    def _ensure_gamepad(self) -> None:
        if self._joystick is not None:
            return
        import pygame

        if not pygame.get_init():
            pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No gamepad found for deadman")
        js = pygame.joystick.Joystick(0)
        js.init()
        self._joystick = js
        logger.info(
            "Deadman button index=%d on %s (%d buttons)",
            self.button_index,
            js.get_name(),
            js.get_numbuttons(),
        )

    def poll(self) -> bool:
        if self._force is not None:
            held = bool(self._force)
        elif self.backend == "mock":
            held = False
        else:
            import pygame

            self._ensure_gamepad()
            pygame.event.pump()
            n = self._joystick.get_numbuttons()
            held = self.button_index < n and bool(self._joystick.get_button(self.button_index))

        if held and not self._was:
            logger.info("  >> DEADMAN ON — motion armed")
        if not held and self._was:
            logger.info("  >> DEADMAN OFF — motion blocked")
        self._was = held
        return held

    def reset(self) -> None:
        self._was = False
        self._force = None


def _default_safety_config(
    *,
    safety: SafetyConfig | None = None,
    require_deadman: bool | None = None,
    speed_scale: float | None = None,
) -> SafetyConfig:
    if safety is not None:
        return safety
    cfg = SafetyConfig()
    if require_deadman is not None:
        cfg.require_deadman = require_deadman
    if speed_scale is not None:
        cfg.speed_scale = speed_scale
    return cfg


class MockRM65Adapter(RobotAdapter):
    """In-process fake arm for S2-0..S2-2 dry-run (no ROS / no pygame required)."""

    def __init__(
        self,
        *,
        camera_set: str = "stack",
        image_size: tuple[int, int] = (480, 640),
        intervene_schedule: set[int] | None = None,
        seed: int = 0,
        safety: SafetyConfig | None = None,
        # Offline mock: rate-limit on, deadman off (no gamepad)
        require_deadman: bool = False,
        speed_scale: float | None = None,
    ):
        self.camera_names = CAMERA_SETS.get(camera_set, CAMERA_SETS["default"])
        self.image_size = image_size  # (H, W)
        self.intervene_schedule = intervene_schedule or set()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(7, dtype=np.float32)
        self._step = 0
        self._connected = False
        self.intervention = HoldButtonIntervention(backend="mock")
        self.deadman = DeadmanHold(backend="mock")
        self.safety = SafeActionGate(
            _default_safety_config(
                safety=safety, require_deadman=require_deadman, speed_scale=speed_scale
            )
        )
        self.sent_actions: list[np.ndarray] = []
        self.blocked_actions: int = 0

    def connect(self) -> None:
        self._connected = True
        self._step = 0
        self.sent_actions.clear()
        self.blocked_actions = 0
        self.intervention.reset()
        self.deadman.reset()
        self.safety.reset()
        logger.info(
            "MockRM65Adapter connected cameras=%s safety=%s",
            self.camera_names,
            self.safety.stats()["config"],
        )

    def reset_episode(self) -> None:
        self._step = 0
        self.intervention.reset()
        self.deadman.reset()
        self.safety.reset()

    def disconnect(self) -> None:
        self._connected = False
        logger.info(
            "MockRM65Adapter disconnected (sent %d blocked %d) safety=%s",
            len(self.sent_actions),
            self.blocked_actions,
            self.safety.stats(),
        )

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("Mock adapter not connected")
        # Scripted hold windows: step indices in intervene_schedule
        self.intervention.set_force(self._step in self.intervene_schedule)
        h, w = self.image_size
        images = {
            name: self.rng.random((3, h, w), dtype=np.float32) for name in self.camera_names
        }
        # Slow drift so consecutive states differ
        self.state = self.state + self.rng.normal(0, 0.01, size=7).astype(np.float32)
        return RobotObservation(state=self.state.copy(), images=images, raw={"step": self._step})

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == 7, f"expected 7-D action, got {action.shape}"
        cmd = self.safety.filter(
            action, current_state=self.state, deadman_held=self.poll_deadman()
        )
        if cmd is None:
            self.blocked_actions += 1
            return
        self.state = 0.8 * self.state + 0.2 * cmd  # pretend low-pass tracking
        self.sent_actions.append(cmd.copy())
        self._step += 1

    def poll_intervention(self) -> bool:
        return self.intervention.poll()

    def poll_deadman(self) -> bool:
        return self.deadman.poll()

    def set_force_deadman(self, value: bool | None) -> None:
        self.deadman.set_force(value)

    def human_action(self) -> np.ndarray:
        # Human "correction": nudge state toward a fixed pose
        target = np.array([10, -20, 30, 0, 15, 0, 0.5], dtype=np.float32)
        return (0.7 * self.state + 0.3 * target).astype(np.float32)


class GamepadHoldMixin:
    """Shared helpers for SendRobot+gamepad adapter (S2-4)."""

    @staticmethod
    def make_hold_intervention(button_index: int = 6) -> HoldButtonIntervention:
        return HoldButtonIntervention(button_index=button_index, backend="gamepad")


class SendRobotRM65Adapter(RobotAdapter):
    """RM65 via SendRobot + hold-to-intervene gamepad (S2-4).

    ``dry_run=True`` (default for offline tests): no ROS / no pygame required;
    uses the same observation/action contract as the real path (joint dicts,
    camera keys) so Stage2 can accept ``--backend send_robot`` before hardware
    is up. Pass ``dry_run=False`` and a connected ``SendRobot`` instance for
    real robot use.

    Software safety (deadman + Δq limit + speed_scale + watchdog) sits in
    ``send_action`` via ``SafeActionGate``. Does **not** replace hardware E-Stop.
    """

    def __init__(
        self,
        *,
        camera_set: str = "stack",
        image_size: tuple[int, int] = (480, 640),
        intervene_button: int = 6,
        deadman_button: int = 5,
        dry_run: bool = True,
        robot=None,
        human_action_provider=None,
        seed: int = 0,
        safety: SafetyConfig | None = None,
        require_deadman: bool | None = None,
        speed_scale: float | None = None,
    ):
        self.camera_names = CAMERA_SETS.get(camera_set, CAMERA_SETS["default"])
        self.image_size = image_size
        self.dry_run = dry_run
        self.robot = robot
        self.human_action_provider = human_action_provider
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(7, dtype=np.float32)
        self._connected = False
        self._step = 0
        self.sent_actions: list[np.ndarray] = []
        self.sent_joint_dicts: list[dict[str, float]] = []
        self.blocked_actions: int = 0
        backend = "mock" if dry_run else "gamepad"
        self.intervention = HoldButtonIntervention(
            button_index=intervene_button, backend=backend
        )
        self.deadman = DeadmanHold(button_index=deadman_button, backend=backend)
        # Real robot: deadman required by default; dry_run: off (no pad)
        if require_deadman is None:
            require_deadman = not dry_run
        self.safety = SafeActionGate(
            _default_safety_config(
                safety=safety, require_deadman=require_deadman, speed_scale=speed_scale
            )
        )

    def connect(self) -> None:
        if not self.dry_run:
            if self.robot is None:
                raise RuntimeError("SendRobotRM65Adapter(dry_run=False) requires robot=")
            if hasattr(self.robot, "connect") and not getattr(self.robot, "is_connected", True):
                self.robot.connect()
        self._connected = True
        self._step = 0
        self.sent_actions.clear()
        self.sent_joint_dicts.clear()
        self.blocked_actions = 0
        self.intervention.reset()
        self.deadman.reset()
        self.safety.reset()
        logger.info(
            "SendRobotRM65Adapter connected dry_run=%s cameras=%s safety=%s",
            self.dry_run,
            self.camera_names,
            self.safety.stats()["config"],
        )

    def reset_episode(self) -> None:
        self._step = 0
        self.intervention.reset()
        self.deadman.reset()
        self.safety.reset()

    def disconnect(self) -> None:
        if not self.dry_run and self.robot is not None and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception as exc:  # noqa: BLE001 — best-effort shutdown
                logger.warning("robot.disconnect failed: %s", exc)
        self._connected = False
        logger.info(
            "SendRobotRM65Adapter disconnected (sent %d blocked %d) safety=%s",
            len(self.sent_actions),
            self.blocked_actions,
            self.safety.stats(),
        )

    def _obs_from_robot_dict(self, raw: dict) -> RobotObservation:
        state = joint_dict_to_vector(raw, dim=7)
        images: dict[str, np.ndarray] = {}
        h, w = self.image_size
        for name in self.camera_names:
            img = None
            for alias in CAMERA_KEY_ALIASES.get(name, (name,)):
                if alias in raw and raw[alias] is not None:
                    img = np.asarray(raw[alias])
                    break
            if img is not None:
                images[name] = img
            else:
                # Missing camera → zeros (VLA empty_cameras / plumbing still runs)
                images[name] = np.zeros((3, h, w), dtype=np.float32)
        return RobotObservation(state=state, images=images, raw=raw)

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("SendRobot adapter not connected")
        if self.dry_run:
            h, w = self.image_size
            images = {
                name: self.rng.random((3, h, w), dtype=np.float32) for name in self.camera_names
            }
            self.state = self.state + self.rng.normal(0, 0.01, size=7).astype(np.float32)
            raw = {**vector_to_joint_dict(self.state), "step": self._step, "dry_run": True}
            return RobotObservation(state=self.state.copy(), images=images, raw=raw)

        assert self.robot is not None
        raw = self.robot.get_observation()
        if not raw:
            raise RuntimeError("SendRobot returned empty observation")
        obs = self._obs_from_robot_dict(raw)
        self.state = obs.state.copy()
        return obs

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"expected 7-D action, got {action.shape}")
        cmd = self.safety.filter(
            action, current_state=self.state, deadman_held=self.poll_deadman()
        )
        if cmd is None:
            self.blocked_actions += 1
            return
        joint = vector_to_joint_dict(cmd)
        self.sent_actions.append(cmd.copy())
        self.sent_joint_dicts.append(joint)
        if self.dry_run:
            self.state = 0.8 * self.state + 0.2 * cmd
            self._step += 1
            return
        assert self.robot is not None
        self.robot.send_action(joint)
        self._step += 1

    def poll_intervention(self) -> bool:
        return self.intervention.poll()

    def poll_deadman(self) -> bool:
        return self.deadman.poll()

    def set_force_deadman(self, value: bool | None) -> None:
        self.deadman.set_force(value)

    def human_action(self) -> np.ndarray:
        if self.human_action_provider is not None:
            return np.asarray(self.human_action_provider(), dtype=np.float32).reshape(7)
        # Default: mild correction toward a safe-ish pose (real teleop should override)
        target = np.array([10, -20, 30, 0, 15, 0, 0.5], dtype=np.float32)
        return (0.7 * self.state + 0.3 * target).astype(np.float32)

    def set_force_intervention(self, value: bool | None) -> None:
        """Test hook: script hold windows without a physical gamepad."""
        self.intervention.set_force(value)
