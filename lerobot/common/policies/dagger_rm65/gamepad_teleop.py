"""Gamepad clutch teleop for RM65 DAgger (hold → human, release → policy).

Uses the same StickIntegrator mapping as ``rm65_movep_teleop`` / record:
  left stick Y/Z, right stick X/ry, hat roll/yaw, LB/RB gripper.

Human actions are returned as 7-D joint targets (IK) so they share the
``rm/sent_actions`` → ``play_action_rm65`` path with the policy.

Intervene binding:
  - On connect, waits a few seconds for you to **hold** the clutch control
    (face button or trigger). That index is saved for the session.
  - Or set ``DAGGER_INTERVENE_BUTTON`` / ``DAGGER_INTERVENE_AXIS`` to skip calibrate.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return list(default)
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or list(default)


class GamepadClutchTeleop:
    """Hold intervene button/trigger → Cartesian stick + IK → joint+gripper action."""

    def __init__(
        self,
        *,
        intervene_button: int | None = None,
        end_episode_button: int | None = None,
        ip: str = "192.168.73.18",
        port: int = 8080,
        sensitivity: float | None = None,
        orient_sensitivity: float | None = None,
        deadzone: float | None = None,
        calibrate_s: float | None = None,
    ):
        # Multiple face-button candidates (GameSir Y is often 3; some modes use 4)
        if intervene_button is not None:
            default_btns = [int(intervene_button)]
        else:
            default_btns = [3, 4]
        self.intervene_buttons = _parse_int_list(
            os.environ.get("DAGGER_INTERVENE_BUTTONS"),
            default_btns,
        )
        # Single-button env overrides list
        if "DAGGER_INTERVENE_BUTTON" in os.environ:
            self.intervene_buttons = [int(os.environ["DAGGER_INTERVENE_BUTTON"])]

        self.end_episode_button = int(
            os.environ.get(
                "DAGGER_END_EPISODE_BUTTON",
                end_episode_button if end_episode_button is not None else 1,
            )
        )
        # Trigger axis: -1 = disabled until calibrated / env set
        axis_env = os.environ.get("DAGGER_INTERVENE_AXIS", "").strip()
        self.intervene_axis: int | None = int(axis_env) if axis_env != "" else None
        self.intervene_axis_threshold = float(os.environ.get("DAGGER_INTERVENE_AXIS_THRESH", "0.55"))
        self.calibrate_s = float(
            os.environ.get(
                "DAGGER_CALIBRATE_S",
                calibrate_s if calibrate_s is not None else 8.0,
            )
        )
        self.skip_calibrate = os.environ.get("DAGGER_SKIP_CALIBRATE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        self.ip = ip
        self.port = port
        self.sensitivity = float(
            os.environ.get("TELEOP_SENSITIVITY", sensitivity if sensitivity is not None else 0.001)
        )
        self.orient_sensitivity = float(
            os.environ.get(
                "TELEOP_ORIENT_SENSITIVITY",
                orient_sensitivity if orient_sensitivity is not None else 0.003,
            )
        )
        self.deadzone = float(os.environ.get("GAMEPAD_DEADZONE", deadzone if deadzone is not None else 0.15))

        self._kin = None
        self._stick = None
        self._gripper = 1.0
        self._was_holding = False
        self._prev_end = False
        self._end_requested = False
        self._connected = False
        self._bind_desc = f"buttons={self.intervene_buttons}"

    def connect(self) -> None:
        from rm65_stick import StickIntegrator, _axis_ids, _env_bool
        from sendrobot.send_robot.rm_ik import RM65Kinematics

        self._kin = RM65Kinematics(ip=self.ip, port=self.port)
        self._kin.connect()
        pose = [float(v) for v in self._kin.get_current_pose()[:6]]
        ay, az, ax, apitch = _axis_ids()
        self._stick = StickIntegrator(
            pose[0],
            pose[1],
            pose[2],
            pose[3],
            pose[4],
            pose[5],
            sensitivity=self.sensitivity,
            orient_sensitivity=self.orient_sensitivity,
            deadzone=self.deadzone,
            bounds={
                "x": (-0.8, 0.8),
                "y": (-0.8, 0.8),
                "z": (0.05, 0.85),
            },
            axis_y=ay,
            axis_z=az,
            axis_x=ax,
            axis_pitch=apitch,
            hat_lock_until_release=_env_bool("GAMEPAD_HAT_SINGLE_AXIS", True),
            hat_block_pitch=_env_bool("GAMEPAD_HAT_BLOCK_PITCH", True),
        )
        self._stick.start()
        self._gripper = 1.0
        self._connected = True

        if not self.skip_calibrate and self.calibrate_s > 0:
            self._calibrate_intervene(self.calibrate_s)
        else:
            self._bind_desc = self._describe_bind()
            logger.info("GamepadClutchTeleop ready %s (skip calibrate)", self._bind_desc)

        logger.info(
            "介入=按住 [%s] | 结束本集=点 b%d | LB/RB=夹爪",
            self._bind_desc,
            self.end_episode_button,
        )

    def _describe_bind(self) -> str:
        parts = [f"b{b}" for b in self.intervene_buttons]
        if self.intervene_axis is not None:
            parts.append(f"axis{self.intervene_axis}>{self.intervene_axis_threshold:.2f}")
        return " or ".join(parts) if parts else "(unbound)"

    def _calibrate_intervene(self, timeout_s: float) -> None:
        """Wait for user to hold a button or pull a trigger; bind that control."""
        assert self._stick is not None and self._stick.joystick is not None
        import pygame

        js = self._stick.joystick
        reserved = {
            self._stick.btn_gripper_open,
            self._stick.btn_gripper_close,
            self.end_episode_button,
            self._stick.btn_rec_start,
            self._stick.btn_rec_stop,
            self._stick.btn_rec_rerecord,
        }
        pygame.event.pump()
        base_axes = [float(js.get_axis(i)) for i in range(js.get_numaxes())]
        logger.info(
            ">>> 标定介入键：请在 %.0f 秒内【按住】Y 或【扣住】LT/RT（不要按 LB/RB/A/B）…",
            timeout_s,
        )
        print(
            f"\n*** 请按住介入键/扳机 ({timeout_s:.0f}s) — GameSir Y 或 LT ***\n",
            flush=True,
        )
        t0 = time.time()
        bound = False
        while time.time() - t0 < timeout_s:
            pygame.event.pump()
            for i in range(js.get_numbuttons()):
                if i in reserved:
                    continue
                if js.get_button(i):
                    self.intervene_buttons = [i]
                    self.intervene_axis = None
                    self._bind_desc = f"b{i}"
                    logger.info("已绑定介入键 button[%d] name-hint=%s", i, js.get_name())
                    print(f"*** 已绑定 button[{i}] — 松开后开始采集 ***\n", flush=True)
                    bound = True
                    # wait for release
                    while js.get_button(i):
                        pygame.event.pump()
                        time.sleep(0.05)
                    break
            if bound:
                break
            for i in range(js.get_numaxes()):
                v = float(js.get_axis(i))
                if abs(v - base_axes[i]) >= self.intervene_axis_threshold:
                    self.intervene_axis = i
                    # Keep buttons as secondary fallback
                    self._bind_desc = self._describe_bind()
                    logger.info(
                        "已绑定介入轴 axis[%d] (Δ=%.2f, thresh=%.2f)",
                        i,
                        v - base_axes[i],
                        self.intervene_axis_threshold,
                    )
                    print(f"*** 已绑定 axis[{i}] — 松开后开始采集 ***\n", flush=True)
                    bound = True
                    while abs(float(js.get_axis(i)) - base_axes[i]) >= self.intervene_axis_threshold * 0.5:
                        pygame.event.pump()
                        time.sleep(0.05)
                    break
            if bound:
                break
            time.sleep(0.04)

        if not bound:
            self._bind_desc = self._describe_bind()
            logger.warning(
                "标定超时，回退 %s — 若仍无法介入请设 DAGGER_INTERVENE_BUTTON=实际编号",
                self._bind_desc,
            )
            print(
                f"*** 未检测到按键，回退 {self._bind_desc} ***\n",
                flush=True,
            )

    def disconnect(self) -> None:
        if self._stick is not None:
            try:
                self._stick.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("stick.cleanup failed: %s", exc)
            self._stick = None
        if self._kin is not None:
            try:
                self._kin.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("kin.disconnect failed: %s", exc)
            self._kin = None
        self._connected = False

    def _reseed_from_arm(self) -> None:
        assert self._kin is not None and self._stick is not None
        pose = [float(v) for v in self._kin.get_current_pose()[:6]]
        self._stick.x, self._stick.y, self._stick.z = pose[0], pose[1], pose[2]
        self._stick.rx, self._stick.ry, self._stick.rz = pose[3], pose[4], pose[5]
        self._stick._R = Rotation.from_euler("xyz", [self._stick.rx, self._stick.ry, self._stick.rz])
        logger.info(
            "  clutch reseed xyz=(%.3f,%.3f,%.3f) rpy=(%.2f,%.2f,%.2f)",
            pose[0],
            pose[1],
            pose[2],
            pose[3],
            pose[4],
            pose[5],
        )

    def _holding_now(self) -> bool:
        assert self._stick is not None and self._stick.joystick is not None
        import pygame

        pygame.event.pump()
        js = self._stick.joystick
        n = js.get_numbuttons()
        for b in self.intervene_buttons:
            if b < n and bool(js.get_button(b)):
                return True
        if self.intervene_axis is not None and self.intervene_axis < js.get_numaxes():
            # Triggers often rest at -1 or 0; treat |value| large or value>thresh
            v = float(js.get_axis(self.intervene_axis))
            if v >= self.intervene_axis_threshold or v <= -self.intervene_axis_threshold:
                return True
        return False

    def poll(self) -> bool:
        """Hold-to-intervene (same contract as HoldButtonIntervention.poll)."""
        if not self._connected or self._stick is None or self._stick.joystick is None:
            return False
        import pygame

        pygame.event.pump()
        js = self._stick.joystick
        n = js.get_numbuttons()
        holding = self._holding_now()

        if holding and not self._was_holding:
            try:
                self._reseed_from_arm()
            except Exception as exc:  # noqa: BLE001
                logger.warning("clutch reseed failed: %s", exc)
            logger.info("  >> INTERVENTION ON (%s) — human takeover", self._bind_desc)
        if not holding and self._was_holding:
            logger.info("  >> INTERVENTION OFF (release) — back to policy")
        self._was_holding = holding

        end_down = self.end_episode_button < n and bool(js.get_button(self.end_episode_button))
        if end_down and not self._prev_end:
            self._end_requested = True
            logger.info("  >> END EPISODE requested (button %d)", self.end_episode_button)
        self._prev_end = end_down

        return holding

    def consume_end_episode(self) -> bool:
        if self._end_requested:
            self._end_requested = False
            return True
        return False

    def human_action(self, current_state: np.ndarray | None = None) -> np.ndarray:
        """Return 7-D joint+gripper target from stick + IK."""
        if not self._connected or self._stick is None or self._kin is None:
            raise RuntimeError("GamepadClutchTeleop not connected")

        x, y, z, rx, ry, rz = self._stick.update()
        if self._stick.gripper_event == "open":
            self._gripper = 1.0
            logger.info("  gripper → open")
        elif self._stick.gripper_event == "close":
            self._gripper = 0.0
            logger.info("  gripper → close")
        if current_state is not None:
            ref = [float(v) for v in np.asarray(current_state, dtype=np.float32).reshape(-1)[:6]]
        else:
            ref = self._kin.get_current_joints_deg()[:6]

        q = self._kin.try_inverse_kinematics([x, y, z, rx, ry, rz], ref)
        if q is None:
            q = self._kin.try_inverse_kinematics_quat(
                [x, y, z], self._stick.as_quat(), ref
            )
        if q is None:
            q = ref
        out = np.asarray([*q, float(self._gripper)], dtype=np.float32)
        return out

    def reset(self) -> None:
        self._was_holding = False
        self._prev_end = False
        self._end_requested = False
