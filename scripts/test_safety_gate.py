#!/usr/bin/env python3
"""Offline acceptance for SafeActionGate + Mock adapter wiring."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_rate_limit() -> dict:
    from lerobot.common.policies.smolvla_rlt.safety import SafeActionGate, SafetyConfig

    gate = SafeActionGate(
        SafetyConfig(
            require_deadman=False,
            max_delta_joints=2.0,
            max_delta_gripper=0.08,
            speed_scale=0.5,
            watchdog_timeout_s=10.0,
        )
    )
    current = np.zeros(7, dtype=np.float32)
    desired = np.array([100, -100, 50, 0, 0, 0, 1.0], dtype=np.float32)
    cmd = gate.filter(desired, current_state=current, deadman_held=True)
    assert cmd is not None
    # After clip (Δ≤2) then *0.5 → |step| ≤ 1.0 on joints
    assert np.max(np.abs(cmd[:6])) <= 1.0 + 1e-5
    assert abs(cmd[6]) <= 0.04 + 1e-5
    assert gate.last_status.clipped
    return {"pass_rate_limit": True, "cmd0": float(cmd[0])}


def test_deadman_blocks() -> dict:
    from lerobot.common.policies.smolvla_rlt.safety import SafeActionGate, SafetyConfig

    gate = SafeActionGate(SafetyConfig(require_deadman=True, watchdog_timeout_s=10.0))
    current = np.zeros(7, dtype=np.float32)
    desired = np.ones(7, dtype=np.float32)
    assert gate.filter(desired, current_state=current, deadman_held=False) is None
    assert gate.last_status.blocked_reason == "deadman_released"
    cmd = gate.filter(desired, current_state=current, deadman_held=True)
    assert cmd is not None
    return {"pass_deadman": True}


def test_watchdog() -> dict:
    from lerobot.common.policies.smolvla_rlt.safety import SafeActionGate, SafetyConfig

    gate = SafeActionGate(
        SafetyConfig(require_deadman=False, watchdog_timeout_s=0.05, speed_scale=1.0)
    )
    current = np.zeros(7, dtype=np.float32)
    desired = np.full(7, 0.1, dtype=np.float32)
    assert gate.filter(desired, current_state=current, deadman_held=True) is not None
    time.sleep(0.08)
    assert gate.filter(desired, current_state=current, deadman_held=True) is None
    assert gate.last_status.watchdog_tripped
    gate.reset()
    assert gate.filter(desired, current_state=current, deadman_held=True) is not None
    return {"pass_watchdog": True}


def test_mock_adapter() -> dict:
    from lerobot.common.policies.smolvla_rlt.robot_adapter import MockRM65Adapter
    from lerobot.common.policies.smolvla_rlt.safety import SafetyConfig

    # Deadman required + force released → no send
    robot = MockRM65Adapter(
        safety=SafetyConfig(require_deadman=True, watchdog_timeout_s=10.0, speed_scale=1.0),
        seed=1,
    )
    robot.connect()
    robot.set_force_deadman(False)
    robot.get_observation()
    before = len(robot.sent_actions)
    robot.send_action(np.full(7, 5.0, dtype=np.float32))
    assert len(robot.sent_actions) == before
    assert robot.blocked_actions >= 1

    robot.set_force_deadman(True)
    state_before = robot.state.copy()
    robot.send_action(np.full(7, 5.0, dtype=np.float32))
    assert len(robot.sent_actions) == before + 1
    delta = robot.sent_actions[-1] - state_before
    assert float(np.max(np.abs(delta[:6]))) <= 2.0 + 1e-4
    assert float(abs(delta[6])) <= 0.08 + 1e-4
    robot.disconnect()
    return {
        "pass_mock_deadman": True,
        "pass_mock_rate_limit": True,
        "blocked": robot.blocked_actions,
        "sent": len(robot.sent_actions),
        "max_joint_delta": float(np.max(np.abs(delta[:6]))),
    }


def main() -> int:
    report = {}
    for name, fn in [
        ("rate_limit", test_rate_limit),
        ("deadman", test_deadman_blocks),
        ("watchdog", test_watchdog),
        ("mock_adapter", test_mock_adapter),
    ]:
        try:
            report.update(fn())
            report[f"ok_{name}"] = True
        except Exception as exc:  # noqa: BLE001
            report[f"ok_{name}"] = False
            report[f"err_{name}"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(report, indent=2))
            return 1

    report["pass"] = all(
        report.get(k)
        for k in (
            "pass_rate_limit",
            "pass_deadman",
            "pass_watchdog",
            "pass_mock_deadman",
            "pass_mock_rate_limit",
        )
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
