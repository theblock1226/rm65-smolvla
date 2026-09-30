#!/usr/bin/env python3
"""Offline acceptance for waiting-period tools (profile / interp / usb eval / replay)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> dict:
    print("\n>>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")
    text = proc.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise SystemExit("no JSON in stdout")
    return json.loads(text[start:])


def test_interpolator_unit() -> dict:
    from lerobot.common.policies.rtc.action_interpolator import ActionInterpolator
    import torch

    interp = ActionInterpolator(multiplier=2)
    a0 = torch.zeros(7)
    a1 = torch.ones(7)
    interp.add(a0)
    assert interp.get() is not None  # first: no prev → single
    assert interp.needs_new_action()
    interp.add(a1)
    mid = interp.get()
    end = interp.get()
    assert mid is not None and end is not None
    assert torch.allclose(mid, torch.full((7,), 0.5), atol=1e-5)
    assert torch.allclose(end, a1)
    assert interp.emitted_policy_action is True
    return {"pass": True, "multiplier": 2}


def main() -> int:
    py = sys.executable
    env = ["env", "PYTHONPATH=.", "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1"]

    # 1) profiler mock
    p1 = run(
        env
        + [
            py,
            "scripts/profile_inference_loop.py",
            "--policy_mode",
            "mock",
            "--fps",
            "20",
            "--control_steps",
            "80",
            "--fake_chunk_ms",
            "80",
            "--request_when_qsize_le",
            "30",
            "--interpolation_multiplier",
            "1",
            "--output",
            "outputs/profiling/offline_test_report.json",
        ]
    )
    assert p1.get("pass"), p1
    assert p1.get("n_chunks", 0) >= 1, p1

    # 2) interpolator unit + profiler with N=2
    interp = test_interpolator_unit()
    p2 = run(
        env
        + [
            py,
            "scripts/profile_inference_loop.py",
            "--policy_mode",
            "mock",
            "--fps",
            "20",
            "--control_steps",
            "60",
            "--fake_chunk_ms",
            "60",
            "--interpolation_multiplier",
            "2",
            "--output",
            "outputs/profiling/offline_test_interp2.json",
        ]
    )
    assert p2.get("pass") and p2.get("interpolation_multiplier") == 2, p2

    # 3) usb eval dry_run
    p3 = run(
        env
        + [
            py,
            "scripts/usb_eval_protocol.py",
            "--dry_run",
            "--n_sample_trials",
            "3",
            "--policy_tag",
            "offline_test",
            "--output_dir",
            "outputs/usb_eval/offline_test",
        ]
    )
    assert p3.get("pass"), p3
    assert Path(ROOT / "outputs/usb_eval/offline_test/session.json").is_file()
    assert Path(ROOT / "outputs/usb_eval/offline_test/PROTOCOL.md").is_file()

    # 4) open-loop replay
    p4 = run(
        env
        + [
            py,
            "scripts/replay_demo_openloop.py",
            "--dataset_root",
            "datasets/cleaned/rm65_usb_insert",
            "--episode",
            "0",
            "--max_steps",
            "40",
            "--backend",
            "send_robot_dry",
            "--output",
            "outputs/replay/offline_test.json",
        ]
    )
    assert p4.get("pass") and p4.get("action_dim") == 7, p4

    # config field exists
    import dataclasses
    from lerobot.scripts.server import configs as cfgmod

    fields = {f.name for f in dataclasses.fields(cfgmod.RobotClientConfig)}
    assert "interpolation_multiplier" in fields

    summary = {
        "pass": True,
        "profile": {
            "pass": p1["pass"],
            "underruns": p1["underruns"],
            "suggested_rtc": p1.get("suggested_rtc"),
            "chunk_p95_ms": (p1.get("chunk_latency_ms") or {}).get("p95"),
        },
        "interpolator": {
            "unit": interp,
            "profile_n2_pass": p2["pass"],
            "control_fps": p2.get("control_fps"),
        },
        "usb_eval": p3.get("summary"),
        "replay": {
            "frames": p4["frames_replayed"],
            "tracking_mae": (p4.get("tracking") or {}).get("mae_mean"),
        },
        "robot_client_config_has_interpolation": True,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("PASS waiting tools 1-4 offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
