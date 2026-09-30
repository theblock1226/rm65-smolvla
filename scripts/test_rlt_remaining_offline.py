#!/usr/bin/env python3
"""Offline acceptance for RLT work that does not need cloud RM65 SmolVLA.

Runs:
  1) Stage1 local dataset load + mock_vla train loop
  2) Stage2 S2-3 plumbing + S2-4 send_robot dry_run
"""

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
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    # Root JSON object starts at the first '{' of stdout.
    text = proc.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise SystemExit("no JSON report in stdout")
    return json.loads(text[start:])


def main() -> int:
    py = sys.executable
    env_prefix = ["env", "PYTHONPATH=.", "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1"]

    s1_out = ROOT / "checkpoints" / "rlt_stage1_offline_test"
    s1 = run(
        env_prefix
        + [
            py,
            "scripts/train_rlt_stage1_local.py",
            "--mock_vla",
            "--dataset_root",
            "datasets/cleaned/rm65_stacking_blocks",
            "--episodes",
            "0",
            "--steps",
            "8",
            "--batch_size",
            "2",
            "--device",
            "cpu",
            "--log_every",
            "4",
            "--save_every",
            "8",
            "--output_dir",
            str(s1_out.relative_to(ROOT)),
        ]
    )
    assert s1.get("pass"), s1
    assert s1.get("state_dim_data") == 7 and s1.get("action_dim_data") == 7, s1
    assert any("front" in k for k in s1.get("image_keys", [])), s1

    s2_out = ROOT / "checkpoints" / "rlt_stage2_s23_s24_test"
    s2 = run(
        env_prefix
        + [
            py,
            "scripts/train_rlt_stage2_rm65.py",
            "--backend",
            "send_robot",
            "--send_robot_dry_run",
            "--policy_mode",
            "plumbing",
            "--auto_reward",
            "1",
            "--max_episodes",
            "2",
            "--steps_per_episode",
            "10",
            "--warmup_episodes",
            "1",
            "--batch_size",
            "8",
            "--device",
            "cpu",
            "--intervene_steps",
            "2,3,4",
            "--camera_set",
            "stack",
            "--output_dir",
            str(s2_out.relative_to(ROOT)),
        ]
    )
    assert s2.get("pass"), s2
    assert s2.get("pass_s2_3"), s2
    assert s2.get("pass_s2_4"), s2
    assert s2.get("plumbing_calls", 0) > 0, s2
    assert any("front" in k for k in s2.get("batch_image_shapes", {})), s2
    assert any("astra" in k for k in s2.get("batch_image_shapes", {})), s2

    summary = {
        "pass": True,
        "stage1": s1,
        "stage2": {
            "pass": s2["pass"],
            "pass_s2_3": s2["pass_s2_3"],
            "pass_s2_4": s2["pass_s2_4"],
            "sources_seen": s2["sources_seen"],
            "batch_image_shapes": s2["batch_image_shapes"],
            "intervention_transitions": s2["intervention_transitions"],
            "total_updates": s2["total_updates"],
        },
        "blocked_on_cloud_smolvla": [
            "Stage1 real extract_embeddings",
            "Stage2 --policy_mode real",
            "S2-5 on-robot training",
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("PASS RLT offline remaining (Stage1 mock + S2-3/S2-4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
