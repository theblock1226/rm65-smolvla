#!/usr/bin/env python3
"""End-to-end offline test: collect mock DAgger → convert → verify LeRobot dataset.

Exit 0 only if all stages pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/sdznkj/zkx_ws/sendlerobot")
BUNDLE = ROOT / "datasets" / "dagger_rm65_mock_e2e"
OUT_ROOT = ROOT / "datasets" / "dagger_lerobot"
REPO = "rm65_dagger_e2e"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)})


def main() -> int:
    import shutil

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    ds = OUT_ROOT / REPO
    if ds.exists():
        shutil.rmtree(ds)

    run(
        [
            sys.executable,
            "scripts/dagger_rm65_collect.py",
            "--backend",
            "mock",
            "--max_episodes",
            "3",
            "--steps_per_episode",
            "10",
            "--intervene_steps",
            "3,4,5",
            "--corrections_only",
            "--fps",
            "100",
            "--output_dir",
            str(BUNDLE),
        ]
    )
    # ensure images exist
    eps = sorted(BUNDLE.glob("episode_*"))
    assert eps, "no episodes"
    assert (eps[0] / "images").is_dir() or True  # may have cameras from mock
    # mock always has cameras in frames → should have images/
    assert any((e / "images").is_dir() for e in eps), "expected images/ in episode bundles"

    run(
        [
            sys.executable,
            "scripts/dagger_rm65_to_lerobot.py",
            "--bundle_dir",
            str(BUNDLE),
            "--out_root",
            str(OUT_ROOT),
            "--repo_id",
            REPO,
            "--fps",
            "20",
        ]
    )

    report = json.loads((OUT_ROOT / REPO / "dagger_convert_report.json").read_text())
    verify = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "scripts/dagger_rm65_to_lerobot.py",
                "--verify_only",
                str(OUT_ROOT / REPO),
            ],
            cwd=str(ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        ).decode()
    )
    summary = {
        "collect_episodes": len(list(BUNDLE.glob("episode_*"))),
        "convert": report,
        "verify": verify,
        "pass": bool(verify.get("ok") and verify.get("intervention_sum", 0) > 0 and report.get("episodes", 0) > 0),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT_ROOT / REPO / "e2e_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
