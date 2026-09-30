#!/usr/bin/env python3
"""USB insert evaluation protocol + trial logger.

Defines a fixed rubric for comparing SFT / RTC / RECAP / interventions on the
``rm65_usb_insert`` task. Does **not** require a policy — you run trials on the
robot (or dry-run) and record outcomes.

Dry-run (writes a sample session)::

    PYTHONPATH=. python3 scripts/usb_eval_protocol.py --dry_run \\
      --output_dir outputs/usb_eval/sample_session

Interactive (operator at the robot)::

    PYTHONPATH=. python3 scripts/usb_eval_protocol.py \\
      --operator alice --output_dir outputs/usb_eval/$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "usb_eval_v1"

# Fixed setup checklist — keep identical across SFT / RTC / RECAP comparisons
SETUP_CHECKLIST = [
    "Arm at agreed home / ready pose (same joint targets every session)",
    "USB plug + port fixtures fixed; cable slack consistent",
    "Cameras: front + mono streaming; mono lag acknowledged (+10 frames in cleaned data)",
    "Lighting / background unchanged vs recording days if possible",
    "Control fps = 20 (or document deviation); Kalman on SendRobot as usual",
    "Policy ckpt / RTC / interpolator / RECAP CFG noted in session meta",
]

SUCCESS_CRITERIA = {
    "success": "Plug fully seated and released without external help; stays seated 1s",
    "partial": "Reached port / partial insert but did not seat, or needed light touch assist",
    "fail": "Missed, jammed, timed out, or required full human takeover",
    "timeout_s": 60.0,
}

OUTCOMES = ("success", "partial", "fail", "abort")


@dataclass
class TrialRecord:
    trial_index: int
    outcome: str
    elapsed_s: float
    intervention: bool = False
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionMeta:
    protocol_version: str = PROTOCOL_VERSION
    task: str = "usb_insert"
    dataset_ref: str = "datasets/cleaned/rm65_usb_insert"
    operator: str = ""
    policy_path: str = ""
    policy_tag: str = ""  # e.g. sft / rtc / recap / dagger
    rtc_enabled: bool = False
    interpolation_multiplier: int = 1
    control_fps: float = 20.0
    timeout_s: float = 60.0
    setup_checklist: list[str] = field(default_factory=lambda: list(SETUP_CHECKLIST))
    success_criteria: dict[str, Any] = field(default_factory=lambda: dict(SUCCESS_CRITERIA))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def summarize(trials: list[TrialRecord]) -> dict[str, Any]:
    n = len(trials)
    if n == 0:
        return {"n_trials": 0, "success_rate": None}
    counts = {k: sum(1 for t in trials if t.outcome == k) for k in OUTCOMES}
    succ = counts["success"]
    return {
        "n_trials": n,
        "counts": counts,
        "success_rate": succ / n,
        "partial_rate": counts["partial"] / n,
        "fail_rate": counts["fail"] / n,
        "intervention_rate": sum(1 for t in trials if t.intervention) / n,
        "mean_elapsed_s": sum(t.elapsed_s for t in trials) / n,
    }


def write_session(out_dir: Path, meta: SessionMeta, trials: list[TrialRecord]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(trials)
    payload = {
        "meta": asdict(meta),
        "trials": [asdict(t) for t in trials],
        "summary": summary,
    }
    (out_dir / "session.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["trial_index", "outcome", "elapsed_s", "intervention", "notes", "timestamp"],
        )
        w.writeheader()
        for t in trials:
            w.writerow(asdict(t))
    (out_dir / "PROTOCOL.md").write_text(
        "\n".join(
            [
                f"# USB eval protocol ({PROTOCOL_VERSION})",
                "",
                "## Setup checklist",
                *[f"- [ ] {x}" for x in SETUP_CHECKLIST],
                "",
                "## Success criteria",
                *[f"- **{k}**: {v}" for k, v in SUCCESS_CRITERIA.items()],
                "",
                "## How to run a trial",
                "1. Reset to home pose",
                "2. Start policy / open-loop / teleop as tagged in session meta",
                "3. Start timer; stop on success / fail / timeout / abort",
                "4. Record outcome + whether any hold-to-intervene happened",
                "",
                "## Compare fair",
                "Same home pose, same fixtures, same timeout, same operator when possible.",
                "Change only the policy tag (sft / rtc / recap / …).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="USB insert eval protocol logger")
    p.add_argument("--output_dir", type=str, default="outputs/usb_eval/session")
    p.add_argument("--operator", type=str, default="")
    p.add_argument("--policy_path", type=str, default="")
    p.add_argument("--policy_tag", type=str, default="unspecified")
    p.add_argument("--rtc_enabled", action="store_true")
    p.add_argument("--interpolation_multiplier", type=int, default=1)
    p.add_argument("--control_fps", type=float, default=20.0)
    p.add_argument("--timeout_s", type=float, default=60.0)
    p.add_argument("--dry_run", action="store_true", help="Write a sample session without stdin")
    p.add_argument("--n_sample_trials", type=int, default=3)
    return p.parse_args()


def interactive_trials(timeout_s: float) -> list[TrialRecord]:
    trials: list[TrialRecord] = []
    print("USB eval — commands: s=success p=partial f=fail a=abort q=quit session")
    print(f"Timeout guideline: {timeout_s}s (you still enter the outcome manually)")
    idx = 1
    while True:
        input(f"\n--- Trial {idx}: reset to home, then Enter to START timer --- ")
        t0 = time.perf_counter()
        raw = input("outcome [s/p/f/a/q] (+ optional 'i' if intervention, e.g. si): ").strip().lower()
        elapsed = time.perf_counter() - t0
        if not raw or raw[0] == "q":
            break
        intervention = "i" in raw
        key = raw[0]
        outcome = {"s": "success", "p": "partial", "f": "fail", "a": "abort"}.get(key)
        if outcome is None:
            print("  invalid; try again")
            continue
        notes = input("notes (optional): ").strip()
        trials.append(
            TrialRecord(
                trial_index=idx,
                outcome=outcome,
                elapsed_s=round(elapsed, 3),
                intervention=intervention,
                notes=notes,
            )
        )
        idx += 1
    return trials


def sample_trials(n: int) -> list[TrialRecord]:
    # Deterministic sample for offline acceptance of the logger itself
    outcomes = ["success", "partial", "fail"]
    return [
        TrialRecord(
            trial_index=i + 1,
            outcome=outcomes[i % 3],
            elapsed_s=12.0 + i,
            intervention=(i % 3 == 2),
            notes="dry_run sample",
        )
        for i in range(n)
    ]


def main() -> int:
    args = parse_args()
    meta = SessionMeta(
        operator=args.operator,
        policy_path=args.policy_path,
        policy_tag=args.policy_tag,
        rtc_enabled=args.rtc_enabled,
        interpolation_multiplier=args.interpolation_multiplier,
        control_fps=args.control_fps,
        timeout_s=args.timeout_s,
    )
    trials = sample_trials(args.n_sample_trials) if args.dry_run else interactive_trials(args.timeout_s)
    payload = write_session(Path(args.output_dir), meta, trials)
    print(json.dumps({"pass": True, "summary": payload["summary"], "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
