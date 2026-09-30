#!/usr/bin/env python3
"""Open-loop replay of cleaned demo joint trajectories (no policy).

Loads actions from a local LeRobot episode and either:
  - dry-run: compute tracking stats vs recorded states (no robot)
  - send_robot dry adapter: exercise joint-dict conversion at dataset fps
  - (optional later) real SendRobot when ROS is up

Dry-run::

    PYTHONPATH=. python3 scripts/replay_demo_openloop.py \\
      --dataset_root datasets/cleaned/rm65_usb_insert \\
      --episode 0 --dry_run

Exclude known-bad USB episode 4 by default for this dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# From datasets/cleaned/README.md
DEFAULT_SKIP = {"datasets/cleaned/rm65_usb_insert": {4}}


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop demo replay")
    p.add_argument("--dataset_root", type=str, default="datasets/cleaned/rm65_usb_insert")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=0.0, help="0 = use dataset fps")
    p.add_argument("--max_steps", type=int, default=0, help="0 = full episode")
    p.add_argument("--dry_run", action="store_true", default=True)
    p.add_argument("--no_dry_run", action="store_false", dest="dry_run")
    p.add_argument(
        "--backend",
        choices=["none", "send_robot_dry"],
        default="none",
        help="none=stats only; send_robot_dry=SendRobotRM65Adapter dry_run",
    )
    p.add_argument("--output", type=str, default="outputs/replay/openloop_report.json")
    p.add_argument("--allow_skipped", action="store_true", help="Allow episodes marked failed/skip")
    return p.parse_args()


def load_episode_arrays(dataset_root: Path, episode: int):
    from lerobot.common.policies.smolvla_rlt.local_dataset import load_local_lerobot_dataset

    ds = load_local_lerobot_dataset(dataset_root, episodes=[episode], download_videos=False)
    actions = []
    states = []
    timestamps = []
    for i in range(len(ds)):
        row = ds[i]
        actions.append(np.asarray(row["action"], dtype=np.float32))
        states.append(np.asarray(row["observation.state"], dtype=np.float32))
        timestamps.append(float(row["timestamp"]))
    return {
        "actions": np.stack(actions, axis=0),
        "states": np.stack(states, axis=0),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "n": len(actions),
        "fps": float(ds.meta.fps),
        "task": ds[0].get("task", ""),
    }


def tracking_stats(actions: np.ndarray, states: np.ndarray) -> dict:
    """Compare action[t] to state[t+1] as a crude open-loop tracking proxy."""
    if len(actions) < 2:
        return {}
    # Next-state vs commanded action (same units in cleaned data)
    err = states[1:] - actions[:-1]
    abs_err = np.abs(err)
    return {
        "mae_per_dim": abs_err.mean(axis=0).tolist(),
        "mae_mean": float(abs_err.mean()),
        "mae_p95": float(np.quantile(abs_err, 0.95)),
        "max_abs": float(abs_err.max()),
        "action_jerk_mae": float(np.abs(np.diff(actions, axis=0)).mean()),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root)
    skip = DEFAULT_SKIP.get(str(root), set()) | DEFAULT_SKIP.get(args.dataset_root, set())
    # Also match resolved relative path key
    for k, v in DEFAULT_SKIP.items():
        if Path(k).name == root.name:
            skip |= v

    if args.episode in skip and not args.allow_skipped:
        raise SystemExit(
            f"episode {args.episode} is in skip list for {root} (USB ep4=failed). "
            "Pass --allow_skipped to override."
        )

    data = load_episode_arrays(root, args.episode)
    fps = args.fps if args.fps > 0 else data["fps"]
    n = data["n"] if args.max_steps <= 0 else min(args.max_steps, data["n"])
    actions = data["actions"][:n]
    states = data["states"][:n]
    dt = 1.0 / fps

    logger.info(
        "episode=%d frames=%d/%d fps=%.1f task=%r",
        args.episode,
        n,
        data["n"],
        fps,
        data["task"][:80] if isinstance(data["task"], str) else data["task"],
    )

    robot = None
    if args.backend == "send_robot_dry":
        from lerobot.common.policies.smolvla_rlt.robot_adapter import SendRobotRM65Adapter

        robot = SendRobotRM65Adapter(camera_set="usb", dry_run=True)
        robot.connect()

    sent = []
    t_wall0 = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        a = actions[i]
        sent.append(a.copy())
        if robot is not None:
            robot.send_action(a)
        if not args.dry_run and robot is None:
            raise SystemExit("non-dry_run requires --backend send_robot_dry (or future real backend)")
        sleep = dt - (time.perf_counter() - t0)
        if sleep > 0 and args.backend != "none":
            time.sleep(sleep)

    wall = time.perf_counter() - t_wall0
    if robot is not None:
        robot.disconnect()

    stats = tracking_stats(actions, states)
    report = {
        "dataset_root": str(root.resolve()),
        "episode": args.episode,
        "frames_replayed": n,
        "fps": fps,
        "backend": args.backend,
        "dry_run": args.dry_run,
        "wall_time_s": wall,
        "ideal_time_s": n * dt,
        "tracking": stats,
        "action_dim": int(actions.shape[1]),
        "state_dim": int(states.shape[1]),
        "pass": actions.shape[1] == 7 and n > 0 and stats.get("mae_mean", 0) >= 0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
