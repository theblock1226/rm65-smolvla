#!/usr/bin/env python3
"""Convert DAgger-lite episode bundles → local LeRobot v2.1 dataset.

Input layout (from ``dagger_rm65_collect.py``)::

    <bundle>/episode_000000/{meta.json,trajectory.npz,images/<cam>/frame_*.ppm}

Output::

    <out_root>/<repo_id>/{meta/,data/,...}   # use_videos=False (embedded images)

Example::

    PYTHONPATH=. python3 scripts/dagger_rm65_to_lerobot.py \\
      --bundle_dir datasets/dagger_rm65_mock \\
      --out_root datasets/dagger_lerobot \\
      --repo_id rm65_dagger_mock \\
      --fps 20
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JOINT_KEYS = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
]


def _read_ppm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if not raw.startswith(b"P6"):
        raise ValueError(f"not a P6 PPM: {path}")
    # header: P6\nW H\n255\n
    parts = raw.split(b"\n", 3)
    # handle possible comment lines — our writer has no comments
    dims = parts[1].split()
    w, h = int(dims[0]), int(dims[1])
    body = parts[3]
    arr = np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3).copy()
    return arr


def _build_features(cameras: list[str], image_hw: tuple[int, int], use_video: bool) -> dict:
    from lerobot.common.datasets.utils import hw_to_dataset_features

    h, w = image_hw
    obs_hw: dict = {k: float for k in JOINT_KEYS}
    for cam in cameras:
        obs_hw[cam] = (h, w, 3)
    act_hw = {k: float for k in JOINT_KEYS}
    features = {
        **hw_to_dataset_features(act_hw, "action", use_video),
        **hw_to_dataset_features(obs_hw, "observation", use_video),
        "intervention": {"dtype": "float32", "shape": (1,), "names": None},
        "policy_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": list(JOINT_KEYS),
        },
    }
    return features


def convert_bundle(
    bundle_dir: Path,
    out_root: Path,
    repo_id: str,
    *,
    fps: int = 20,
    use_videos: bool = False,
) -> Path:
    bundle_dir = Path(bundle_dir)
    out_root = Path(out_root)
    ds_root = out_root / repo_id
    if ds_root.exists():
        shutil.rmtree(ds_root)

    ep_dirs = sorted(bundle_dir.glob("episode_*"))
    if not ep_dirs:
        raise FileNotFoundError(f"no episode_* under {bundle_dir}")

    # Probe first episode for cameras / size
    meta0 = json.loads((ep_dirs[0] / "meta.json").read_text(encoding="utf-8"))
    cameras = list(meta0.get("cameras") or [])
    if not cameras:
        # synthesize solid images if older bundles lack cameras
        cameras = ["front"]
        synth_images = True
        image_hw = (48, 64)
    else:
        synth_images = False
        sample = next((ep_dirs[0] / "images" / cameras[0]).glob("frame_*"))
        img0 = _read_ppm(sample)
        image_hw = (img0.shape[0], img0.shape[1])

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = _build_features(cameras, image_hw, use_video=use_videos)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=ds_root,
        robot_type="rm65_movep",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=2,
    )

    total_interv = 0
    for ep_dir in ep_dirs:
        meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
        traj = np.load(ep_dir / "trajectory.npz", allow_pickle=True)
        states = traj["states"]
        actions = traj["actions"]
        policy_actions = traj["policy_actions"]
        intervention = traj["intervention"].astype(bool)
        task = meta.get("task") or "dagger_task"
        n = states.shape[0]
        cams = list(meta.get("cameras") or cameras)

        for i in range(n):
            frame: dict = {
                "action": actions[i].astype(np.float32),
                "observation.state": states[i].astype(np.float32),
                "intervention": np.asarray([1.0 if intervention[i] else 0.0], dtype=np.float32),
                "policy_action": policy_actions[i].astype(np.float32),
            }
            for cam in cams:
                if synth_images:
                    h, w = image_hw
                    frame[f"observation.images.{cam}"] = np.full((h, w, 3), 128, dtype=np.uint8)
                else:
                    ppm = ep_dir / "images" / cam / f"frame_{i:06d}.ppm"
                    frame[f"observation.images.{cam}"] = _read_ppm(ppm)
            dataset.add_frame(frame, task=task)
            if intervention[i]:
                total_interv += 1
        dataset.save_episode()
        logger.info("converted %s frames=%d interv=%d", ep_dir.name, n, int(intervention.sum()))

    dataset.stop_image_writer()
    info = json.loads((ds_root / "meta" / "info.json").read_text(encoding="utf-8"))
    report = {
        "out": str(ds_root),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "intervention_frames": total_interv,
        "features": sorted(info.get("features", {}).keys()),
        "use_videos": use_videos,
    }
    (ds_root / "dagger_convert_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("done: %s", report)
    return ds_root


def verify_local_dataset(ds_root: Path) -> dict:
    """Offline structural checks (no HuggingFace)."""
    ds_root = Path(ds_root)
    info = json.loads((ds_root / "meta" / "info.json").read_text(encoding="utf-8"))
    n_ep = int(info["total_episodes"])
    n_fr = int(info["total_frames"])
    parquet = list(ds_root.rglob("*.parquet"))
    assert len(parquet) == n_ep, f"parquet count {len(parquet)} != episodes {n_ep}"

    import pyarrow.parquet as pq

    interv_sum = 0
    rows = 0
    for pf in sorted(parquet):
        table = pq.read_table(pf)
        rows += table.num_rows
        if "intervention" in table.column_names:
            col = table.column("intervention")
            # list/fixed size -> flatten
            vals = col.to_pylist()
            for v in vals:
                if isinstance(v, (list, tuple)):
                    interv_sum += int(float(v[0]) > 0.5)
                else:
                    interv_sum += int(float(v) > 0.5)
        assert "action" in table.column_names
        assert "observation.state" in table.column_names

    assert rows == n_fr, f"rows {rows} != total_frames {n_fr}"
    assert "intervention" in info["features"]
    result = {
        "ok": True,
        "episodes": n_ep,
        "frames": n_fr,
        "parquet_files": len(parquet),
        "intervention_sum": interv_sum,
        "has_intervention_feature": True,
    }
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_dir", type=str, default="")
    p.add_argument("--out_root", type=str, default="datasets/dagger_lerobot")
    p.add_argument("--repo_id", type=str, default="rm65_dagger_mock")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--use_videos", action="store_true", default=False)
    p.add_argument("--verify_only", type=str, default="", help="Only verify an existing dataset root")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        report = verify_local_dataset(Path(args.verify_only))
        report["pass"] = bool(report.get("ok") and report.get("intervention_sum", 0) > 0)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["pass"] else 1

    if not args.bundle_dir:
        raise SystemExit("--bundle_dir is required unless --verify_only is set")

    ds_root = convert_bundle(
        Path(args.bundle_dir),
        Path(args.out_root),
        args.repo_id,
        fps=args.fps,
        use_videos=args.use_videos,
    )
    report = verify_local_dataset(ds_root)
    report["pass"] = bool(
        report.get("ok")
        and report.get("episodes", 0) > 0
        and report.get("intervention_sum", 0) > 0
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
