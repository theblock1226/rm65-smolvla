#!/usr/bin/env python3
"""Propose and apply intra-episode trims for RM65 LeRobot v2.1 datasets.

Does not modify the source dataset. Default proposals only cut leading/trailing
idle (no motion). Internal pauses and joint jumps are listed for review.

  python3 scripts/rm65_trim_idle.py propose datasets/small_tasks/rm65_stacking_blocks
  python3 scripts/rm65_trim_idle.py apply datasets/small_tasks/_clean_preview/rm65_stacking_blocks/trim_plan.json \\
      --output datasets/cleaned/rm65_stacking_blocks
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_DIMS = 6


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def video_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def episode_parquet(root: Path, info: dict, ep: int) -> Path:
    chunk = ep // int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    return root / rel


def episode_video(root: Path, info: dict, ep: int, key: str) -> Path:
    chunk = ep // int(info.get("chunks_size", 1000))
    rel = info["video_path"].format(episode_chunk=chunk, episode_index=ep, video_key=key)
    return root / rel


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    n = len(mask)
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if start is not None and (not v or i == n - 1):
            end = i + 1 if v and i == n - 1 else i
            if end > start:
                runs.append((start, end))
            start = None
    return runs


def motion_mask(
    action: np.ndarray,
    idle_deg: float,
    gripper_settle_frames: int = 0,
) -> np.ndarray:
    """True where the arm is moving, or the gripper is actuating.

    Gripper is recorded as a 0/1 command jump, not the physical close. After a
    command change the arm often stays still while the jaws finish, so those
    frames are treated as motion (not idle).
    """
    t = len(action)
    moving = np.zeros(t, dtype=bool)
    if t < 2:
        return moving
    djoint = np.linalg.norm(np.diff(action[:, :JOINT_DIMS], axis=0), axis=1)
    dgrip = np.abs(np.diff(action[:, -1]))
    moving[1:] = (djoint >= idle_deg) | (dgrip >= 0.005)
    if gripper_settle_frames > 0:
        pre = max(1, int(round(gripper_settle_frames * 0.1)))
        for i in np.where(dgrip >= 0.05)[0] + 1:
            lo = max(0, int(i) - pre)
            hi = min(t, int(i) + int(gripper_settle_frames) + 1)
            moving[lo:hi] = True
    moving[0] = moving[1]
    return moving


def detect_joint_jerks(
    action: np.ndarray,
    fps: float,
    jump_deg: float = 8.0,
    min_l2: float = 1.8,
    ratio: float = 10.0,
    window_s: float = 1.0,
) -> list[dict]:
    """Flag snaps: huge single-axis jump, or a burst after ~1s of near-idle.

    ep000-style camera/arm shake is often only 2–4° in 50ms after holding still,
    which a fixed 8° threshold misses.
    """
    if len(action) < 3:
        return []
    d = np.diff(action[:, :JOINT_DIMS].astype(np.float64), axis=0)
    l2 = np.linalg.norm(d, axis=1)
    maxax = np.max(np.abs(d), axis=1)
    w = max(1, int(round(window_s * fps)))
    pad = max(1, int(round(0.25 * fps)))
    merge = max(1, int(round(0.5 * fps)))
    raw: list[dict] = []
    i = 0
    n = len(l2)
    while i < n:
        lo = max(0, i - w)
        med = float(np.median(l2[lo:i])) if i > lo else 0.0
        sudden = med < 0.35 and l2[i] >= min_l2 and l2[i] >= ratio * max(med, 0.04)
        hard = maxax[i] >= jump_deg
        if sudden or hard:
            j = i + 1
            peak = i
            while j < n and (j - i) <= merge:
                if l2[j] > l2[peak]:
                    peak = j
                if maxax[j] >= jump_deg or l2[j] >= min_l2:
                    j += 1
                else:
                    break
            start = max(0, i)
            end = min(len(action), j + 1)
            raw.append(
                {
                    "i0": start,
                    "i1": end,
                    "peak": peak + 1,
                    "l2": float(l2[peak]),
                    "maxax": float(maxax[peak]),
                    "mode": "hard" if hard else "sudden",
                }
            )
            i = j
        else:
            i += 1
    merged: list[dict] = []
    for ev in raw:
        if merged and ev["i0"] - merged[-1]["i1"] <= merge:
            prev = merged[-1]
            prev["i1"] = ev["i1"]
            if ev["l2"] > prev["l2"]:
                prev["l2"] = ev["l2"]
                prev["maxax"] = ev["maxax"]
                prev["peak"] = ev["peak"]
                prev["mode"] = ev["mode"]
        else:
            merged.append(dict(ev))
    out = []
    n_frames = len(action)
    for ev in merged:
        start = max(0, ev["peak"] - pad)
        end = min(n_frames, ev["peak"] + pad)
        t = ev["peak"] / fps
        if ev["mode"] == "hard":
            reason = (
                f"{t:.2f}s 单轴跳变 {ev['maxax']:.1f}°（L2 {ev['l2']:.1f}°），"
                "可能是误触/姿态翻转"
            )
        else:
            reason = (
                f"{t:.2f}s 静止后突然抽动：单帧 L2 {ev['l2']:.1f}°、最大单轴 {ev['maxax']:.1f}°。"
                "关节阈值 8° 测不到，但相机/手臂会猛抖一下"
            )
        out.append(
            {
                "kind": "joint_jump",
                "start": start,
                "end": end,
                "seconds": round((end - start) / fps, 2),
                "apply": False,
                "frame": ev["peak"],
                "max_deg": round(ev["maxax"], 2),
                "l2_deg": round(ev["l2"], 2),
                "reason": reason,
            }
        )
    return out


def _signed_wrist_runs(j6: np.ndarray, fps: float, dead: float = 0.12, gap_s: float = 0.4) -> list[dict]:
    dj = np.diff(j6.astype(np.float64))
    sig = np.sign(np.where(np.abs(dj) >= dead, dj, 0.0))
    raw: list[list] = []
    s = 0
    for k in range(1, len(sig) + 1):
        if k == len(sig) or sig[k] != sig[s]:
            if sig[s] != 0:
                raw.append([s, k, float(dj[s:k].sum()), int(sig[s])])
            s = k
    if not raw:
        return []
    gap = max(1, int(round(gap_s * fps)))
    merged = [raw[0][:]]
    for r in raw[1:]:
        prev = merged[-1]
        if r[3] == prev[3] and r[0] - prev[1] <= gap:
            prev[1] = r[1]
            prev[2] += r[2]
        else:
            merged.append(r[:])
    return [
        {"start": a, "end": b, "net": net, "dir": d, "seconds": (b - a) / fps}
        for a, b, net, d in merged
    ]


def detect_wrong_rotate(j6: np.ndarray, fps: float, min_deg: float = 20.0) -> dict | None:
    """First wrist rotation: a very short reverse, then a longer turn the other way.

    Marks only the short reverse. The following long alignment is kept.
    """
    if len(j6) < int(3 * fps):
        return None
    runs = _signed_wrist_runs(j6, fps)
    max_short_s = 1.5
    max_gap_s = 1.0
    min_short_deg = 2.0
    min_span = max(3, int(round(0.15 * fps)))

    for i, long in enumerate(runs):
        if abs(long["net"]) < min_deg or long["seconds"] < 2.0:
            continue
        if i == 0:
            return None
        short = runs[i - 1]
        gap = (long["start"] - short["end"]) / fps
        if gap > max_gap_s:
            return None
        if short["dir"] == long["dir"]:
            return None
        if short["seconds"] > max_short_s or abs(short["net"]) < min_short_deg:
            return None
        if not (
            long["seconds"] >= 1.8 * short["seconds"]
            or abs(long["net"]) >= 2.5 * abs(short["net"])
        ):
            return None
        start, end = int(short["start"]), int(short["end"])
        if end - start < min_span:
            return None
        return {
            "kind": "wrong_rotate",
            "start": start,
            "end": end,
            "seconds": round((end - start) / fps, 2),
            "apply": False,
            "reason": (
                f"第一次旋转开头短反向 {short['net']:+.0f}° / {short['seconds']:.2f}s，"
                f"随后连续 {long['net']:+.0f}° / {long['seconds']:.1f}s。只标这短反向。"
            ),
        }
    return None


def propose_episode(
    action: np.ndarray,
    fps: float,
    idle_deg: float,
    min_cut_frames: int,
    internal_min_frames: int,
    jump_deg: float,
    gripper_settle_frames: int = 0,
) -> dict:
    n = len(action)
    moving = motion_mask(action, idle_deg, gripper_settle_frames=gripper_settle_frames)
    idle = ~moving

    first = next((i for i in range(n) if moving[i]), 0)
    last = next((i for i in range(n - 1, -1, -1) if moving[i]), n - 1)
    keep_start, keep_end = first, last + 1

    cuts = []
    if keep_start >= min_cut_frames:
        cuts.append(
            {
                "kind": "lead_idle",
                "start": 0,
                "end": keep_start,
                "seconds": round(keep_start / fps, 2),
                "apply": True,
                "reason": "开头静止/未开始操作",
            }
        )
    else:
        keep_start = 0
    trail = n - keep_end
    if trail >= min_cut_frames:
        cuts.append(
            {
                "kind": "trail_idle",
                "start": keep_end,
                "end": n,
                "seconds": round(trail / fps, 2),
                "apply": True,
                "reason": "结尾静止/操作已结束",
            }
        )
    else:
        keep_end = n

    for s, e in _runs(idle):
        if e - s < internal_min_frames:
            continue
        if s < keep_start or e > keep_end:
            continue
        if s == keep_start or e == keep_end:
            continue
        cuts.append(
            {
                "kind": "internal_idle",
                "start": s,
                "end": e,
                "seconds": round((e - s) / fps, 2),
                "apply": False,
                "reason": (
                    f"中间停住约 {(e - s) / fps:.1f}s（臂停且夹爪未动作；"
                    "夹取闭合过程已排除。默认不裁）"
                ),
            }
        )

    rot = detect_wrong_rotate(action[:, 5], fps)
    if rot:
        cuts.append(rot)

    jerks = detect_joint_jerks(action, fps, jump_deg=jump_deg)
    cuts.extend(jerks)
    flags = [
        {
            "kind": "joint_jump",
            "frame": j["frame"],
            "seconds": round(j["frame"] / fps, 2),
            "max_deg": j["max_deg"],
            "reason": j["reason"],
        }
        for j in jerks
    ]

    return {
        "n_frames": n,
        "keep": {
            "start": keep_start,
            "end": keep_end,
            "apply": keep_end - keep_start < n,
        },
        "cuts": cuts,
        "flags": flags,
        "idle_fraction": round(float(idle.mean()), 4),
        "duration_s": round(n / fps, 2),
        "keep_s": round((keep_end - keep_start) / fps, 2),
        "cut_s": round((n - (keep_end - keep_start)) / fps, 2),
    }


def write_html(plan: dict, html_path: Path, source: Path) -> None:
    plan = dict(plan)
    plan.setdefault("chunks_size", 1000)
    plan["dataset_name"] = source.name
    plan["video_url_root"] = f"/small_tasks/{source.name}"
    payload = json.dumps(plan, ensure_ascii=False).replace("<", "\\u003c")
    tmpl = Path(__file__).with_name("rm65_trim_preview.html").read_text(encoding="utf-8")
    html_path.write_text(tmpl.replace("__PLAN_JSON__", payload), encoding="utf-8")


def cmd_propose(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    info = load_info(source)
    fps = float(info["fps"])
    n_ep = int(info["total_episodes"])
    keys = video_keys(info)
    min_cut = max(1, int(round(args.min_cut_s * fps)))
    internal_min = max(1, int(round(args.internal_min_s * fps)))
    grip_settle = max(0, int(round(args.gripper_settle_s * fps)))

    episodes = []
    for ep in range(n_ep):
        df = pd.read_parquet(episode_parquet(source, info, ep))
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        rec = propose_episode(
            action,
            fps=fps,
            idle_deg=args.idle_deg,
            min_cut_frames=min_cut,
            internal_min_frames=internal_min,
            jump_deg=args.jump_deg,
            gripper_settle_frames=grip_settle,
        )
        rec["episode_index"] = ep
        rec["task"] = None
        episodes.append(rec)
        print(
            f"ep {ep:03d}: {rec['duration_s']:.1f}s -> {rec['keep_s']:.1f}s  "
            f"cuts={len(rec['cuts'])} flags={len(rec['flags'])}",
            flush=True,
        )

    cut_s = sum(e["cut_s"] for e in episodes)
    keep_s = sum(e["keep_s"] for e in episodes)
    src_s = sum(e["duration_s"] for e in episodes)
    plan = {
        "source": str(source),
        "fps": fps,
        "idle_deg": args.idle_deg,
        "min_cut_s": args.min_cut_s,
        "internal_min_s": args.internal_min_s,
        "gripper_settle_s": args.gripper_settle_s,
        "jump_deg": args.jump_deg,
        "video_keys": keys,
        "chunks_size": int(info.get("chunks_size", 1000)),
        "video_url_root": f"/small_tasks/{source.name}",
        "summary": {
            "n_episodes": n_ep,
            "src_seconds": src_s,
            "keep_seconds": keep_s,
            "cut_seconds": cut_s,
        },
        "episodes": episodes,
    }

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else source.parent / "_clean_preview" / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "trim_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = out_dir / "index.html"
    write_html(plan, html_path, source)
    print(f"\nplan: {plan_path}")
    print(f"html: {html_path}")
    print(f"preview: http://192.168.73.204:8765/small_tasks/_clean_preview/{source.name}/index.html")
    print(f"summary: cut {cut_s:.0f}s / {src_s:.0f}s")
    return 0


def _keep_windows(ep: dict) -> list[tuple[int, int]]:
    n = ep["n_frames"]
    applied = [c for c in ep.get("cuts") or [] if c.get("apply")]
    if not applied:
        return [(0, n)]
    cut_mask = np.zeros(n, dtype=bool)
    for c in applied:
        cut_mask[int(c["start"]) : int(c["end"])] = True
    keep = ~cut_mask
    if not keep.any():
        return []
    return _runs(keep)


def _indices_to_windows(idx: np.ndarray, n: int) -> list[tuple[int, int]]:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(idx, dtype=np.int64)] = True
    return _runs(mask)


def _parse_video_shifts(items: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--video-shift 需要 key=frames，例如 observation.images.mono=10，收到: {item}")
        key, raw = item.split("=", 1)
        frames = int(raw)
        if frames < 0:
            raise SystemExit("--video-shift 只支持把某路画面往前配（正整数）")
        out[key.strip()] = frames
    return out


def _feature_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    axis = 0
    keepdims = arr.ndim == 1
    return {
        "min": np.min(arr, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(arr, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(arr, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(arr, axis=axis, keepdims=keepdims).tolist(),
        "count": [int(arr.shape[0])],
    }


def _trim_video(src: Path, dst: Path, windows: list[tuple[int, int]], fps: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if len(windows) == 1 and windows[0][0] == 0:
        # maybe full copy — caller handles
        pass
    import av

    src_c = av.open(str(src))
    vstream = src_c.streams.video[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(dst), "w")
    ostream = out.add_stream("h264", rate=int(round(fps)))
    ostream.width = vstream.width
    ostream.height = vstream.height
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"preset": "veryfast", "crf": "23"}

    wanted = []
    for s, e in windows:
        wanted.extend(range(s, e))
    wanted_set = set(wanted)
    idx = 0
    written = 0
    for frame in src_c.decode(video=0):
        if idx in wanted_set:
            frame.pts = None
            frame.time_base = ostream.codec_context.time_base
            for packet in ostream.encode(frame.reformat(format="yuv420p")):
                out.mux(packet)
            written += 1
        idx += 1
        if written >= len(wanted):
            break
    for packet in ostream.encode():
        out.mux(packet)
    out.close()
    src_c.close()


def cmd_apply(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    source = Path(plan["source"])
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"输出目录非空，拒绝覆盖: {out}  （加 --force 才覆盖）")
        shutil.rmtree(out)
    info = load_info(source)
    fps = float(plan["fps"])
    keys = plan["video_keys"]
    shifts = _parse_video_shifts(getattr(args, "video_shift", None))
    max_shift = max(shifts.values()) if shifts else 0
    unknown = set(shifts) - set(keys)
    if unknown:
        raise SystemExit(f"--video-shift 里有数据集没有的 video key: {sorted(unknown)}")
    orig_stats = {}
    stats_path = source / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        for line in stats_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                orig_stats[int(rec["episode_index"])] = rec["stats"]

    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)

    failed = {int(x) for x in (plan.get("failed_episodes") or [])}
    skip_trim = {int(x) for x in (plan.get("skip_trim_episodes") or [])}
    tasks_src = source / "meta" / "tasks.jsonl"
    orig_tasks = []
    if tasks_src.is_file():
        orig_tasks = [json.loads(l) for l in tasks_src.read_text().splitlines() if l.strip()]
    main_task = orig_tasks[0]["task"] if orig_tasks else "task"
    out_tasks = [{"task_index": 0, "task": main_task}]
    if failed:
        out_tasks.append({"task_index": 1, "task": "failed"})
    (out / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_tasks), encoding="utf-8"
    )

    episodes_out = []
    stats_out = []
    global_index = 0
    new_ep = 0
    for ep in plan["episodes"]:
        old_i = int(ep["episode_index"])
        if old_i in failed or old_i in skip_trim:
            windows = [(0, int(ep["n_frames"]))]
        else:
            windows = _keep_windows(ep)
        if not windows:
            print(f"skip ep {old_i}: keep empty")
            continue
        df = pd.read_parquet(episode_parquet(source, info, old_i))
        n_src = int(ep["n_frames"])
        idx_keep = np.concatenate([np.arange(s, e) for s, e in windows])
        if max_shift:
            idx_keep = idx_keep[idx_keep + max_shift < n_src]
        if len(idx_keep) == 0:
            print(f"skip ep {old_i}: empty after video shift")
            continue
        part = df.iloc[idx_keep].copy().reset_index(drop=True)
        n = len(part)
        part["frame_index"] = np.arange(n, dtype=np.int64)
        part["episode_index"] = np.int64(new_ep)
        part["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
        part["timestamp"] = (part["frame_index"].astype(np.float32) / np.float32(fps))
        task_index = 1 if old_i in failed else 0
        if "task_index" in part.columns:
            part["task_index"] = np.int64(task_index)
        chunk = new_ep // int(info.get("chunks_size", 1000))
        pq = out / info["data_path"].format(episode_chunk=chunk, episode_index=new_ep)
        pq.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(pq, index=False)

        full_copy = (not shifts) and windows == [(0, ep["n_frames"])]
        for key in keys:
            src_v = episode_video(source, info, old_i, key)
            dst_v = out / info["video_path"].format(episode_chunk=chunk, episode_index=new_ep, video_key=key)
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            sh = int(shifts.get(key, 0))
            idx_v = idx_keep + sh
            win_v = _indices_to_windows(idx_v, n_src)
            if full_copy and sh == 0:
                shutil.copy2(src_v, dst_v)
            else:
                print(
                    f"encode {dst_v.name} ep {old_i}->{new_ep} shift={sh} windows={win_v}",
                    flush=True,
                )
                _trim_video(src_v, dst_v, win_v, fps)

        length = n
        if old_i in failed:
            task_name = ["failed"]
            print(f"failed category ep {old_i}->{new_ep} full {length} frames", flush=True)
        elif old_i in skip_trim:
            task_name = [main_task]
            print(f"skip trim ep {old_i}->{new_ep} full {length} frames", flush=True)
        else:
            task_name = [main_task]
        episodes_out.append({"episode_index": new_ep, "tasks": task_name, "length": length})

        st = {}
        for col in ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in part.columns:
                st[col] = _feature_stats(np.stack(part[col].to_numpy()) if col in ("action", "observation.state") else part[col].to_numpy())
        old_st = orig_stats.get(old_i, {})
        for key in keys:
            if key in old_st:
                img = dict(old_st[key])
                img["count"] = [max(1, int(math.sqrt(n)))]
                st[key] = img
        stats_out.append({"episode_index": new_ep, "stats": st})
        global_index += n
        new_ep += 1

    (out / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in episodes_out), encoding="utf-8"
    )
    (out / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in stats_out), encoding="utf-8"
    )
    new_info = dict(info)
    new_info["total_episodes"] = new_ep
    new_info["total_frames"] = global_index
    new_info["total_videos"] = new_ep * len(keys)
    new_info["total_tasks"] = len(out_tasks)
    new_info["splits"] = {"train": f"0:{new_ep}"}
    if shifts:
        new_info["video_shift"] = shifts
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=4) + "\n", encoding="utf-8")
    plan_out = dict(plan)
    if shifts:
        plan_out["video_shift"] = shifts
    (out / "meta" / "trim_plan.json").write_text(
        json.dumps(plan_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {new_ep} episodes, {global_index} frames -> {out}")
    if shifts:
        print(f"video shift: {shifts} （关节/未列出的相机用第 i 帧，被平移的相机用第 i+shift 帧）")
    print("未重算全局 meta/stats.json；训练前可用 scripts/rm65_dataset_finalize.py 补齐。")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RM65 LeRobot idle trim (propose / apply)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("propose", help="Detect idle ranges and write a review plan")
    sp.add_argument("source")
    sp.add_argument("--out-dir", default="")
    sp.add_argument("--idle-deg", type=float, default=0.08, help="Joint L2 delta below this counts as idle")
    sp.add_argument("--min-cut-s", type=float, default=0.4, help="Ignore lead/trail idle shorter than this")
    sp.add_argument("--internal-min-s", type=float, default=0.8, help="List internal idle longer than this")
    sp.add_argument(
        "--gripper-settle-s",
        type=float,
        default=2.0,
        help="After a gripper open/close command, do not mark the next N seconds as idle",
    )
    sp.add_argument("--jump-deg", type=float, default=8.0, help="Flag single-frame joint jumps")
    sp.set_defaults(func=cmd_propose)

    sa = sub.add_parser("apply", help="Write a trimmed copy from a plan JSON")
    sa.add_argument("plan")
    sa.add_argument("--output", required=True)
    sa.add_argument("--force", action="store_true")
    sa.add_argument(
        "--video-shift",
        action="append",
        default=[],
        metavar="KEY=FRAMES",
        help="Pair a delayed camera forward, e.g. observation.images.mono=10",
    )
    sa.set_defaults(func=cmd_apply)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
