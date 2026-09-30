#!/usr/bin/env python3
"""Write per-frame subtask language into cleaned LeRobot v2.1 datasets.

Shared vocab from datasets/cleaned/subtask_language.json.
Each train string is "{overall} {subtask}". Gripper open/close cuts grasp vs place;
place_to_slot and USB further split the closed-gripper window into align vs insert.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/sdznkj/zkx_ws/sendlerobot/datasets/cleaned")
VOCAB_PATH = ROOT / "subtask_language.json"
PAIR_DATASETS = {"rm65_stacking_blocks", "rm65_cleanup_the_desk"}
ALIGN_DATASETS = {"rm65_place_to_slot", "rm65_usb_insert"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )


def _feature_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def gripper_edges(grip: np.ndarray) -> tuple[list[int], list[int]]:
    closed = (grip < 0.5).astype(np.int32)
    d = np.diff(closed)
    closes = (np.where(d == 1)[0] + 1).tolist()
    opens = (np.where(d == -1)[0] + 1).tolist()
    return closes, opens


def split_align_insert(action: np.ndarray, start: int, end: int, fps: float) -> int:
    """Return cut index in [start+1, end-1] separating lift/align from insert."""
    n = end - start
    min_align = max(1, int(round(0.6 * fps)))
    min_insert = max(1, int(round(0.8 * fps)))
    if n <= 2:
        return start + max(1, n // 2)
    if n <= min_align + min_insert:
        return start + max(1, min(n - 1, int(round(n * 0.45))))
    j6 = action[start:end, 5]
    dj = np.abs(np.diff(j6, prepend=j6[:1]))
    cum = np.cumsum(dj)
    total = float(cum[-1])
    if total < 12.0:
        cut = min(n - min_insert, max(min_align, int(round(1.0 * fps))))
        return start + int(cut)
    cut = int(np.searchsorted(cum, 0.80 * total))
    cut = max(min_align, min(n - min_insert, cut))
    return start + int(cut)


def assign_pair(n: int, grip: np.ndarray, phrases: list[str]) -> tuple[list[int], list[dict]]:
    """2 phrases per object: grasp (open), place+release (closed)."""
    closes, opens = gripper_edges(grip)
    n_pairs = max(1, len(phrases) // 2)
    labels = np.zeros(n, dtype=np.int64)
    windows: list[dict] = []
    t = 0
    ci = oi = 0
    for i in range(max(n_pairs, len(closes))):
        pair = min(i, n_pairs - 1)
        grasp_i = 2 * pair
        place_i = 2 * pair + 1
        close = closes[ci] if ci < len(closes) else n
        ci += 1
        if close < t:
            close = t
        open_t = n
        while oi < len(opens) and opens[oi] <= close:
            oi += 1
        if oi < len(opens):
            open_t = opens[oi]
            oi += 1
        if close > t:
            labels[t:close] = grasp_i
            windows.append({"start": t, "end": close, "local_index": int(grasp_i), "kind": "grasp"})
        if open_t > close:
            labels[close:open_t] = place_i
            windows.append({"start": close, "end": open_t, "local_index": int(place_i), "kind": "place"})
        t = max(t, open_t)
        if t >= n:
            break
    if t < n:
        last = int(labels[t - 1]) if t > 0 else 2 * (n_pairs - 1) + 1
        labels[t:] = last
        windows.append({"start": t, "end": n, "local_index": last, "kind": "tail"})
    return labels.tolist(), windows


def assign_align(n: int, action: np.ndarray, phrases: list[str], fps: float) -> tuple[list[int], list[dict]]:
    """3 phrases per object: pre-grasp, lift/align, insert+release."""
    grip = action[:, -1]
    closes, opens = gripper_edges(grip)
    n_obj = max(1, len(phrases) // 3)
    labels = np.zeros(n, dtype=np.int64)
    windows: list[dict] = []
    t = 0
    ci = oi = 0
    n_cycles = max(n_obj, len(closes))
    for i in range(n_cycles):
        obj = min(i, n_obj - 1)
        grasp_i = 3 * obj
        align_i = 3 * obj + 1
        insert_i = 3 * obj + 2
        close = closes[ci] if ci < len(closes) else n
        ci += 1
        if close < t:
            close = t
        open_t = n
        while oi < len(opens) and opens[oi] <= close:
            oi += 1
        if oi < len(opens):
            open_t = opens[oi]
            oi += 1
        if close > t:
            labels[t:close] = grasp_i
            windows.append({"start": t, "end": close, "local_index": int(grasp_i), "kind": "grasp"})
        if open_t > close:
            cut = split_align_insert(action, close, open_t, fps)
            cut = max(close + 1, min(open_t - 1, cut)) if open_t - close >= 2 else close
            if cut > close:
                labels[close:cut] = align_i
                windows.append({"start": close, "end": cut, "local_index": int(align_i), "kind": "align"})
            if open_t > cut:
                labels[cut:open_t] = insert_i
                windows.append({"start": cut, "end": open_t, "local_index": int(insert_i), "kind": "insert"})
        t = max(t, open_t)
        if t >= n:
            break
    if t < n:
        last = int(labels[t - 1]) if t > 0 else 3 * (n_obj - 1) + 2
        labels[t:] = last
        windows.append({"start": t, "end": n, "local_index": last, "kind": "tail"})
    return labels.tolist(), windows


def apply_one(root: Path, spec: dict) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 20))
    overall = spec["overall"]
    phrases = list(spec["subtasks"])
    train_tasks = [f"{overall} {p}" for p in phrases]
    failed_task = None
    name = root.name
    if name == "rm65_usb_insert":
        failed_task = "failed"
        train_tasks.append(failed_task)

    tasks_rows = [{"task_index": i, "task": t} for i, t in enumerate(train_tasks)]
    _write_jsonl(root / "meta" / "tasks.jsonl", tasks_rows)

    episodes = _load_jsonl(root / "meta" / "episodes.jsonl")
    stats_rows = _load_jsonl(root / "meta" / "episodes_stats.jsonl")
    stats_by_ep = {int(r["episode_index"]): r for r in stats_rows}
    plan_eps = []
    train_ids = []
    failed_ids = []

    for ep in episodes:
        ep_i = int(ep["episode_index"])
        pq = root / info["data_path"].format(
            episode_chunk=ep_i // int(info.get("chunks_size", 1000)),
            episode_index=ep_i,
        )
        df = pd.read_parquet(pq)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        n = len(df)
        is_failed = name == "rm65_usb_insert" and (
            "failed" in (ep.get("tasks") or []) or int(df["task_index"].iloc[0]) == 1
        )
        if is_failed:
            failed_idx = train_tasks.index(failed_task)
            labels = [failed_idx] * n
            windows = [{"start": 0, "end": n, "local_index": failed_idx, "kind": "failed"}]
            ep_task_names = [failed_task]
            failed_ids.append(ep_i)
        else:
            if name in PAIR_DATASETS:
                local, windows = assign_pair(n, action[:, -1], phrases)
            elif name in ALIGN_DATASETS:
                local, windows = assign_align(n, action, phrases, fps)
            else:
                raise ValueError(name)
            labels = local
            for w in windows:
                w["task"] = train_tasks[int(w["local_index"])]
            used = []
            for idx in labels:
                t = train_tasks[int(idx)]
                if t not in used:
                    used.append(t)
            ep_task_names = used
            train_ids.append(ep_i)

        df["task_index"] = np.asarray(labels, dtype=np.int64)
        df.to_parquet(pq, index=False)
        ep["tasks"] = ep_task_names

        st = stats_by_ep.get(ep_i, {"episode_index": ep_i, "stats": {}})
        st["stats"]["task_index"] = _feature_stats(np.asarray(labels, dtype=np.int64))
        stats_by_ep[ep_i] = st
        plan_eps.append(
            {
                "episode_index": ep_i,
                "n_frames": n,
                "windows": windows,
                "failed": bool(is_failed),
            }
        )

    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", [stats_by_ep[int(e["episode_index"])] for e in episodes])

    info["total_tasks"] = len(train_tasks)
    if failed_ids:
        parts = []
        # contiguous ranges over train_ids
        ids = sorted(train_ids)
        if ids:
            s = prev = ids[0]
            ranges = []
            for x in ids[1:]:
                if x == prev + 1:
                    prev = x
                else:
                    ranges.append(f"{s}:{prev + 1}")
                    s = prev = x
            ranges.append(f"{s}:{prev + 1}")
            info["splits"] = {
                "train": ",".join(ranges),
                "failed": ",".join(f"{i}:{i + 1}" for i in failed_ids),
            }
        else:
            info["splits"] = {"train": "0:0", "failed": f"0:{info['total_episodes']}"}
    else:
        info["splits"] = {"train": f"0:{info['total_episodes']}"}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    plan = {
        "schema": "rm65/subtask_plan/v1",
        "dataset": name,
        "overall": overall,
        "subtasks": phrases,
        "train_string": "{overall} {subtask}",
        "fps": fps,
        "episodes": plan_eps,
    }
    (root / "meta" / "subtask_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "dataset": name,
        "episodes": len(episodes),
        "tasks": len(train_tasks),
        "failed": failed_ids,
        "splits": info["splits"],
    }


def main() -> int:
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    summaries = []
    for name, spec in vocab["datasets"].items():
        root = ROOT / name
        print(f"apply {name} ...", flush=True)
        summaries.append(apply_one(root, spec))
        print(f"  tasks={summaries[-1]['tasks']} splits={summaries[-1]['splits']}", flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
