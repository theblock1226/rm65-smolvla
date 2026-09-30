#!/usr/bin/env python3
"""Time the 20 Hz schedule against real SmolVLA inference. No robot, no cameras.

Playback pops one absolute joint target every 50 ms and never waits on images.
The next chunk starts as soon as the GPU is free. Each chunk is 24 absolute
steps (q+Δq already applied inside the policy). Overlap is averaged in joint
space. An empty queue repeats the last command instead of blocking.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"
OUT = ROOT / "outputs/diag/ep000_schedule_dryrun"
HZ = 20
DT = 1.0 / HZ


def load_episode(episode: int) -> dict:
    info = json.loads((DS / "meta/info.json").read_text())
    tasks = {}
    for line in (DS / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    import pandas as pd

    pq = DS / info["data_path"].format(episode_chunk=0, episode_index=episode)
    df = pd.read_parquet(pq)
    return {
        "state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        "action": np.stack(df["action"].to_numpy()).astype(np.float32),
        "task_index": df["task_index"].to_numpy().astype(np.int64),
        "tasks": tasks,
        "info": info,
        "n": len(df),
    }


class _Video:
    def __init__(self, path: Path):
        import av

        self._container = av.open(str(path))
        self._gen = self._container.decode(video=0)
        self._idx = -1
        self._frame: np.ndarray | None = None

    def get(self, index: int) -> np.ndarray:
        while self._idx < index:
            self._frame = next(self._gen).to_ndarray(format="rgb24")
            self._idx += 1
        if self._frame is None:
            raise RuntimeError(f"no frame at {index}")
        return self._frame

    def close(self) -> None:
        self._container.close()


def _rgb(img: np.ndarray, torch):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def _stats(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}: 无"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"{name}: n={len(arr)} 中位 {np.median(arr):.3f}s  "
        f"p95 {np.quantile(arr, 0.95):.3f}s  最大 {arr.max():.3f}s  最小 {arr.min():.3f}s"
    )


def main() -> None:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(ROOT))
    import torch

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.scripts.server.rm65_ckpt import load_dataset_stats_from_ckpt, trim_stats_to_config

    episode = 0
    data = load_episode(episode)
    info = data["info"]
    n = data["n"]
    config = PreTrainedConfig.from_pretrained(str(CKPT))
    if not getattr(config, "use_relative_actions", False):
        raise SystemExit("checkpoint 不是相对关节，拒绝把增量当成绝对角度")
    if config.rtc_config is not None and getattr(config.rtc_config, "enabled", False):
        raise SystemExit("rtc_config 已打开。这次试验不走去噪内部的 RTC")
    stats = trim_stats_to_config(load_dataset_stats_from_ckpt(CKPT), config)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats or None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("没有 CUDA，这次计时没有意义")
    policy.to(device)
    policy.eval()
    horizon = int(policy.config.chunk_size)
    budget = horizon * DT
    print(
        f"干跑 ep{episode:03d}  {n} 帧  chunk={horizon} 步  "
        f"n_action_steps={int(policy.config.n_action_steps)}  "
        f"预算 {budget:.2f}s/块  下发 {HZ} Hz  device={device}  机械臂不动",
        flush=True,
    )

    front = _Video(
        DS / info["video_path"].format(episode_chunk=0, episode_index=episode, video_key="observation.images.front")
    )
    astra = _Video(
        DS / info["video_path"].format(episode_chunk=0, episode_index=episode, video_key="observation.images.astra")
    )
    video_lock = threading.Lock()

    def infer_at(index: int) -> np.ndarray:
        index = min(index, n - 1)
        with video_lock:
            front_img = front.get(index)
            astra_img = astra.get(index)
        policy.reset()
        state_np = data["state"][index, :7].copy()
        batch = {
            "task": data["tasks"][int(data["task_index"][index])],
            "observation.state": torch.from_numpy(state_np).unsqueeze(0).to(device),
            "observation.images.camera1": _rgb(front_img, torch).to(device),
            "observation.images.camera2": _rgb(astra_img, torch).to(device),
        }
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)[0].detach().float().cpu().numpy()
        torch.cuda.synchronize()
        if chunk.shape[0] < horizon:
            raise RuntimeError(f"模型只返回 {chunk.shape[0]} 步，不够 {horizon}")
        return chunk[:horizon].astype(np.float32)

    print("预热一次，不计入节拍", flush=True)
    warm_t = time.perf_counter()
    warm = infer_at(0)
    warm_dt = time.perf_counter() - warm_t
    print(f"预热 {warm_dt:.3f}s  输出 {tuple(warm.shape)}  关节0={np.round(warm[0, :6], 1).tolist()}", flush=True)

    clock_lock = threading.Lock()
    clock = {"step": 0}
    queue_lock = threading.Lock()
    queue: dict[int, np.ndarray] = {}
    stop = threading.Event()
    error: list[BaseException] = []
    infer_dts: list[float] = []
    overlaps: list[int] = []
    ages: list[int] = []
    aligned_left: list[int] = []
    queued_before: list[int] = []

    def playback_step() -> int:
        with clock_lock:
            return clock["step"]

    def inference_loop() -> None:
        try:
            while not stop.is_set():
                s_start = playback_step()
                if s_start >= n:
                    return
                t0 = time.perf_counter()
                chunk = infer_at(s_start)
                dt = time.perf_counter() - t0
                s_now = playback_step()
                age = max(0, s_now - s_start)
                infer_dts.append(dt)
                blended = 0
                # Chunk index 0 is applied at arrival. The image it saw is `age`
                # steps old. Aligning it back to that image would leave only
                # horizon-age steps still in the future, which does not fill the
                # next 1.2 s.
                with queue_lock:
                    pending = sum(1 for step in queue if step >= s_now)
                    for i in range(horizon):
                        step = s_now + i
                        if step >= n:
                            break
                        new = chunk[i]
                        old = queue.get(step)
                        if old is None:
                            queue[step] = new.copy()
                        else:
                            merged = new.copy()
                            merged[:6] = 0.5 * (old[:6] + new[:6])
                            queue[step] = merged
                            blended += 1
                ages.append(age)
                overlaps.append(blended)
                queued_before.append(pending)
                aligned_left.append(max(0, horizon - age))
                print(
                    f"  推理#{len(infer_dts):02d} 起点步 {s_start:4d}  耗时 {dt:.3f}s  "
                    f"画面落后 {age} 步  到达时还剩 {pending} 步  重叠平均 {blended}  "
                    f"若按观测时刻对齐只剩 {max(0, horizon - age)} 步",
                    flush=True,
                )
        except BaseException as exc:  # noqa: BLE001 - report and stop the clock
            error.append(exc)
            stop.set()

    worker = threading.Thread(target=inference_loop, name="infer", daemon=True)
    worker.start()

    commanded = np.full((n, 7), np.nan, dtype=np.float32)
    last = data["state"][0, :7].astype(np.float32).copy()
    warmup_holds = 0
    underruns = 0
    tick_dts: list[float] = []
    first_chunk_seen = False
    t_next = time.perf_counter()
    t_prev = t_next
    try:
        for step in range(n):
            with clock_lock:
                clock["step"] = step
            with queue_lock:
                cmd = queue.pop(step, None)
            if cmd is None:
                if first_chunk_seen:
                    underruns += 1
                else:
                    warmup_holds += 1
                cmd = last
            else:
                first_chunk_seen = True
            commanded[step] = cmd
            last = cmd
            now = time.perf_counter()
            tick_dts.append(now - t_prev)
            t_prev = now
            t_next += DT
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()
            if step % HZ == 0:
                print(f"t={step / HZ:.0f}s  空队列 {underruns}  预热保持 {warmup_holds}", flush=True)
    finally:
        stop.set()
        worker.join(timeout=30.0)
        front.close()
        astra.close()

    if error:
        raise error[0]
    if worker.is_alive():
        raise RuntimeError("推理线程没有退出")

    steady = tick_dts[1:]
    steps = np.abs(np.diff(commanded[:, :6], axis=0)).max(axis=1)
    demo = data["action"][:, :6]
    valid = np.isfinite(commanded[:, 0])
    mae = float(np.abs(commanded[valid, :6] - demo[valid]).mean())
    covered = sum(1 for dt in infer_dts if dt <= budget)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "commanded.npy", commanded)
    print()
    print(f"预热保持 {warmup_holds} 拍（第一块出来之前）")
    print(f"第一块之后的空队列 {underruns} / {n}")
    print(
        f"下发间隔 中位 {np.median(steady)*1000:.1f} ms  "
        f"p95 {np.quantile(steady, 0.95)*1000:.1f} ms  最大 {np.max(steady)*1000:.1f} ms"
    )
    print(_stats("推理", infer_dts))
    print(f"预算 {budget:.2f}s 内完成的推理 {covered}/{len(infer_dts)}")
    if ages:
        print(
            f"画面落后 中位 {np.median(ages):.0f} 步 ({np.median(ages) * DT:.2f}s)  "
            f"到达时队列还剩 中位 {np.median(queued_before):.0f} 步  "
            f"重叠平均 中位 {np.median(overlaps):.0f} 步"
        )
        print(
            f"若改按观测时刻对齐，未来只剩 中位 {np.median(aligned_left):.0f} 步 "
            f"({np.median(aligned_left) * DT:.2f}s)，盖不住下一次 {budget:.2f}s 的推理"
        )
    print(
        f"下发步长 中位 {np.median(steps):.2f}°  p95 {np.quantile(steps, 0.95):.2f}°  "
        f"最大 {steps.max():.2f}°  >0.12° {int(np.sum(steps > 0.12))}/{len(steps)}"
    )
    print(f"与该条演示动作的关节平均差 {mae:.2f}°")
    ok = underruns == 0 and covered == len(infer_dts) and np.median(steady) < 0.06
    print("节拍盖住了" if ok else "节拍没有盖住")


if __name__ == "__main__":
    main()
