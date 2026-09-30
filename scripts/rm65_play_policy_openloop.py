#!/usr/bin/env python3
"""Play one offline open-loop policy trajectory on the real RM65.

No closed-loop inference while the arm is moving. Episode frames are fed to the
checkpoint with overlapping windows, the absolute joint commands are blended and
saved, the arm movej's to that episode's first encoder pose, then the saved
commands are streamed at dataset fps.

Does not start policy_server / play_action_rm65. Those would add a second
command path and the 0.12° rate limit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "smolvla_rm65_stack_30ep_s1000"
DS = ROOT / "datasets/cleaned/rm65_stacking_blocks"
OUT = ROOT / "outputs/diag/stack_30ep_offline"
DEG2RAD = math.pi / 180.0


# Re-infer every 4 frames and blend the 16-step windows. A new window fades in,
# and the tail fades out, so a chunk boundary cannot jump in one frame.
# exp decay still trusts the near part of a window more than its far tail.
STRIDE = 4
DECAY = 0.05
# Demo per-frame joint steps sit near 1°. Anything larger is spread over later frames.
SMOOTH_MAX_STEP_DEG = 1.0


def traj_path(episode: int, stride: int = STRIDE) -> Path:
    return OUT / f"ep{episode:03d}_s{stride}_openloop_smooth.npy"


def ensemble_weights(horizon: int, stride: int, decay: float = DECAY) -> np.ndarray:
    if horizon < 1 or stride < 1:
        raise ValueError(f"horizon={horizon} stride={stride}")
    k = np.arange(horizon, dtype=np.float64)
    fade_in = np.minimum(1.0, (k + 1.0) / float(stride))
    fade_out = np.minimum(1.0, (horizon - k) / float(stride))
    return np.exp(-float(decay) * k) * fade_in * fade_out


def blend_predictions(
    chunks: list[tuple[int, np.ndarray]],
    n: int,
    weights: np.ndarray,
) -> np.ndarray:
    acc = np.zeros((n, 7), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for t0, chunk in chunks:
        take = min(len(weights), chunk.shape[0], n - t0)
        if take <= 0:
            continue
        w = weights[:take]
        acc[t0 : t0 + take] += chunk[:take].astype(np.float64) * w[:, None]
        wsum[t0 : t0 + take] += w
    missing = np.flatnonzero(wsum <= 0)
    if missing.size:
        raise RuntimeError(f"frames with no prediction: {missing[:8].tolist()}")
    return (acc / wsum[:, None]).astype(np.float32)


def grip_hysteresis(values: np.ndarray, low: float = 0.35, high: float = 0.65) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float32)
    state: float | None = None
    for i, value in enumerate(values):
        v = float(value)
        if state is None:
            state = 1.0 if v >= 0.5 else 0.0
        elif state >= 0.5:
            if v <= low:
                state = 0.0
        elif v >= high:
            state = 1.0
        out[i] = state
    return out


def spread_large_steps(joints: np.ndarray, max_step: float = SMOOTH_MAX_STEP_DEG) -> np.ndarray:
    """Follow the blended joints, but never faster than max_step per frame.

    A leftover seam becomes a short ramp. Once the blended curve slows down,
    the command catches the original pose again.
    """
    out = np.empty((len(joints), joints.shape[1]), dtype=np.float64)
    out[0] = joints[0]
    cap = float(max_step)
    for i in range(1, len(joints)):
        delta = np.asarray(joints[i], dtype=np.float64) - out[i - 1]
        out[i] = out[i - 1] + np.clip(delta, -cap, cap)
    return out.astype(np.float32)


def _step_stats(joints: np.ndarray) -> str:
    steps = np.abs(np.diff(joints, axis=0)).max(axis=1)
    if steps.size == 0:
        return "无步长"
    return (
        f"中位 {float(np.median(steps)):.2f}°  "
        f"p95 {float(np.quantile(steps, 0.95)):.2f}°  "
        f"最大 {float(steps.max()):.2f}°  "
        f">1° {int(np.sum(steps > 1.0))}/{len(steps)}"
    )


def load_episode(episode: int) -> dict:
    info = json.loads((DS / "meta/info.json").read_text())
    tasks = {}
    for line in (DS / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    pq = DS / info["data_path"].format(episode_chunk=0, episode_index=episode)
    import pandas as pd

    df = pd.read_parquet(pq)
    return {
        "state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        "action": np.stack(df["action"].to_numpy()).astype(np.float32),
        "task_index": df["task_index"].to_numpy().astype(np.int64),
        "tasks": tasks,
        "info": info,
        "n": len(df),
        "fps": float(info["fps"]),
    }


def read_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    import av

    want = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    container = av.open(str(path))
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            out[i] = frame.to_ndarray(format="rgb24")
        if len(out) == len(want):
            break
    container.close()
    missing = want - set(out)
    if missing:
        raise FileNotFoundError(f"{path} missing frames {sorted(missing)[:8]}")
    return out


def generate(episode: int, stride: int = STRIDE) -> np.ndarray:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.scripts.server.rm65_ckpt import load_dataset_stats_from_ckpt, trim_stats_to_config

    data = load_episode(episode)
    info = data["info"]
    n = data["n"]
    config = PreTrainedConfig.from_pretrained(str(CKPT))
    if not getattr(config, "use_relative_actions", False):
        raise RuntimeError("checkpoint use_relative_actions is false; refuse to invent q+dq")
    stats = trim_stats_to_config(load_dataset_stats_from_ckpt(CKPT), config)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT), config=config, dataset_stats=stats or None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    horizon = int(policy.config.n_action_steps)
    if stride > horizon:
        raise RuntimeError(f"stride {stride} is longer than the action horizon {horizon}")
    weights = ensemble_weights(horizon, stride)
    starts = list(range(0, n, stride))
    front = read_video_frames(
        DS / info["video_path"].format(episode_chunk=0, episode_index=episode, video_key="observation.images.front"),
        starts,
    )
    astra = read_video_frames(
        DS / info["video_path"].format(episode_chunk=0, episode_index=episode, video_key="observation.images.astra"),
        starts,
    )
    chunks: list[tuple[int, np.ndarray]] = []
    print(
        f"开环生成 ep{episode:03d}  {n} 帧  每 {stride} 帧推理一次  "
        f"重叠 {horizon} 步  device={device}  此阶段机械臂不动",
        flush=True,
    )
    with torch.no_grad():
        for fi, t0 in enumerate(starts):
            policy.reset()
            state_np = data["state"][t0, :7].copy()
            task = data["tasks"][int(data["task_index"][t0])]
            batch = {
                "task": task,
                "observation.state": torch.from_numpy(state_np).unsqueeze(0).to(device),
                "observation.images.camera1": _rgb(front[t0], torch).to(device),
                "observation.images.camera2": _rgb(astra[t0], torch).to(device),
            }
            chunk = policy.predict_action_chunk(batch)[0].detach().float().cpu().numpy()
            chunks.append((t0, chunk[:horizon]))
            if fi % 10 == 0 or fi + 1 == len(starts):
                print(
                    f"  {fi + 1}/{len(starts)} f={t0} grip0={float(chunk[0, 6]):.0f}",
                    flush=True,
                )
            del batch
            if device.type == "cuda" and fi % 10 == 0:
                torch.cuda.empty_cache()
    blended = blend_predictions(chunks, n, weights)
    print(f"重叠平均后  {_step_stats(blended[:, :6])}")
    pred = blended.copy()
    pred[:, :6] = spread_large_steps(blended[:, :6])
    pred[:, 6] = grip_hysteresis(blended[:, 6])
    if not np.isfinite(pred).all():
        raise RuntimeError("open-loop trajectory has NaN")
    print(f"大步摊开后  {_step_stats(pred[:, :6])}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = traj_path(episode, stride)
    np.save(path, pred)
    demo = data["action"][:, :6]
    mae = float(np.abs(pred[:, :6] - demo).mean())
    print(f"已保存 {path}")
    print(f"与该条演示动作的关节平均差 {mae:.2f}°   时长 {n / data['fps']:.1f}s @ {data['fps']:.0f}Hz")
    return pred


def _rgb(img: np.ndarray, torch):
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0


class _LiveCameras:
    """Grab both real cameras on side threads and store one JPEG per playback frame.

    The 20 Hz command loop only copies the latest frame into a queue. Encoding
    happens off that loop, so a slow disk does not stall movej_canfd.
    """

    def __init__(self, front_dev: int, astra_dev: int, width: int, height: int, live: Path):
        self.front_dev = front_dev
        self.astra_dev = astra_dev
        self.width = width
        self.height = height
        self.live = live
        self.front_dir = live / "front"
        self.astra_dir = live / "astra"
        self._q: queue.Queue = queue.Queue(maxsize=120)
        self.saved = 0
        self.dropped = 0
        self.snaps = 0
        self.repeat_front = 0
        self.repeat_astra = 0
        self.miss_front = 0
        self.miss_astra = 0
        self._ticks: list[dict] = []
        self._prev_front_seq = -1
        self._prev_astra_seq = -1
        self._cams: list[_Grabber] = []
        self._writer: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        import cv2

        cv2.setNumThreads(1)
        self.front_dir.mkdir(parents=True, exist_ok=True)
        self.astra_dir.mkdir(parents=True, exist_ok=True)
        for old in list(self.front_dir.glob("frame_*.jpg")) + list(self.astra_dir.glob("frame_*.jpg")):
            old.unlink()
        opened: list[_Grabber] = []
        try:
            for dev in (self.front_dev, self.astra_dev):
                opened.append(_Grabber(dev, self.width, self.height))
        except Exception:
            for cam in opened:
                cam.close()
            raise
        self._cams = opened
        self._writer = threading.Thread(target=self._write_loop, name="live_jpeg", daemon=True)
        self._writer.start()

    def wait_ready(self, timeout_s: float = 3.0) -> None:
        for cam in self._cams:
            cam.wait_first(timeout_s)
        shapes = [cam.shape() for cam in self._cams]
        print(
            f"相机已出图  front /dev/video{self.front_dev} {shapes[0]}  "
            f"astra /dev/video{self.astra_dev} {shapes[1]}"
        )

    def snap(self, index: int) -> None:
        self.snaps += 1
        front, front_seq = self._cams[0].snapshot()
        astra, astra_seq = self._cams[1].snapshot()
        if front is None:
            self.miss_front += 1
            front = np.zeros((self.height, self.width, 3), np.uint8)
        elif front_seq == self._prev_front_seq:
            self.repeat_front += 1
        if astra is None:
            self.miss_astra += 1
            astra = np.zeros((self.height, self.width, 3), np.uint8)
        elif astra_seq == self._prev_astra_seq:
            self.repeat_astra += 1
        self._prev_front_seq = front_seq
        self._prev_astra_seq = astra_seq
        self._ticks.append({"i": index, "front_seq": front_seq, "astra_seq": astra_seq})
        self._submit(self.front_dir / f"frame_{index:06d}.jpg", front)
        self._submit(self.astra_dir / f"frame_{index:06d}.jpg", astra)

    def _submit(self, path: Path, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait((str(path), frame))
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        import cv2

        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                path, frame = item
                if cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                    self.saved += 1
            finally:
                self._q.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cam in self._cams:
            cam.close()
        if self._writer is not None:
            self._q.put(None)
            self._q.join()
            self._writer.join(timeout=5.0)

    def finish(self, episode: int, fps: float, front_video: Path, astra_video: Path) -> None:
        compare = self.live / "compare.mp4"
        meta = {
            "episode": episode,
            "fps": fps,
            "width": self.width,
            "height": self.height,
            "front_device": self.front_dev,
            "astra_device": self.astra_dev,
            "snaps": self.snaps,
            "jpegs_saved": self.saved,
            "jpeg_queue_drops": self.dropped,
            "repeat_front": self.repeat_front,
            "repeat_astra": self.repeat_astra,
            "miss_front": self.miss_front,
            "miss_astra": self.miss_astra,
            "alignment": (
                "live frame i is the camera image at the moment command i was published; "
                "dataset frame i is the recorded episode image at the same index"
            ),
            "compare_layout": "top dataset-front | live-front; bottom dataset-astra | live-astra",
            "front_video": str(front_video),
            "astra_video": str(astra_video),
            "compare_video": str(compare),
        }
        (self.live / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (self.live / "ticks.json").write_text(json.dumps(self._ticks) + "\n")
        n = _write_compare_video(
            compare, front_video, astra_video, self.front_dir, self.astra_dir, self.snaps, fps, self.width, self.height
        )
        print(f"真实画面 {self.front_dir}  和  {self.astra_dir}")
        print(f"并排对比 {compare}  （{n} 帧，上：前相机 数据|真实，下：Astra 数据|真实）")
        if self.dropped or self.miss_front or self.miss_astra:
            print(
                f"警告: 丢帧 {self.dropped}  前相机空帧 {self.miss_front}  Astra空帧 {self.miss_astra}",
                file=sys.stderr,
            )


class _Grabber:
    def __init__(self, index: int, width: int, height: int):
        import cv2

        self.index = index
        self.width = width
        self.height = height
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开 /dev/video{index}，机械臂不会动")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name=f"cam{index}", daemon=True)
        self._th.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = self._cv2.resize(frame, (self.width, self.height))
            with self._lock:
                self._frame = frame
                self._seq += 1

    def wait_first(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"/dev/video{self.index} 在 {timeout_s:.0f}s 内没有画面，机械臂不会动")

    def shape(self) -> tuple[int, int]:
        with self._lock:
            if self._frame is None:
                return (0, 0)
            return (int(self._frame.shape[1]), int(self._frame.shape[0]))

    def snapshot(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq

    def close(self) -> None:
        self._stop.set()
        self._th.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _write_compare_video(
    path: Path,
    front_video: Path,
    astra_video: Path,
    front_dir: Path,
    astra_dir: Path,
    n: int,
    fps: float,
    width: int,
    height: int,
) -> int:
    import av
    import cv2

    def rgb_frames(video: Path):
        container = av.open(str(video))
        try:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
        finally:
            container.close()

    blank = np.zeros((height, width, 3), np.uint8)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width * 2, height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"写不了对比视频 {path}")
    written = 0
    try:
        paired = zip(rgb_frames(front_video), rgb_frames(astra_video))
        for i, (front_rgb, astra_rgb) in enumerate(paired):
            if i >= n:
                break
            live_front = cv2.imread(str(front_dir / f"frame_{i:06d}.jpg"))
            live_astra = cv2.imread(str(astra_dir / f"frame_{i:06d}.jpg"))
            if live_front is None:
                live_front = blank
            if live_astra is None:
                live_astra = blank
            if live_front.shape[1] != width or live_front.shape[0] != height:
                live_front = cv2.resize(live_front, (width, height))
            if live_astra.shape[1] != width or live_astra.shape[0] != height:
                live_astra = cv2.resize(live_astra, (width, height))
            data_front = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
            data_astra = cv2.cvtColor(astra_rgb, cv2.COLOR_RGB2BGR)
            if data_front.shape[1] != width or data_front.shape[0] != height:
                data_front = cv2.resize(data_front, (width, height))
            if data_astra.shape[1] != width or data_astra.shape[0] != height:
                data_astra = cv2.resize(data_astra, (width, height))
            _stamp(data_front, "dataset front")
            _stamp(live_front, "live front")
            _stamp(data_astra, "dataset astra")
            _stamp(live_astra, "live astra")
            canvas = np.concatenate(
                [
                    np.concatenate([data_front, live_front], axis=1),
                    np.concatenate([data_astra, live_astra], axis=1),
                ],
                axis=0,
            )
            writer.write(canvas)
            written += 1
    finally:
        writer.release()
    return written


def _stamp(img: np.ndarray, text: str) -> None:
    import cv2

    cv2.rectangle(img, (0, 0), (210, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _refuse_other_commanders() -> None:
    import subprocess

    bad = ("play_action_rm65.py", "policy_server.py", "robot_client.py", "record_state_test.py")
    out = subprocess.check_output(["ps", "-eo", "args"], text=True, errors="replace")
    hit = [name for name in bad if name in out]
    if hit:
        raise RuntimeError(
            "仍有进程会抢关节或夹爪 SDK: "
            + ", ".join(hit)
            + "。先停掉推理脚本再回放。"
        )


def play(episode: int, pred: np.ndarray, max_step_deg: float) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rm_ros_interfaces.msg import Jointpos

    sys.path.insert(0, str(ROOT / "scripts"))
    from rm65_ready import movej_ready

    data = load_episode(episode)
    fps = data["fps"]
    start = data["state"][0, :6].astype(np.float64)
    grip0 = float(data["state"][0, 6])
    print(
        f"回放 ep{episode:03d}  {len(pred)} 帧  {len(pred) / fps:.1f}s  "
        f"{_step_stats(pred[:, :6])}  安全限幅 {max_step_deg:.1f}°"
    )
    print(f"先 movej 到该条数据第 0 帧: {[round(float(v), 2) for v in start]}")
    front_shape = data["info"]["features"]["observation.images.front"]["shape"]
    cam_h, cam_w = int(front_shape[0]), int(front_shape[1])
    front_dev = int(os.environ.get("FRONT_DEV", "0"))
    astra_dev = int(os.environ.get("ASTRA_DEV", "2"))
    video_path = data["info"]["video_path"]
    front_video = DS / video_path.format(
        episode_chunk=0, episode_index=episode, video_key="observation.images.front"
    )
    astra_video = DS / video_path.format(
        episode_chunk=0, episode_index=episode, video_key="observation.images.astra"
    )
    print(f"打开真实相机 /dev/video{front_dev} 和 /dev/video{astra_dev}。打不开则不会动臂。")
    recorder = _LiveCameras(front_dev, astra_dev, cam_w, cam_h, OUT / f"ep{episode:03d}_live")
    recorder.start()
    interrupted = False
    clips = 0
    try:
        recorder.wait_ready()
        print(f"回放过程按帧保存到 {recorder.live}。Ctrl+C 停止。不启动策略服务。")
        print("3 秒后开始运动。")
        time.sleep(3.0)

        _refuse_other_commanders()
        rclpy.init()
        try:
            movej_ready(speed=5, joints_deg=start.tolist(), pose_path=Path(f"episode_{episode:03d}_frame0"))
        except Exception:
            if rclpy.ok():
                rclpy.shutdown()
            raise

        from sendrobot.send_robot.dahuan_gripper import DahuanGripperModbus
        from sendrobot.send_robot.rm_ik import RM65Kinematics

        class _Log:
            def info(self, msg: str) -> None:
                print(msg)

            def warning(self, msg: str) -> None:
                print("警告:", msg, file=sys.stderr)

        kin = RM65Kinematics(ip="192.168.73.18", port=8080)
        kin.connect()
        grip = DahuanGripperModbus(kin.robot)
        grip.setup(_Log())
        grip.set_position_normalized(grip0, min_delta=0.0)
        print(f"夹爪设到数据起点 norm={grip0:.2f}")

        node = Node("rm65_openloop_play")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        pub = node.create_publisher(Jointpos, "/rm_driver/movej_canfd_cmd", qos)
        deadline = time.time() + 3.0
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if pub.get_subscription_count() == 0:
            kin.disconnect()
            node.destroy_node()
            rclpy.shutdown()
            raise RuntimeError("rm_driver 未订阅 movej_canfd_cmd")

        last = start.copy()
        dt = 1.0 / fps
        t_next = time.perf_counter()
        try:
            for i, row in enumerate(pred):
                cmd = row[:6].astype(np.float64).copy()
                delta = cmd - last
                over = np.abs(delta) > max_step_deg
                if np.any(over):
                    clips += 1
                    delta = np.clip(delta, -max_step_deg, max_step_deg)
                    cmd = last + delta
                msg = Jointpos()
                msg.joint = [float(v) * DEG2RAD for v in cmd]
                msg.follow = True
                msg.expand = 0.0
                msg.dof = 6
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                recorder.snap(i)
                g = float(row[6])
                if abs(g - grip0) >= 0.5:
                    grip.set_position_normalized(1.0 if g >= 0.5 else 0.0, min_delta=0.0)
                    grip0 = 1.0 if g >= 0.5 else 0.0
                last = cmd
                t_next += dt
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.perf_counter()
                if i % int(fps) == 0:
                    print(f"t={i / fps:.0f}s  {[round(float(v), 1) for v in cmd]}  grip={grip0:.0f}", flush=True)
        finally:
            kin.disconnect()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    except KeyboardInterrupt:
        interrupted = True
        print("回放中断，正在把已录画面写完")
    finally:
        recorder.close()
    print(f"回放结束  超限步数 {clips}/{len(pred)}")
    if recorder.snaps:
        recorder.finish(episode, fps, front_video, astra_video)
    if interrupted:
        raise SystemExit(130)


def main() -> None:
    p = argparse.ArgumentParser(description="Replay one offline open-loop policy trajectory")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=STRIDE, help="每隔多少帧重新推理一次")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--replay-only", action="store_true", help="使用已保存的 npy，不再推理")
    p.add_argument(
        "--max-step-deg",
        type=float,
        default=1.5,
        help="回放安全限幅。平滑后单步应不超过 1°，这项只拦住意外大步",
    )
    args = p.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride 必须 >= 1")
    path = traj_path(args.episode, args.stride)
    if args.replay_only:
        if not path.is_file():
            raise SystemExit(f"没有已保存轨迹: {path}")
        pred = np.load(path)
    elif path.is_file() and not args.generate_only:
        print(f"使用已有轨迹 {path}  （要重算请先删掉这个文件）")
        pred = np.load(path)
    else:
        print("正在离线开环生成，此阶段机械臂不动")
        pred = generate(args.episode, args.stride)
    if args.generate_only:
        return
    play(args.episode, pred, args.max_step_deg)


if __name__ == "__main__":
    main()
