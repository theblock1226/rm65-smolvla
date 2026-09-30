#!/usr/bin/env python3
"""
RM65 关节空间数据集录制（手柄 movep 遥操 + 双相机同步）。

控制：笛卡尔 movep_canfd（与 rm65_movep_teleop 相同）
落盘：LeRobot v2.1
  - observation.state = SDK 实测关节(度) + 夹爪(0~1)
  - action = 遥操积分位姿 IK 目标关节 + 夹爪指令
  - observation.images.front = 大白 USB（/dev/video0）
  - observation.images.astra = Astra Pro USB（/dev/video2）

USB 设备录制时独占，独立网页预览必须停掉。本脚本用已取帧在
8090/8093 继续出预览（大白/Astra），无需再占设备。

按键（G7 Pro）：
  A  双路相机同步开始写入当前 episode
  B  同步结束并保存 episode
  X  丢弃当前 episode 缓冲（重录）
  LB 夹爪开 / RB 夹爪合
  Ctrl+C 退出且不保存当前未完成的 episode

用法：
  /home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_record.sh
  /home/sdznkj/zkx_ws/sendlerobot/scripts/rm65_run_record.sh --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# 保证可 import 同目录遥操模块与 sendrobot / lerobot
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

from rm65_ready import movej_ready  # noqa: E402
from rm65_stick import _env_bool, _make_stick  # noqa: E402

MOTOR_KEYS = [f"left_joint_{i}.pos" for i in range(1, 7)] + ["left_gripper.pos"]
DEFAULT_FRONT_KEY = "front"
DEFAULT_ASTRA_KEY = "astra"
DEFAULT_ASTRA_DEVICE = "/dev/video2"
_PREVIEW_LOCK = threading.Lock()
_PREVIEW_FRAMES: dict[str, np.ndarray] = {}


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def _upgrade_empty_dataset_cameras(
    root: Path,
    expected_keys: list[str],
    cam_shape: tuple[int, int, int],
    use_video: bool,
) -> list[str]:
    """0 episode 空壳可补上缺失的图像 feature；已有数据则不改。"""
    info_path = root / "meta" / "info.json"
    info = _load_info(root)
    have = _camera_keys_from_info(root)
    if sorted(have) == sorted(expected_keys):
        return have
    if int(info.get("total_episodes", 0) or 0) > 0:
        return have
    extra = set(have) - set(expected_keys)
    changed = False
    if extra:
        # 0 episode 空壳：丢掉不再使用的图像 key（例如 astra → mono）
        for key in extra:
            info["features"].pop(f"observation.images.{key}", None)
        print(f"空数据集已移除旧相机 feature: {sorted(extra)}", flush=True)
        changed = True
    dtype = "video" if use_video else "image"
    added: list[str] = []
    for key in expected_keys:
        full = f"observation.images.{key}"
        if full in info["features"]:
            continue
        info["features"][full] = {
            "dtype": dtype,
            "shape": list(cam_shape),
            "names": ["height", "width", "channels"],
        }
        added.append(key)
    if added:
        print(f"空数据集已补相机 feature: {added}", flush=True)
        changed = True
    if changed:
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
    return _camera_keys_from_info(root)


def _start_mjpeg_preview(port: int, key: str, title: str) -> bool:
    """用录制已取 RGB 帧出预览；不另开 V4L2。JPEG 较小以免网页积压。"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = (
                    "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
                    f"<h3>{title}（录制预览）</h3>"
                    "<img src='/stream/rgb' style='max-width:100%'/>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if not self.path.startswith("/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with _PREVIEW_LOCK:
                        fr = _PREVIEW_FRAMES.get(key)
                        img = None if fr is None else fr.copy()
                    if img is None:
                        time.sleep(0.05)
                        continue
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(
                        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 55]
                    )
                    if not ok:
                        continue
                    payload = (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                    self.wfile.write(payload)
                    time.sleep(0.07)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError as exc:
        print(f"预览端口 {port} 无法绑定（{title}）: {exc}", file=sys.stderr)
        return False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"预览: http://0.0.0.0:{port}/  ({title})")
    return True


def _camera_keys_from_info(root: Path) -> list[str]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return [DEFAULT_FRONT_KEY, DEFAULT_ASTRA_KEY]
    info = _load_info(root)
    keys = [
        key.removeprefix("observation.images.")
        for key in info.get("features", {})
        if key.startswith("observation.images.")
    ]
    return keys or [DEFAULT_FRONT_KEY, DEFAULT_ASTRA_KEY]


def _ensure_local_meta_stubs(root: Path) -> None:
    """空壳数据集常只有 info.json；补齐空 jsonl，避免 LeRobot 去拉 HuggingFace。"""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        path = meta / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


def _open_local_recording_dataset(
    repo_id: str,
    root: Path,
    *,
    image_writer_threads: int = 4,
):
    """纯本地打开录制数据集，禁止访问 HuggingFace Hub（否则无网/假 repo 会卡住）。"""
    from lerobot.common.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        LeRobotDataset,
        LeRobotDatasetMetadata,
        get_safe_default_codec,
    )

    # 强制离线：续录路径绝不访问 Hub
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    _ensure_local_meta_stubs(root)
    info = _load_info(root)
    total_episodes = int(info.get("total_episodes", 0) or 0)

    if total_episodes == 0:
        # 0 episode 时没有 data/*.parquet，LeRobotDataset() 会误走 Hub 下载
        print("本地打开空数据集（跳过 HuggingFace）…", flush=True)
        meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
        meta.repo_id = repo_id
        meta.root = root
        meta.revision = CODEBASE_VERSION
        meta.load_metadata()

        obj = LeRobotDataset.__new__(LeRobotDataset)
        obj.meta = meta
        obj.repo_id = repo_id
        obj.root = root
        obj.revision = None
        obj.tolerance_s = 1e-4
        obj.image_writer = None
        obj.episode_buffer = None
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = get_safe_default_codec()
        obj.start_image_writer(num_processes=0, num_threads=image_writer_threads)
        obj.episode_buffer = obj.create_episode_buffer()
        return obj

    print(f"本地续录已有 {total_episodes} episodes（跳过 HuggingFace）…", flush=True)
    dataset = LeRobotDataset(repo_id, root=root)
    dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)
    if dataset.episode_buffer is None:
        dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def _dataset_features(
    cameras: dict[str, tuple[int, int, int]],
    use_video: bool,
) -> dict:
    from lerobot.common.datasets.utils import hw_to_dataset_features

    obs_hw: dict = {k: float for k in MOTOR_KEYS}
    obs_hw.update(cameras)
    act_hw = {k: float for k in MOTOR_KEYS}
    return {
        **hw_to_dataset_features(act_hw, "action", use_video),
        **hw_to_dataset_features(obs_hw, "observation", use_video),
    }


def _resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    # cv2.resize 期望 HxW；输入为 RGB
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _open_usb_camera(
    device: str,
    fps: int | None,
    width: int,
    height: int,
):
    """USB 相机。fps=None 时不强制帧率（Astra Pro RGB 常为 15）。"""
    from lerobot.common.cameras.configs import ColorMode
    from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = OpenCVCameraConfig(
        index_or_path=device,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        warmup_s=1,
    )
    cam = OpenCVCamera(cfg)
    cam.connect()
    return cam


class RtspCamera:
    """后台线程拉 RTSP，接口对齐 OpenCVCamera.async_read（返回 RGB）。"""

    def __init__(self, url: str):
        self.url = url
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._err: str | None = None
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        t0 = time.time()
        while time.time() - t0 < 8.0:
            with self._lock:
                fr = None if self._frame is None else self._frame
                err = self._err
            if fr is not None:
                self.height, self.width = int(fr.shape[0]), int(fr.shape[1])
                return
            if err:
                raise RuntimeError(err)
            time.sleep(0.05)
        raise RuntimeError(f"RTSP 无画面: {self.url.split('@')[-1]}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self._err = f"RTSP 打开失败: {self.url.split('@')[-1]}"
                time.sleep(1.0)
                continue
            n = 0
            t0 = time.time()
            while not self._stop.is_set():
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = rgb
                n += 1
                if n == 40:
                    dt = time.time() - t0
                    if dt > 0:
                        self.fps = n / dt
            cap.release()
            if not self._stop.is_set():
                time.sleep(0.3)

    def latest_rgb(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def async_read(self, timeout_ms: float = 500.0) -> np.ndarray:
        # 有帧立刻返回，避免遥操主循环被 RTSP 卡住。
        fr = self.latest_rgb()
        if fr is not None:
            return fr
        deadline = time.time() + max(0.0, timeout_ms) / 1000.0
        while time.time() < deadline:
            fr = self.latest_rgb()
            if fr is not None:
                return fr
            time.sleep(0.01)
        raise TimeoutError("RTSP async_read timeout")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _latest_rgb(cam) -> np.ndarray | None:
    """取后台线程已有最新帧，不阻塞等待下一帧（否则 USB 15fps 会把 50Hz 遥操卡成一顿一顿）。"""
    if isinstance(cam, RtspCamera):
        return cam.latest_rgb()
    lock = getattr(cam, "frame_lock", None)
    if lock is not None:
        with lock:
            fr = getattr(cam, "latest_frame", None)
            event = getattr(cam, "new_frame_event", None)
            if event is not None:
                event.clear()
            return None if fr is None else np.ascontiguousarray(fr)
    try:
        return cam.async_read(timeout_ms=0.0)
    except Exception:
        return None


def _kick_async(cam, name: str, timeout_ms: float = 3000.0) -> None:
    try:
        cam.async_read(timeout_ms=timeout_ms)
    except Exception as exc:
        raise RuntimeError(f"相机 {name} 预热读帧失败: {exc}") from exc


class JointReader:
    """关节角缓存。SDK 查询放后台线程，禁止在 50Hz movep 循环里同步打 TCP。"""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = int(port)
        self._kin = None
        self._last: list[float] | None = None
        self._ros_last: list[float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def on_ros_joint_state(self, names, positions) -> None:
        name_to_i = {f"joint{i}": i - 1 for i in range(1, 7)}
        joints = [0.0] * 6
        ok = False
        for name, pos in zip(names, positions):
            if name in name_to_i:
                joints[name_to_i[name]] = math.degrees(float(pos))
                ok = True
        if ok and max(abs(v) for v in joints) > 0.01:
            with self._lock:
                self._ros_last = joints

    def start_sdk_poll(self, hz: float = 10.0) -> None:
        if self._thread is not None:
            return
        period = 1.0 / max(1.0, float(hz))

        def _loop() -> None:
            while not self._stop.wait(period):
                kin = self._kin
                if kin is None:
                    continue
                try:
                    q = [float(v) for v in kin.get_current_joints_deg()[:6]]
                except Exception:
                    continue
                with self._lock:
                    self._last = q

        self._thread = threading.Thread(target=_loop, daemon=True, name="joint_sdk_poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def read_deg(self) -> list[float]:
        with self._lock:
            if self._ros_last is not None:
                return list(self._ros_last)
            if self._last is not None:
                return list(self._last)
        raise RuntimeError("无法读取关节角")


def _motor_dict(joints_deg: list[float], gripper_norm: float) -> dict[str, float]:
    out = {f"left_joint_{i}.pos": float(joints_deg[i - 1]) for i in range(1, 7)}
    out["left_gripper.pos"] = float(gripper_norm)
    return out


def run_record(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rm_ros_interfaces.msg import Cartepos
    from sensor_msgs.msg import JointState

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import build_dataset_frame
    from rm65_dataset_finalize import finalize_dataset

    root = Path(args.root).expanduser().resolve()
    repo_id = args.repo_id
    task = args.task
    fps = int(args.fps)
    control_hz = float(args.rate)
    episode_time_s = float(args.episode_time_s)
    num_episodes = int(args.num_episodes)
    front_key = args.front_key or DEFAULT_FRONT_KEY
    astra_key = args.astra_key or DEFAULT_ASTRA_KEY
    target_h = int(args.cam_height)
    target_w = int(args.cam_width)

    cam_shape = (target_h, target_w, 3)
    expected_keys = [front_key, astra_key]

    # --- 数据集（先于相机：避免 Hub 卡住后相机线程无法干净退出）---
    front_cam = None
    astra_cam = None
    if args.resume and (root / "meta" / "info.json").is_file():
        info_keys = _upgrade_empty_dataset_cameras(
            root, expected_keys, cam_shape, args.video
        )
        if sorted(info_keys) != sorted(expected_keys):
            raise SystemExit(
                f"续录相机 key 不匹配。\n"
                f"  数据集已有: {info_keys}\n"
                f"  当前脚本:   {expected_keys}\n"
                f"旧单相机数据集请换新 --root 新建，或改 key 与 features 对齐。"
            )
        dataset = _open_local_recording_dataset(repo_id, root)
        print(f"续录数据集: {root} 已有 episodes={dataset.num_episodes} cams={info_keys}")
    else:
        features = _dataset_features(
            {
                front_key: cam_shape,
                astra_key: cam_shape,
            },
            use_video=args.video,
        )
        if root.exists() and any(root.iterdir()):
            # 空壳（仅 meta）自动按续录打开，避免反复“目录非空”
            if (root / "meta" / "info.json").is_file():
                info_keys = _upgrade_empty_dataset_cameras(
                    root, expected_keys, cam_shape, args.video
                )
                if sorted(info_keys) == sorted(expected_keys):
                    print("检测到已有本地空/半成品数据集，自动续录（不连 Hub）…", flush=True)
                    dataset = _open_local_recording_dataset(repo_id, root)
                    print(
                        f"续录数据集: {root} 已有 episodes={dataset.num_episodes} cams={info_keys}"
                    )
                else:
                    raise SystemExit(
                        f"数据集目录非空且相机 key 不匹配: {root}\n"
                        f"  已有: {info_keys}\n  需要: {expected_keys}"
                    )
            else:
                raise SystemExit(
                    f"数据集目录非空: {root}\n"
                    f"请换 --root，或加 --resume 续录，或先清空该目录。"
                )
        else:
            print("新建本地数据集…", flush=True)
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=root,
                robot_type="rm65_movep",
                features=features,
                use_videos=args.video,
                image_writer_processes=0,
                image_writer_threads=4,
            )
            print(f"新建数据集: {root} cams={expected_keys}")

    # --- 双相机：启动时同时连接并预热后台取流线程，保证 A/B 时同步启停写盘 ---
    try:
        front_cam = _open_usb_camera(args.camera, fps, target_w, target_h)
        print(
            f"相机 front: {args.camera} key={front_key} "
            f"{front_cam.width}x{front_cam.height}@{front_cam.fps}"
        )
        astra_rtsp = (args.astra_rtsp or "").strip()
        if astra_rtsp:
            astra_cam = RtspCamera(astra_rtsp)
            astra_cam.connect()
            print(
                f"相机 cam2/{astra_key}: RTSP {astra_rtsp.split('@')[-1]} "
                f"{astra_cam.width}x{astra_cam.height}@{astra_cam.fps:.1f}"
            )
        else:
            astra_cam = _open_usb_camera(args.astra_camera, None, target_w, target_h)
            print(
                f"相机 astra: {args.astra_camera} key={astra_key} "
                f"{astra_cam.width}x{astra_cam.height}@{astra_cam.fps}"
            )
        _kick_async(front_cam, front_key)
        _kick_async(astra_cam, astra_key)
    except Exception:
        for cam in (front_cam, astra_cam):
            if cam is not None:
                try:
                    cam.disconnect()
                except Exception:
                    pass
        raise

    if args.preview:
        front_preview_port = int(args.front_preview_port)
        astra_preview_port = int(args.astra_preview_port)
        _start_mjpeg_preview(front_preview_port, front_key, "DaBai / front")
        _start_mjpeg_preview(astra_preview_port, astra_key, args.astra_preview_title)

    # --- ROS / 就绪 ---
    if args.go_ready:
        if not rclpy.ok():
            rclpy.init()
        try:
            movej_ready(speed=args.ready_speed)
        finally:
            pass
        time.sleep(0.5)

    if not rclpy.ok():
        rclpy.init()

    joint_reader = JointReader(args.ip, args.port)
    from sendrobot.send_robot.dahuan_gripper import DahuanGripperModbus
    from sendrobot.send_robot.rm_ik import RM65Kinematics

    kin = RM65Kinematics(ip=args.ip, port=args.port)
    kin.connect()
    joint_reader._kin = kin
    joint_reader._last = [float(v) for v in kin.get_current_joints_deg()[:6]]
    print(f"关节反馈(SDK): {[round(v, 2) for v in joint_reader._last]}")

    gripper_hw: DahuanGripperModbus | None = None
    gripper_norm = 1.0  # 1=开, 0=合
    if not args.no_gripper:
        try:

            class _Log:
                def info(self, m):
                    print(m)

                def warning(self, m):
                    print(f"警告: {m}", file=sys.stderr)

            assert kin.robot is not None
            gripper_hw = DahuanGripperModbus(kin.robot)
            gripper_hw.setup(_Log())
            print("夹爪: 已挂到录制 SDK 连接")
        except Exception as exc:
            print(f"夹爪初始化失败（继续录制）: {exc}", file=sys.stderr)
            gripper_hw = None

    seed: list[float]
    if args.pose:
        seed = [float(v) for v in args.pose.split(",")]
        if len(seed) != 6:
            raise ValueError("--pose 需要 x,y,z,rx,ry,rz")
        print(f"初始位姿(手动): {[round(v, 4) for v in seed]}")
    else:
        seed = [float(v) for v in kin.get_current_pose()[:6]]
        print(
            f"初始位姿(SDK): xyz=({seed[0]:.4f},{seed[1]:.4f},{seed[2]:.4f}) "
            f"rpy=({seed[3]:.4f},{seed[4]:.4f},{seed[5]:.4f})"
        )

    stick = _make_stick(seed, args)
    stick.start()

    node = rclpy.create_node("rm65_record_movep")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
    )
    pub = node.create_publisher(Cartepos, "/rm_driver/movep_canfd_cmd", qos)
    node.create_subscription(
        JointState,
        "/joint_states",
        lambda msg: joint_reader.on_ros_joint_state(msg.name, msg.position),
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
    )

    for _ in range(50):
        if pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        raise RuntimeError("rm_driver 未订阅 movep_canfd_cmd")

    dt = 1.0 / control_hz
    record_interval = 1.0 / fps
    print(
        f"控制 {control_hz:g}Hz | 录制 {fps}Hz | 单集最长 {episode_time_s:.0f}s | 任务: {task}\n"
        f"按键: A=双相机同步开始  B=同步结束保存  X=丢弃重录  LB/RB=夹爪  Ctrl+C=退出(不保存当前集)\n"
        f"透传: movep 高跟随 follow={int(bool(args.follow))}\n"
        f"目标 episode 数: {num_episodes}（当前已有 {dataset.num_episodes}）"
    )

    recording = False
    ep_start_t = 0.0
    last_record_t = 0.0
    frames_in_ep = 0
    saved_eps = 0
    last_log = 0.0
    saving_episode = False
    ik_fallback_warned = False

    def _buffer_has_frames() -> bool:
        buf = dataset.episode_buffer
        return buf is not None and int(buf.get("size", 0)) >= 1

    def _discard_unsaved() -> None:
        nonlocal recording, frames_in_ep
        try:
            dataset._wait_image_writer()
        except Exception:
            pass
        try:
            dataset.clear_episode_buffer()
        except Exception:
            pass
        recording = False
        frames_in_ep = 0

    def _save_episode_safe(label: str = "") -> bool:
        nonlocal recording, frames_in_ep, saved_eps, saving_episode
        if saving_episode:
            return False
        if frames_in_ep < 1 and not _buffer_has_frames():
            return False
        saving_episode = True
        try:
            dataset.save_episode()
            saved_eps += 1
            tag = f" ({label})" if label else ""
            print(f"<<< 已保存 episode{tag}，累计 episodes={dataset.num_episodes}")
            return True
        except Exception as exc:
            print(f"保存失败: {exc}", file=sys.stderr)
            return False
        finally:
            saving_episode = False
            recording = False
            frames_in_ep = 0

    last_images: dict[str, np.ndarray] = {}

    def _read_all_images() -> dict[str, np.ndarray] | None:
        front_img = _latest_rgb(front_cam)
        astra_img = _latest_rgb(astra_cam)
        if front_img is None:
            front_img = last_images.get(front_key)
        if astra_img is None:
            astra_img = last_images.get(astra_key)
        if front_img is None or astra_img is None:
            return None
        out = {
            front_key: _resize_rgb(front_img, target_h, target_w),
            astra_key: _resize_rgb(astra_img, target_h, target_w),
        }
        last_images.update(out)
        return out

    try:
        while saved_eps < num_episodes:
            t0 = time.time()
            x, y, z, rx, ry, rz = stick.update()
            rclpy.spin_once(node, timeout_sec=0.0)

            # 夹爪（SDK 写寄存器放到后台，避免堵住 50Hz 透传）
            if stick.gripper_event == "open":
                gripper_norm = 1.0
                if gripper_hw is not None:
                    def _go():
                        try:
                            gripper_hw.set_position_normalized(1.0, min_delta=0.0)
                            print("夹爪开")
                        except Exception as exc:
                            print(f"夹爪开失败: {exc}", file=sys.stderr)

                    threading.Thread(target=_go, daemon=True).start()
            elif stick.gripper_event == "close":
                gripper_norm = 0.0
                if gripper_hw is not None:
                    def _gc():
                        try:
                            gripper_hw.set_position_normalized(0.0, min_delta=0.0)
                            print("夹爪合")
                        except Exception as exc:
                            print(f"夹爪合失败: {exc}", file=sys.stderr)

                    threading.Thread(target=_gc, daemon=True).start()

            # 录制状态机：A/B 同时控制双路写盘启停
            ev = stick.record_event
            if ev == "start" and not recording:
                recording = True
                ep_start_t = time.time()
                last_record_t = 0.0
                frames_in_ep = 0
                print(
                    f">>> 开始同步录制 episode #{dataset.num_episodes} "
                    f"（front+{astra_key}，最多 {episode_time_s:.0f}s）"
                )
            elif ev == "stop" and recording:
                if not _save_episode_safe("B"):
                    print("本集无帧，取消保存")
                    _discard_unsaved()
            elif ev == "rerecord":
                if recording or frames_in_ep > 0:
                    _discard_unsaved()
                    print("xxx 已丢弃当前 episode，可按 A 重新开始")
                else:
                    print("当前无缓冲可丢弃")

            if recording and (time.time() - ep_start_t) >= episode_time_s:
                print(f"达到单集时限 {episode_time_s:.0f}s，自动保存")
                if not _save_episode_safe("timeout"):
                    _discard_unsaved()

            # 发 movep（与 rm65_movep_teleop 相同，控制器自己逆解）
            quat = stick.as_quat()
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])
            msg = Cartepos()
            msg.pose = pose
            msg.follow = args.follow
            pub.publish(msg)

            # 取双路帧（预览一直更新；录制中同时落盘）
            now = time.time()
            if (now - last_record_t) >= record_interval:
                last_record_t = now
                images = _read_all_images()
                if images is not None:
                    with _PREVIEW_LOCK:
                        _PREVIEW_FRAMES.update(images)
                if recording and images is not None:
                    joints = joint_reader.read_deg()
                    action_joints = kin.try_inverse_kinematics(
                        [x, y, z, rx, ry, rz], joints
                    )
                    if action_joints is None:
                        action_joints = list(joints)
                        if not ik_fallback_warned:
                            ik_fallback_warned = True
                            print("警告: IK 失败时将 action 回退为实测关节", file=sys.stderr)
                    obs_motors = _motor_dict(joints, gripper_norm)
                    act_motors = _motor_dict(action_joints, gripper_norm)
                    observation = {**obs_motors, **images}
                    action = act_motors
                    obs_frame = build_dataset_frame(
                        dataset.features, observation, prefix="observation"
                    )
                    act_frame = build_dataset_frame(
                        dataset.features, action, prefix="action"
                    )
                    dataset.add_frame({**obs_frame, **act_frame}, task=task)
                    frames_in_ep += 1

            if now - last_log >= 1.0:
                last_log = now
                st = "REC" if recording else "idle"
                loop_ms = (time.time() - t0) * 1000.0
                print(
                    f"[{st}] xyz=({x:.3f},{y:.3f},{z:.3f}) "
                    f"rpy=({rx:.3f},{ry:.3f},{rz:.3f}) "
                    f"frames={frames_in_ep} saved={dataset.num_episodes}/{num_episodes} "
                    f"grip={gripper_norm:.0f} loop={loop_ms:.0f}ms"
                )

            elapsed = time.time() - t0
            if elapsed > dt * 2.5:
                print(f"控制环超时 {elapsed*1000:.0f}ms（透传会断流，容易发响）", flush=True)
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print("\n用户中断")
        if recording or _buffer_has_frames():
            _discard_unsaved()
            print("已丢弃未完成的 episode（Ctrl+C 不保存）")
    finally:
        stick.cleanup()
        try:
            joint_reader.stop()
        except Exception:
            pass
        for cam in (front_cam, astra_cam):
            if cam is None:
                continue
            try:
                cam.disconnect()
            except Exception:
                pass
        try:
            kin.disconnect()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if args.finalize_on_exit and dataset.num_episodes > 0:
            try:
                finalize_dataset(
                    root,
                    embodiment_id=args.embodiment_id,
                    camera_device=str(args.camera),
                    camera_devices={
                        f"observation.images.{front_key}": str(args.camera),
                        f"observation.images.{astra_key}": (
                            (args.astra_rtsp or "").strip() or str(args.astra_camera)
                        ),
                    },
                )
            except Exception as exc:
                print(f"finalize 失败: {exc}", file=sys.stderr)
        print(f"录制结束。数据目录: {dataset.root}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RM65 movep 遥操 + 双相机同步 LeRobot 录制")
    p.add_argument("--ip", default=os.environ.get("RM_IP", "192.168.73.18"))
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--rate", type=float, default=50.0, help="遥操控制频率 Hz")
    p.add_argument("--fps", type=int, default=20, help="数据集写入频率")
    p.add_argument(
        "--sensitivity",
        type=float,
        default=float(os.environ.get("TELEOP_SENSITIVITY", "0.001")),
    )
    p.add_argument(
        "--orient-sensitivity",
        type=float,
        default=float(os.environ.get("TELEOP_ORIENT_SENSITIVITY", "0.003")),
    )
    p.add_argument(
        "--deadzone",
        type=float,
        default=float(os.environ.get("GAMEPAD_DEADZONE", "0.15")),
    )
    p.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TELEOP_FOLLOW", True),
        help="movep CANFD 高跟随(默认开)",
    )
    p.add_argument("--go-ready", action="store_true")
    p.add_argument("--ready-speed", type=int, default=5)
    p.add_argument("--pose", default="")
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--x-min", type=float, default=-0.8)
    p.add_argument("--x-max", type=float, default=0.0)
    p.add_argument("--y-min", type=float, default=-0.5)
    p.add_argument("--y-max", type=float, default=0.5)
    p.add_argument("--z-min", type=float, default=0.05)
    p.add_argument("--z-max", type=float, default=0.85)
    p.add_argument(
        "--camera",
        default=os.environ.get("RECORD_CAMERA", "/dev/video0"),
        help="大白 USB 设备（front）",
    )
    p.add_argument(
        "--rtsp",
        default="",
        help=argparse.SUPPRESS,  # 已停用海康，保留以免旧命令行报错
    )
    p.add_argument(
        "--astra-camera",
        default=os.environ.get("RECORD_ASTRA_CAMERA", DEFAULT_ASTRA_DEVICE),
        help="Astra Pro USB 设备（astra）；若设置 --astra-rtsp 则忽略",
    )
    p.add_argument(
        "--astra-rtsp",
        default=os.environ.get("RECORD_ASTRA_RTSP", ""),
        help="第二路改为 RTSP（USB insert 单目子码流）；设置后不再打开 USB Astra",
    )
    p.add_argument(
        "--front-key",
        default=os.environ.get("RECORD_FRONT_KEY", DEFAULT_FRONT_KEY),
        help="大白图像 feature 短名，默认 front",
    )
    p.add_argument(
        "--astra-key",
        default=os.environ.get("RECORD_ASTRA_KEY", DEFAULT_ASTRA_KEY),
        help="Astra 图像 feature 短名，默认 astra",
    )
    p.add_argument(
        "--exterior-key",
        default="",
        help=argparse.SUPPRESS,  # 已停用海康
    )
    p.add_argument(
        "--camera-key",
        default=os.environ.get("RECORD_CAMERA_KEY", ""),
        help="兼容旧参数；若设置则覆盖 --front-key",
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument(
        "--repo-id",
        default=os.environ.get("RECORD_REPO_ID", "local/rm65_usb_insert_dual"),
    )
    p.add_argument(
        "--root",
        default=os.environ.get(
            "RECORD_ROOT",
            str(ROOT / "datasets" / "rm65_usb_insert_dual"),
        ),
    )
    p.add_argument(
        "--task",
        default="Pick up the USB plug and insert it into the port",
    )
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=float(os.environ.get("RECORD_EPISODE_TIME_S", "120")),
        help="单条 episode 最长秒数，超时自动保存（默认 120）",
    )
    p.add_argument("--video", action="store_true", default=True)
    p.add_argument("--no-video", action="store_false", dest="video")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--embodiment-id",
        default=os.environ.get("RECORD_EMBODIMENT_ID", "rm65_dahuan"),
    )
    p.add_argument(
        "--finalize-on-exit",
        action="store_true",
        default=True,
        help="退出时写 stats.json + clean_report.json（默认开）",
    )
    p.add_argument("--no-finalize-on-exit", action="store_false", dest="finalize_on_exit")
    p.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="录制进程用已取帧出预览（默认开）",
    )
    p.add_argument(
        "--front-preview-port",
        type=int,
        default=int(os.environ.get("RECORD_FRONT_PREVIEW_PORT", "8090")),
        help="大白预览端口（默认 8090）",
    )
    p.add_argument(
        "--astra-preview-port",
        type=int,
        default=int(os.environ.get("RECORD_ASTRA_PREVIEW_PORT", "8093")),
        help="Astra 预览端口（默认 8093；USB insert 新 Astra 用 8094）",
    )
    p.add_argument(
        "--astra-preview-title",
        default=os.environ.get("RECORD_ASTRA_PREVIEW_TITLE", "Astra Pro"),
        help="Astra 预览页标题",
    )
    args = p.parse_args()
    if args.camera_key:
        args.front_key = args.camera_key
    try:
        return run_record(args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
