# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
# from lerobot.robots import (  # noqa: F401
#     Robot,
#     RobotConfig,
#     koch_follower,
#     make_robot_from_config,
#     so100_follower,
#     so101_follower,
# )
from worobot.robot import Robot
from worobot.config import RobotConfig
from worobot.unix_robot import UnixRobot, UnixRobotConfig
from worobot.utils import make_robot_from_config

from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.constants import SUPPORTED_ROBOTS
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
from lerobot.common.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.transport.utils import grpc_channel_options, send_bytes_in_chunks

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32, Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """

        # ROS2 initialization
        if not rclpy.ok():
            rclpy.init()

        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)

        self.executor = None
        self.spin_thread = None
        if isinstance(self.robot, Node):
            self.executor = rclpy.executors.MultiThreadedExecutor()
            self.executor.add_node(self.robot)
            self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()

        # 若是ROS2节点，创建executor 并启动spin
        # self.robot.connect()
        # 优化代码结构
        self._connect_robot()

        # === 打印 action features ===
        # self.logger.info(f"Robot action features: {self.robot.action_features}")

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        if config.verify_robot_cameras:
            # Load policy config for validation
            policy_config = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
            policy_image_features = policy_config.image_features

            # The cameras specified for inference must match the one supported by the policy chosen
            validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # ==== 新增缓存变量 ====
        self._task = config.task  # 保存初始 task
        self._task_lock = threading.Lock()  # 保证线程安全

        # ==== ROS2 topic 订阅和发布：动态切换任务&发布状态 ====
        self.task_sub = None
        self.vla_status_pub = None  
        
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        if isinstance(self.robot, Node):

            # 发布 vla_task_status 话题
            self.vla_status_pub = self.robot.create_publisher(Int32, "/vla_task_status", qos_profile)
            
            # 订阅 /vla_task_switch 话题
            self.task_sub = self.robot.create_subscription(
                String,
                "/vla_task_switch",
                self._ros_task_callback,
                qos_profile
            )
            
            # 发布 movej_left_angle / movej_right_angle话题
            self.movej_left_pub = self.robot.create_publisher(Float64MultiArray, '/movej_left_angle', qos_profile)
            self.movej_right_pub = self.robot.create_publisher(Float64MultiArray, '/movej_right_angle', qos_profile)

            # 订阅 /arm_move_info 话题，获取机器人复位状态反馈
            self.arm_move_status_sub = self.robot.create_subscription(
                Int32,
                '/arm_move_info',
                self.arm_move_status_callback,
                qos_profile
            )

            self.logger.info("Created publisher for /vla_task_status")
            self.logger.info("Subscribed to /vla_task_switch for dynamic task switching")
            self.logger.info("Created publishers for /movej_left_angle and /movej_right_angle")
            self.logger.info("Subscribed to /arm_move_info for arm move status feedback")

        # 完成状态控制
        self._init_pose = None                # 初始参考姿态
        self._task_start_time = None          # 任务开始时间戳
        self._task_moved = False              # 是否动过
        self._large_movement_threshold = 0.3  # 必须偏离多少弧度(≈15°)才算动过
        self._min_execution_time = 5.0        # 有时候vla需要几秒钟才能让机器人真正动起来，而不是抖动
        self._pose_tolerance = 0.25           # 容差 (可调)
        self._task_timeout = 55.0             # 任务超时时间（秒）

        # 保存图片控制
        self._done_pose_captured = False      # 是否已保存过图片
        self._stable_counter = 0              # 稳定计数器
        self._stable_threshold = 60           # 连续 30 帧稳定（fps=30 -> 大约1秒） (可调)

        # 日志控制
        self._receive_logged = False          # 接收动作日志
        self._action_logged = False           # 动作执行日志
        self._observation_logged = False      # 观察发送日志
        
        # 添加复位状态跟踪（双臂版本）
        self._reset_completed = False         # 机器人复位状态
        self._left_arm_completed = False      # 左臂复位状态
        self._right_arm_completed = False     # 右臂复位状态

        # ==== 防抖控制 ====
        self._last_status_send_time = 0.0   # 记录最近一次状态帧发送时间戳

        # ==== ROS context bookkeeping ====
        self.executor_active = True
        self._spin_thread_lock = threading.Lock()
        self._ros_objects = [
            self.vla_status_pub,
            self.task_sub,
            self.movej_left_pub,
            self.movej_right_pub,
            self.arm_move_status_sub,
        ]

        # ==== Reset 控制锁，防止重复重启 ROS ====
        self._reset_in_progress = threading.Lock()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s") #调试用，不会显示

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )#调试用

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    # 优化后
    def stop(self):
        """Stop the robot client (graceful and ROS-safe shutdown)"""
        self.shutdown_event.set()

        try:
            # 🧩 Step 1. 停止 Executor
            if self.executor is not None:
                try:
                    self.executor.shutdown()
                except Exception:
                    pass
                if self.spin_thread is not None and self.spin_thread.is_alive():
                    self.spin_thread.join(timeout=2.0)
                self.logger.info("🛑 ROS executor stopped.")
                self.executor = None
                self.spin_thread = None

            # 🧩 Step 2. 解除 Node 绑定，防止垃圾回收时再次触发 C++ 析构
            try:
                if hasattr(self, "robot"):
                    self.executor_active = False
                    del self.robot
                    self.logger.info("🧹 Cleared self.robot reference to avoid late destruction.")
            except Exception as e:
                self.logger.warning(f"Node cleanup warning: {e}")

            # 🧩 Step 3. 延迟关闭 ROS runtime
            if rclpy.ok():
                time.sleep(0.3)  # 等待 DDS 线程退出
                rclpy.shutdown()
                self.logger.info("🔻 ROS shutdown cleanly.")

        except Exception as e:
            self.logger.warning(f"⚠️ Safe ROS shutdown failed: {e}")

        # 🧩 Step 4. 关闭 gRPC 通道
        try:
            self.channel.close()
        except Exception:
            pass

        # 🧩 Step 5. 延迟退出，让后台线程彻底清理
        time.sleep(0.2)
        self.logger.info("🧩 Client stopped and channel closed (safe exit).")


    # ==== 增加接口 _set_task ====
    def _set_task(self, new_task: str):
        """Dynamically update the current task (thread-safe)."""
        # 状态日志 + 写入任务
        with self._task_lock:
            old_task = getattr(self, "_task", None)
            self._task = new_task

        if new_task.strip().lower() == "no task":
            self.logger.info(" ")
            self.logger.info(f"任务切换：机器人已进入空闲模式（当前任务: {new_task}）")
        elif old_task is not None and old_task.strip().lower() == "no task":
            self.logger.info(" ")
            self.logger.info(f"任务切换：机器人恢复任务执行（当前任务: {new_task}）")
        else:
            self.logger.info(" ")
            self.logger.info(f"任务切换：{new_task}")

        # ==== 不同任务设定不同的参数 ====
        task_key = new_task.strip().lower()
        if task_key == "pick up the clothes from sofa":
            self._min_execution_time = 15.0
            self._task_timeout = 55.0
        elif task_key == "open the washing machine door":
            self._min_execution_time = 15.0
            self._task_timeout = 15.0
        elif task_key == "pick up the clothes from the storage basket":
            self._min_execution_time = 15.0
            self._task_timeout = 55.0
        elif task_key == "close the washing machine door":
            self._min_execution_time = 15.0 
            self._task_timeout = 55.0 
        else:
            self._min_execution_time = 15.0
            self._task_timeout = 55.0

        # ==== 🩵 新任务开始时重置计时器与状态（关键修复） ====
        if new_task.strip().lower() not in ["no task", "task completed"]:
            # 每次切换到新任务都重置任务计时器与状态变量
            self._task_start_time = None
            self._done_pose_captured = False
            self._task_moved = False
            self._stable_counter = 0
            self._observation_logged = False
            self.logger.info("⏱️ Reset task timer and pose state for new task.")

        # ==== 如果切换到空闲或任务完成状态，执行彻底清理 ====
        if new_task.strip().lower() in ["no task", "task completed"]:
            task_key = new_task.strip().lower()

            if task_key == "task completed":
                try:
                    # 判断任务是否在超时内完成（成功 or 失败）
                    elapsed = time.time() - (self._task_start_time or 0)
                    if self._task_moved and elapsed < self._task_timeout:
                        self.logger.info("✅ Task success detected, publishing completion status and resetting robot...")
                        self._publish_task_completion(1)  # 成功状态
                    else:
                        self.logger.info("❌ Task failed or timeout, publishing failure status and resetting robot...")
                        self._publish_task_completion(0)  # 失败状态
                except Exception as e:
                    self.logger.warning(f"⚠️ Task completion publish failed: {e}")

            # 等待复位完成（最多 3 秒）
            for _ in range(30):
                if self._reset_completed:
                    self.logger.info("✅ Reset motion confirmed, ready to cleanup runtime.")
                    break
                time.sleep(0.1)

            # 延迟少许，确保发布器完成通信再销毁 ROS context
            time.sleep(0.3) # 给服务器一点 reset 缓冲时间，（典型 0.1~0.3 秒）

            # ✅ 安全地调用 reset，防止同时触发
            if self._reset_in_progress.acquire(blocking=False):
                try:
                    self.logger.info("♻️ Performing safe runtime reset from _set_task() ...")
                    self._reset_runtime_state()
                except Exception as e:
                    self.logger.warning(f"⚠️ Safe reset failed inside _set_task: {e}")
                finally:
                    self._reset_in_progress.release()
            else:
                self.logger.info("⏳ Another reset already in progress, skip duplicate call.")



    def _connect_robot(self):
        """安全地连接或重连机器人"""
        try:
            connected = False
            if hasattr(self.robot, "is_connected"):
                try:
                    # 支持 is_connected 是方法或布尔属性
                    connected = self.robot.is_connected() if callable(self.robot.is_connected) else self.robot.is_connected
                except Exception:
                    connected = False

            if connected:
                self.logger.info("Robot already connected, skipping reconnection.")
                return

            self.robot.connect()
            self.logger.info("✅ Robot connection established or restored.")

        except Exception as e:
            self.logger.warning(f"⚠️ Robot connection attempt failed: {e}")

    def _reset_ros_context(self):
        """彻底重启 ROS executor 与发布器，模拟客户端重启效果。"""

        # 🧩 Step 1：确保旧的 spin 线程和 executor 全部退出
        if hasattr(self, "spin_thread") and self.spin_thread and self.spin_thread.is_alive():
            self.logger.info("⏳ Waiting spin thread to fully exit before ROS shutdown...")
            try:
                self.executor.shutdown()
                self.spin_thread.join(timeout=3.0)
                self.logger.info("✅ Spin thread fully exited before ROS context shutdown.")
            except Exception as e:
                self.logger.warning(f"Spin thread shutdown pre-check failed: {e}")

        self.logger.info("♻️ Rebuilding ROS context for clean restart...")

        # 🧠 Step 2：优雅关闭 ROS runtime（彻底释放 DDS）
        if rclpy.ok():
            try:
                self.logger.info("💤 Waiting briefly before ROS shutdown to flush DDS callbacks...")
                time.sleep(0.5)
                rclpy.shutdown()
                time.sleep(0.5)
                rclpy.init()
                self.logger.info("🔄 Reinitialized ROS context cleanly.")

                # 🧩 关键新增：重建 Node
                try:
                    self.logger.info("🧱 Recreating robot Node after rclpy reinit...")
                    if isinstance(self.robot, Node):
                        node_name = self.robot.get_name() if hasattr(self.robot, "get_name") else "robot_client"
                        robot_cfg = getattr(self.config, "robot", None)
                        # 重新创建 ROS Node 实例
                        self.robot = make_robot_from_config(robot_cfg)
                        self.logger.info(f"✅ Node '{node_name}' recreated successfully.")
                except Exception as e:
                    self.logger.error(f"❌ Failed to recreate Node after ROS restart: {e}")

            except Exception as e:
                self.logger.warning(f"ROS reinit warning: {e}")

        # 🧩 Step 3：彻底销毁旧 executor
        if self.executor_active:
            try:
                self.executor.shutdown()
                self.executor_active = False
                if self.spin_thread and self.spin_thread.is_alive():
                    self.logger.info("⏳ Waiting old spin thread to terminate...")
                    self.spin_thread.join(timeout=2.0)
                self.logger.info("🛑 Old executor fully stopped.")
            except Exception as e:
                self.logger.warning(f"Executor shutdown failed: {e}")

        # 🧩 Step 4：销毁旧的 ROS 通信对象（publisher/subscriber）
        for obj in getattr(self, "_ros_objects", []):
            try:
                if obj is not None and hasattr(obj, "destroy"):
                    obj.destroy()
            except Exception:
                pass

        # 🧩 Step 5：重建 executor
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.robot)

        # 🧩 Step 6：启动安全 spin 线程（增加异常捕获）
        def _spin_executor():
            try:
                self.executor.spin()
            except Exception as e:
                # 防止 “InvalidHandle” 报错终止
                self.logger.warning(f"Spin loop exited safely: {e}")

        with self._spin_thread_lock:
            self.spin_thread = threading.Thread(
                target=_spin_executor,
                name="ros_spin_thread",
                daemon=True
            )
            self.spin_thread.start()
            self.executor_active = True

        # 🧩 Step 7：使用安全 publisher 创建函数
        def safe_create_publisher(msg_type, topic):
            try:
                return self.robot.create_publisher(msg_type, topic, qos_profile)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create publisher for {topic}: {e}")
                return None

        # 🧩 Step 8：重新创建 ROS 通信对象
        self.vla_status_pub = safe_create_publisher(Int32, "/vla_task_status")
        self.task_sub = self.robot.create_subscription(String, "/vla_task_switch", self._ros_task_callback, qos_profile)
        self.movej_left_pub = safe_create_publisher(Float64MultiArray, "/movej_left_angle")
        self.movej_right_pub = safe_create_publisher(Float64MultiArray, "/movej_right_angle")
        self.arm_move_status_sub = self.robot.create_subscription(Int32, "/arm_move_info", self.arm_move_status_callback, qos_profile)

        self._ros_objects = [
            self.vla_status_pub,
            self.task_sub,
            self.movej_left_pub,
            self.movej_right_pub,
            self.arm_move_status_sub,
        ]

        self.logger.info("✅ ROS context fully rebuilt and clean.")

    def _reset_runtime_state(self):
        """彻底清理机器人客户端运行状态，使其与重启程序时完全一致"""

        # === 🧩 防止嵌套调用 ===
        if getattr(self, "_reset_in_progress", None) and self._reset_in_progress.locked():
            self.logger.warning("⚠️ Skip nested _reset_runtime_state call (already resetting).")
            return

        with self._reset_in_progress:  # 确保只允许一个 reset 同时进行
            # === 🧩 暂停所有运行线程，等待完全停止 ===
            self.logger.info("⏸️ Stopping all runtime threads before ROS reset...")
            self.must_go.clear()
            self.shutdown_event.set()
            time.sleep(0.2)

        # 🔒 等待 send_observation() 与 control_loop() 全部退出
        active_threads = [t for t in threading.enumerate() if "control_loop" in t.name or "observation" in t.name]
        for t in active_threads:
            self.logger.info(f"⏳ Waiting thread {t.name} to exit safely before ROS shutdown...")
            t.join(timeout=2.0)

        self._observation_logged = True

        self.logger.info("🧹 Resetting robot client runtime state (full clean)...")

        # 0️⃣ 先重启 ROS 通信管线（彻底清空缓存与线程）
        self._reset_ros_context()

        # 0.5️⃣ 重建 gRPC 通道，避免旧的异步流滞留
        try:
            self.channel.close()
        except Exception:
            pass
        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{self.config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info("🔌 gRPC channel refreshed.")

        # === 1️⃣ 清空动作队列 ===
        with self.action_queue_lock:
            cleared = 0
            while not self.action_queue.empty():
                self.action_queue.get_nowait()
                cleared += 1
            if cleared > 0:
                self.logger.info(f"清空动作队列，共移除 {cleared} 个残留动作。")

        # === 2️⃣ 重置时间与姿态状态 ===
        self._task_start_time = None
        self._init_pose = None
        self._task_moved = False
        self._done_pose_captured = False
        self._stable_counter = 0

        # === 3️⃣ 重置计数器与性能跟踪 ===
        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # === 4️⃣ 重置状态标志 ===
        self._receive_logged = False
        self._action_logged = False
        self._observation_logged = False
        self._reset_completed = False
        self._left_arm_completed = False
        self._right_arm_completed = False

        # === 5️⃣ 延迟防抖时间戳 ===
        self._last_status_send_time = 0.0

        # === 6️⃣ 恢复默认控制参数 ===
        self._pose_tolerance = 0.25
        self._stable_threshold = 60
        self._large_movement_threshold = 0.3
        self.start_barrier = threading.Barrier(2)

        # === 7️⃣ 重置观测控制开关 ===
        self.must_go.clear()
        time.sleep(0.05)
        self.must_go.set()

        # === 8️⃣ 确保机器人底层重新连接 ===
        self._connect_robot()

        # === 🟢 恢复观测与动作线程 ===
        self.must_go.set()
        self._observation_logged = False
        self.logger.info("▶️ Observation/action threads resumed after safe ROS reset.")

        self.logger.info("✅ Runtime state reset complete. Ready for next task.")


    # ==== ROS2 topic 订阅回调函数 ====
    def _ros_task_callback(self, msg: String):
        """ROS2 回调：接收到 /vla_task_switch 消息后切换任务"""
        new_task = msg.data
        self._set_task(new_task)

    def arm_move_status_callback(self, msg: Int32):
        """
        机器人复位状态反馈回调
        10: 左臂失败, 11: 左臂完成
        20: 右臂失败, 21: 右臂完成
        """
        if msg.data == 11:
            self.logger.info("左手臂状态反馈: 动作完成 ✅")
            self._left_arm_completed = True
        elif msg.data == 21:
            self.logger.info("右手臂状态反馈: 动作完成 ✅")
            self._right_arm_completed = True
        else:
            self.logger.warning(f"收到未知的手臂状态反馈: {msg.data}")
        
        # 检查双臂是否都完成
        if self._left_arm_completed and self._right_arm_completed:
            self.logger.info("双臂复位状态已全部收到")
            self._reset_completed = True  # 双臂都完成时才设置为True

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps
    
    # Aggregates incoming actions based on their timesteps.
    # 聚合重叠部分的动作
    # 基于时间步timestep对新传入的动作队列与当前内部的动作队列进行聚合
    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            # 时间步小于等于最新动作时间步的动作，跳过（删除）
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            # 如果新动作的时间步不在当前动作队列中，直接添加
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        # self.logger.info("Action receiving thread starting")

        while self.running:

            # 检测当前任务状态
            with self._task_lock:
                current_task = self._task
            # ==== 当 task 为 "no task" 或 "task completed" 时 ====
            if current_task.strip().lower() in ["no task", "task completed"]:
                if not self._receive_logged:
                    self.logger.info(f"Task is '{current_task}' → 停止接收动作")
                    self._receive_logged = True

                # # 清空动作队列，避免新任务执行旧动作
                # with self.action_queue_lock:
                #     queue_size = self.action_queue.qsize()
                #     while not self.action_queue.empty():
                #         self.action_queue.get_nowait()
                #     self.logger.info("receive_actions：清空动作队列")
                #     if queue_size > 0:
                #         self.logger.info(f"receive_actions：清理了 {queue_size} 个残留动作")

                time.sleep(0.1)
                continue   # 🔴 不去拉动作
            self._receive_logged = False  # 重置标志，下一次进入任务时可以打印日志

            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    self.logger.info("Received empty actions chunk, waiting for next call")
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    # self.logger.info(
                    #     f"Received action chunk for step #{first_action_timestep} | "
                    #     f"Latest action: #{latest_action} | "
                    #     f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    #     f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                    #     f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    # )
                
                first_action_timestep = timed_actions[0].get_timestep()
                self.logger.info(f"Received first action chunk for step #{first_action_timestep}")
                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    # self.logger.info(
                    #     f"Latest action: {latest_action} | "
                    #     f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                    #     f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    #     f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    # )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        #通过线程安全的方式检查动作队列是否有可用的动作
        with self.action_queue_lock:
            #检查动作队列是否为空
            return not self.action_queue.empty()


    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # 检测当前任务状态
        with self._task_lock:
            current_task = self._task
        # ==== 当 task 为 "no task" 或 "task completed" 时 ====
        if current_task.strip().lower() in ["no task", "task completed"]:
            if not self._action_logged:
                self.logger.info(f"Task is '{current_task}' → 停止动作执行")
                self._action_logged = True
    
            # 清空动作队列，避免新任务执行旧动作
            with self.action_queue_lock:
                queue_size = self.action_queue.qsize()
                while not self.action_queue.empty():
                    self.action_queue.get_nowait()
                self.logger.info("control_loop_action：清空动作队列")
                if queue_size > 0:
                    self.logger.info(f"control_loop_action：清理了 {queue_size} 个残留动作")

            time.sleep(0.1)
            return None
        self._action_logged = False  # 重置标志，下一次进入任务时可以打印日志

        #判断主函数有没有运行到这里
        # self.logger.info("Control loop action started")

        # Lock only for queue operations
        get_start = time.perf_counter()
        #使用锁进行线程安全的队列操作
        with self.action_queue_lock:
            #每次获取动作时，记录动作队列的大小
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start 

        #执行动作
        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action()) #将动作张量转换为动作字典
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep() #更新最新动作时间戳

        # self.logger.info(
        #         f"Ts={timed_action.get_timestamp()} | "
        #         f"Action #{timed_action.get_timestep()} performed | "
        #         f"Queue size: {current_queue_size}"
        #     )

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action


    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            # 计算动作队列的当前大小与动作块大小的比值，小于等于阈值时，发送观察结果，进行下一次推理
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold


    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            # self.logger.info(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    # ==== 新增判断是否进入完成姿态 ====
    def _is_within_tolerance(self, current_pose: dict, ref_pose: dict, tol: float) -> bool:
        """判断当前关节是否在参考姿态的容差范围内"""
        for k, v in ref_pose.items():
            if abs(current_pose.get(k, 0.0) - v) > tol:
                return False
        return True
    
    # ==== 新增保存图片 ====
    def _save_camera_images(self, raw_observation: RawObservation):
        """保存 head、left、right 三个相机的图像到 outputs/captured_images 目录"""
        import cv2, os, time
        import numpy as np
        from PIL import Image

        # 自动检测根目录（当前工作目录）
        project_root = os.getcwd()
        save_dir = os.path.join(project_root, "outputs", "captured_images")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = int(time.time())
        for cam_name in ["head", "left", "right"]:
            if cam_name in raw_observation:
                img = raw_observation[cam_name]
                if img is None:
                    self.logger.warning(f"No image data for camera {cam_name}, skipping save.")
                    continue

                # === 类型兼容：PIL → numpy ===
                if isinstance(img, Image.Image):
                    img = np.array(img)

                if not isinstance(img, np.ndarray):
                    self.logger.error(f"Camera {cam_name} has unsupported type {type(img)}, skipping save.")
                    continue

                # save_path = os.path.join(save_dir, f"{cam_name}_{timestamp}.jpg")
                save_path = os.path.join(save_dir, f"{cam_name}.jpg")
                try:
                    cv2.imwrite(save_path, img[:, :, ::-1])  # RGB→BGR
                    # self.logger.info(f"Saved image: {save_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save image for {cam_name}: {e}")
            else:
                self.logger.warning(f"Camera {cam_name} not found in observation, skipping.")

    # ==== ROS2 topic 发布任务完成状态 ====
    def _publish_task_completion(self, status: int):

        # 这个回原点的函数
        # 启动：失败后，再次执行，会抖动！          不启用：失败后，无法成功执行，它会接着之前的动作无法成功
        # 启动，成功后，再次执行，会轻微的抖动       不启用：成功后，非常顺畅的衔接
        '''不管成功与否，都发送机器人复位命令到/movej_left_angle和/movej_right_angle话题'''
        if hasattr(self, 'movej_left_pub') and self.movej_left_pub is not None and \
           hasattr(self, 'movej_right_pub') and self.movej_right_pub is not None:
            
            # 发送机器人复位命令
            left_angle_msg = Float64MultiArray()
            left_angle_msg.data = [0.8, 0.0, 0.0, -2.1, 0.0, 0.5, 0.0]
            self.movej_left_pub.publish(left_angle_msg) # 发送机器人左臂复位命令

            right_angle_msg = Float64MultiArray()
            right_angle_msg.data = [-0.8, 0.0, 0.0, 2.1, 0.0, -0.5, 0.0]
            self.movej_right_pub.publish(right_angle_msg) # 发送机器人右臂复位命令

            # self.logger.info(f"Published movej_left_angle: {left_angle_msg.data} to /movej_left_angle")
            # self.logger.info(f"Published movej_right_angle: {right_angle_msg.data} to /movej_right_angle")

            # 等待机器人复位完成反馈
            self.logger.info("等待双臂复位完成...")
            for i in range(100):  # 10秒 = 100 * 0.1秒
                if self._reset_completed:  # 只有双臂都完成才会被设为True
                    self.logger.info("双臂复位已完成")
                    break
                time.sleep(0.1)
            else:
                # 超时处理：只记录日志，不处理失败情况
                self.logger.error("双臂复位超时，未在5秒内收到双臂完成反馈")
        else:
            self.logger.warning("Joystick publisher not available (robot is not a ROS2 node)")


        """发布任务完成状态到/vla_task_status话题"""
        """1:成功,0:失败"""
        if hasattr(self, 'vla_status_pub') and self.vla_status_pub is not None:
            status_msg = Int32()
            status_msg.data = status  # status 是 int 类型（0 或 1）
            self.vla_status_pub.publish(status_msg)
            self.logger.info(f"Published task completion status ({status}) to /vla_task_status")
        else:
            self.logger.warning("VLA status publisher not available (robot is not a ROS2 node)")

    # def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
    def control_loop_observation(self, verbose: bool = False) -> RawObservation:

        try:
            # 检测当前任务状态
            with self._task_lock:
                current_task = self._task
            # ==== 当 task 为 "no task" 或 "task completed" 时 ====
            if current_task.strip().lower() in ["no task", "task completed"]:

                if not self._observation_logged:
                    now = time.time()
                    # 防止短时间内重复发送同一状态帧（1秒内只发一次）
                    if now - self._last_status_send_time < 1.0:
                        time.sleep(0.1)
                        return None
                    self._last_status_send_time = now

                    # ✅ 获取真实观测，保证字段完整性
                    raw_observation: RawObservation = self.robot.get_observation()
                    raw_observation["task"] = current_task
                    raw_observation["_meta"] = "status_frame"

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    timed_obs = TimedObservation(
                        timestamp=time.time(),
                        observation=raw_observation,
                        timestep=max(latest_action, -1),
                    )
                    self.send_observation(timed_obs)
                    self.logger.info(f"Task is '{current_task}' → 停止发送观察结果，同时发送'{current_task}'到服务器")
                    self._observation_logged = True
                else:
                    time.sleep(0.1)
                    return None
            else:
                self._observation_logged = False  # 重置标志，下一次进入任务时可以打印日志

            # Get serialized observation bytes from the function
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            # self.logger.info(f"Raw observation: {raw_observation}")  # debug: 打印原始观测结果
            # self.logger.info(f"[DEBUG] raw_observation keys: {list(raw_observation.keys())}")
            # if "observation.state" in raw_observation:
            #     self.logger.info(f"[DEBUG] state keys: {list(raw_observation['observation.state'].keys())[:5]} ...")
            raw_observation["task"] = current_task

            # ==== 捕获初始参考姿态 ====
            # if self._init_pose is None:
            #     joint_state = {k: v for k, v in raw_observation.items() if k.endswith(".pos")}
            #     if joint_state:
            #         self._init_pose = joint_state.copy()
            #         self.logger.info(f"[DEBUG] Captured initial reference pose: {self._init_pose}")
            # ==== 固定的初始参考姿态（可选）====
            # 当前使用固定初始姿态，仅适用于静态场景，若更换机器人或重启需重新采样
            # self._init_pose = {
            #     "left_joint_1.pos": 0.8,
            #     "left_joint_2.pos": 0.0,
            #     "left_joint_3.pos": 0.0,
            #     "left_joint_4.pos": -2.1,
            #     "left_joint_5.pos": 0.0,
            #     "left_joint_6.pos": 0.5,
            #     "left_joint_7.pos": 0.0,
            #     "left_gripper.pos": 0.0,
            #     "right_joint_1.pos": -0.8,
            #     "right_joint_2.pos": 0.0,
            #     "right_joint_3.pos": 0.0,
            #     "right_joint_4.pos": 2.1,
            #     "right_joint_5.pos": 0.0,
            #     "right_joint_6.pos": -0.5,
            #     "right_joint_7.pos": 0.0,
            #     "right_gripper.pos": 0.3,   
            # }
            self._init_pose = {
                "left_joint_1.pos": 0.5836303845087929,
                "left_joint_2.pos": -0.06694964226328028,
                "left_joint_3.pos": -0.04279152536784058,
                "left_joint_4.pos": -1.866833488101575,
                "left_joint_5.pos": 0.029810818083625276,
                "left_joint_6.pos": 0.3864591539079374,
                "left_joint_7.pos": 0.013143419183310632,
                "left_gripper.pos": 0.06666666666666667,
                "right_joint_1.pos": -0.7780386711500147,
                "right_joint_2.pos": 0.0013666003203746803,
                "right_joint_3.pos": -0.002700293200110238,
                "right_joint_4.pos": 1.935055744623718,
                "right_joint_5.pos": -0.004615947496325254,
                "right_joint_6.pos": -0.44400365053660595,
                "right_joint_7.pos": -0.0018802900367186045,
                "right_gripper.pos": 0.0,   
            }


            # # ==== 记录任务开始时间 ====
            if self._task_start_time is None and current_task.strip().lower() not in ["no task", "task completed"]:
                self._task_start_time = time.time()
                self._task_moved = False
                self.logger.info("Task started, begin tracking for completion check")

            #时间步与观测封装
            with self.latest_action_lock:
                latest_action = self.latest_action

            #对时间戳和时间步进行封装
            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                # self.logger.info(
                #     f"Obs #{observation.get_timestep()} | "
                #     f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                #     f"Target: {fps_metrics['target_fps']:.2f}"
                # )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )
            
            # self.logger.info(f"Obs timestep #{observation.get_timestep()} ")

            # 计算动作执行时间
            elapsed = time.time() - (self._task_start_time or 0)

            # 检查是否超时（仅在有效任务期间触发）
            if (
                current_task.strip().lower() not in ["no task", "task completed"]
                and self._task_start_time is not None
                and elapsed > self._task_timeout
            ):
                self.logger.warning(f"Task timeout ({self._task_timeout}s), marking as failed")
                # 标记任务动过（保证 _set_task 内判断正确，否则可能被误判为“未执行任务”）
                self._task_moved = True

                # 切换到任务完成，由 _set_task() 统一处理复位与状态发布
                self._set_task("task completed")
                # self._publish_task_completion(0)  # 发布任务失败状态

                return None

            # # 检查是否超时
            # if elapsed > self._task_timeout:
            #     self.logger.warning(f"Task timeout ({self._task_timeout}s), marking as failed")
            #     self._set_task("task completed")
            #     self._publish_task_completion(0)  # 发布任务失败状态
            #     return None

            # ==== 判断是否进入完成姿态 ====
            if (not self._done_pose_captured and self._init_pose is not None):
                # 提取所有关节值
                current_pose = {k: v for k, v in raw_observation.items() if k.endswith(".pos")}

                # 检查是否动过（偏离初始姿态超过阈值）
                if not self._task_moved:
                    for k, v in self._init_pose.items():
                        if abs(current_pose.get(k, 0.0) - v) > self._large_movement_threshold:
                            self._task_moved = True
                            self.logger.info("Robot moved away from init pose, completion check is now enabled")
                            break

                # # 计算任务执行时间
                # elapsed = time.time() - (self._task_start_time or 0)

                # 动过且超过出手时间，才允许进入完成判断
                if self._task_moved and elapsed > self._min_execution_time:

                    # 检查是否在容差范围内
                    if self._is_within_tolerance(current_pose, self._init_pose, self._pose_tolerance):
                        self._stable_counter += 1
                        diffs = {k: abs(current_pose.get(k, 0.0) - self._init_pose.get(k, 0.0)) for k in self._init_pose}
                        self.logger.debug(
                            f"[PoseCheck] Within tolerance=True | Counter={self._stable_counter}/{self._stable_threshold} "
                            f"| Tolerance={self._pose_tolerance} | Elapsed={elapsed:.2f}s | "
                            f"Diffs(sample)={list(diffs.items())[:5]}"
                        )

                        # 直接进入完成状态（不考虑稳定帧数）
                        # self._done_pose_captured = True
                        # self.logger.info("Task completed → stop sending observations") 
                        # self._set_task("task completed")   # 切换到task completed，通知 server 结束推理
                        # self._publish_task_completion(1)   # 发布任务成功状态
                        # return None
                    
                        # 连续多帧稳定后，进入完成状态
                        if self._stable_counter >= self._stable_threshold:
                            self.logger.info("Robot has stabilized at initial pose → saving camera images")
                            self._save_camera_images(raw_observation)
                            self._done_pose_captured = True

                            # 确保 _task_moved 在任何情况下都为 True
                            if not self._task_moved:
                                self._task_moved = True
                            # 仅切换状态，由 _set_task() 统一处理复位与状态发布
                            self._set_task("task completed")   # 切换到task completed，通知 server 结束推理
                            # self._publish_task_completion(1)   # 发布任务成功状态

                            return None

                    else:
                        self.logger.debug(f"[PoseCheck] Within tolerance=False | Reset counter | Elapsed={elapsed:.2f}s")
                        self._stable_counter = 0

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
    

    # def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
    def control_loop(self, verbose: bool = False) -> tuple[Observation, Action]:
    
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available(): # 检查是否有可执行的动作
                self.logger.debug("Actions available, performing action")
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation(): # 检查是否准备好发送观察结果
                # _captured_observation = self.control_loop_observation(task, verbose)
                _captured_observation = self.control_loop_observation(verbose=verbose)

            # self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}") #记录单次循环控制时间
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    '''
    one thread for receiving actions from the server
    one thread for control loop (performing actions and sending observations)
    '''
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        # 创建动作接收线程
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            # 调用主线程运行控制循环
            # client.control_loop(task=cfg.task)

            # 实现动态任务切换
            client.control_loop()

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
