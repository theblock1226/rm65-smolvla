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
Example:
```shell
python src/lerobot/scripts/server/policy_server.py \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue

import draccus
import grpc
import torch

from lerobot.common.policies.factory import get_policy_class
from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.constants import SUPPORTED_POLICIES
from lerobot.scripts.server.helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.common.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.common.transport.utils import receive_bytes_in_chunks


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        #追踪和计算FPS，帮助测量观测数据从客户端流入服务器的速度
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        
        #观测队列，用于存放推理的观测数据
        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None

        # ==== 新增缓存变量，保存上次加载的模型信息 ====
        self._last_policy_path = None
        self._last_policy_type = None
        self._last_policy_device = None
        self._last_task = None # 这里只是用于打印日志，方便观察任务切换。真正的任务切换逻辑在 robot_client.py 里，它把task塞到观测数据里，再传递给策略服务器


    @property
    def running(self):
        #返回服务器是否正在运行
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        #返回当前策略配置的图像特征
        return self.policy.config.image_features

    # def _reset_server(self) -> None:
    #     """Flushes server state when new client connects."""
    #     # only running inference on the latest observation received by the server
    #     self.shutdown_event.set()
    #     self.observation_queue = Queue(maxsize=1)

    #     with self._predicted_timesteps_lock:
    #         self._predicted_timesteps = set()

    # 优化代码
    def _reset_server(self) -> None:
        """Reset the runtime state of the policy server to a clean, startup-like condition."""

        with getattr(self, "_reset_lock", threading.Lock()):  # 防止并发 reset
            self.logger.info("🔄 Resetting PolicyServer runtime state...")

            # === 停止推理线程（防止并发访问）===
            self.shutdown_event.set()
            time.sleep(0.05)

            # === 清空动态运行状态 ===
            while not self.observation_queue.empty():
                try:
                    self.observation_queue.get_nowait()
                except Exception:
                    break

            self.observation_queue = Queue(maxsize=1)             # 清空观测缓存
            self.last_processed_obs = None                        # 清空最后观测
            self._last_task = None                                # 清空任务记录

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.clear()                 # 清空预测步集合

            # === 重置性能计数器 ===
            self.fps_tracker = FPSTracker(target_fps=self.config.fps)

            # === 保留模型与配置 ===
            # 不重置 policy、device、lerobot_features、actions_per_chunk 等

            # === 重新启用推理线程 ===
            self.shutdown_event.clear()

            self.logger.info("✅ PolicyServer runtime reset complete (model retained).")





    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            #检查服务器是否正在运行
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer() #获取客户端ID

        policy_specs = pickle.loads(request.data)  # nosec #数据反序列化

        if not isinstance(policy_specs, RemotePolicyConfig):
            #验证反序列化后的对象类型
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        #加载策略类
        policy_class = get_policy_class(self.policy_type)

        # ==== 新增逻辑：判断是否需要重新加载模型 ====
        if (
            self.policy is not None
            and self._last_policy_path == policy_specs.pretrained_name_or_path
            and self._last_policy_type == policy_specs.policy_type
            and self._last_policy_device == policy_specs.device
        ):
            # 如果模型和上次一样，不重新加载
            self.logger.info("Reusing cached policy (skip reload).")
        else:
            # 否则才加载新模型
            start = time.perf_counter()
            self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
            self.policy.to(self.device) #checkpoint to the correct device
            end = time.perf_counter()
            self.logger.info(
                f"policy loaded and put on device (took {end - start:.4f} seconds)"
            )

            # 更新缓存信息
            self._last_policy_path = policy_specs.pretrained_name_or_path
            self._last_policy_type = policy_specs.policy_type
            self._last_policy_device = policy_specs.device

        # start = time.perf_counter()
        # #加载预训练策略
        # self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        # self.policy.to(self.device) #checkpoint to the correct device
        # end = time.perf_counter()
        # self.logger.info("policy loaded and put on device")
        # self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        #接收观测数据并对观测数据进行判断
        client_id = context.peer()
        # self.logger.info(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        #阻塞地接收数据，直到request_iterator迭代完毕
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        #反序列化接收到的字节流
        timed_observation = pickle.loads(received_bytes)  # nosec
        #这里需要检查反序列化接收到的观测是否正确

        # ==== 新增：打印 task ====
        if "task" in timed_observation.observation:
            readable_client_id = client_id.replace("ipv4:", "")
            current_task = timed_observation.observation["task"]

            # 判断是否与上一次不同
            if self._last_task is None:
                self.logger.info(f"Client {readable_client_id} | First received task: {current_task}")
            elif self._last_task != current_task:
                self.logger.info(
                    f"Client {readable_client_id} | Task switched: {self._last_task} -> {current_task}"
                )
            else:
                self.logger.info(f"Client {readable_client_id} | Task unchanged: {current_task}")

            # === 检测状态 ===
            if current_task in ["no task", "task completed"]:
                # 防止重复 reset
                if self._last_task == current_task and self.running:
                    self.logger.debug(f"Task '{current_task}' already reset, skipping duplicate reset.")
                else:
                    self.logger.info(f"Client {readable_client_id} | Task is '{current_task}' → resetting server state")
                    self._reset_server()
                    self.shutdown_event.clear()  # 重新允许推理线程运行

            # 更新缓存
            self._last_task = current_task

        deserialize_time = time.perf_counter() - start_deserialize

        #debug: 接收到观测结果
        # self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        #debug-> Received observation #0 | Avg FPS: 0.00 | Target: 30.00 | One-way latency: 1.97ms
                #Received observation #0 | Avg FPS: 30.56 | Target: 30.00 | One-way latency: 4.60ms
                #Received observation #0 | Avg FPS: 30.19 | Target: 30.00 | One-way latency: 4.42ms
                #Received observation #0 | Avg FPS: 30.06 | Target: 30.00 | One-way latency: 2.79ms
        self.logger.info(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        # self.logger.debug(
        #     f"Server timestamp: {receive_time:.6f} | "
        #     f"Client timestamp: {obs_timestamp:.6f} | "
        #     f"Deserialization time: {deserialize_time:.6f}s"
        # )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            #观测结果被过滤掉了
            self.logger.info(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        #根据之前接收到的观测数据进行推理并生成动作
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            #从队列中获取观测数据
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            #模型推理，生成动作
            start_time = time.perf_counter()
            #debug: 打印当前正在进行的动作预测
            # self.logger.info("Starting action chunk prediction")
            action_chunk = self._predict_action_chunk(obs)
            # self.logger.info("Action chunk prediction completed")
            inference_time = time.perf_counter() - start_time

            #序列化动作数据
            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        #检查时间步是否已经被处理
        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False
        #检查观测数据是否与上一个观测数据相似
        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.info("Similar observation detected, skipping processing.")
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        # # debug: must_go:false
        # self.logger.info(f"must_go: {obs.must_go}")
        # # debug: last_processed_obs: None
        # if self.last_processed_obs is None:
        #     self.logger.info("Last processed observation is None")
        # # self.logger.info(f"sanity_checks: {self._obs_sanity_checks(obs, self.last_processed_obs)}")   
        # debug:
        # 第一帧：
        # INFO 2025-07-30 11:15:49 y_server.py:303 must_go: True
        # INFO 2025-07-30 11:15:49 y_server.py:306 Last processed observation is None
        # INFO 2025-07-30 11:15:49 y_server.py:328 successfully enqueued observation
        # INFO 2025-07-30 11:15:49 y_server.py:230 Running inference for observation #0 (must_go: True)
        # INFO 2025-07-30 11:15:49 y_server.py:240 Starting action chunk prediction
        # ERROR 2025-07-30 11:15:49 y_server.py:274 Error in StreamActions: 'SmolVLAPolicy' object has no attribute 'predict_action_chunk'

        #如果观测数据必须被处理，或者这是第一个观测数据，或者当前观测数据与上一个观测数据不相似，则将其放入队列
        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            self.logger.info("successfully enqueued observation")
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_observation(self, observation_t: TimedObservation) -> Observation:
        """
        Prepare observation, ready for policy inference.
        E.g.: To keep observation sampling rate high (and network packet tiny) we send int8 [0,255] images from the
        client and then convert them to float32 [0,1] images here, before running inference.
        """
        # self.logger.info(f"[DEBUG] observation_t.observation keys: {list(observation_t.observation.keys())}")
        # self.logger.info(f"[DEBUG] lerobot_features keys: {list(self.lerobot_features.keys())}")

        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.device,
        )
        # processed Observation - right keys, right dtype, right image shape

        return observation

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        # self.logger.info(observation)
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        # self.logger.info("Predicting action chunk")
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        #这里报错
        # self.logger.info(observation_t.observation)
        observation = self._prepare_observation(observation_t)
        # self.logger.info("Observation prepared")
        preprocessing_time = time.perf_counter() - start_time

        self.last_processed_obs: TimedObservation = observation_t

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocessing_time = time.perf_counter() - start_time
        inference_stops = time.perf_counter()

        self.logger.info(
            f"Observation {observation_t.get_timestep()} |"
            f"Inference time: {1000 * (inference_stops - inference_starts):.2f}ms"
        )

        # full-process latency breakdown for debugging purposes
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Preprocessing time: {1000 * (preprocessing_time - inference_starts):.2f}ms | "
            f"Inference time: {1000 * (inference_time - preprocessing_time):.2f}ms | "
            f"Postprocessing time: {1000 * (postprocessing_time - inference_time):.2f}ms | "
            f"Total time: {1000 * (postprocessing_time - inference_starts):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    # Register the PolicyServer servicer
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    # Add insecure port for the server to listen on
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
