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
import os
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
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

    @property
    def running(self):
        #返回服务器是否正在运行
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        #返回当前策略配置的图像特征
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

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
        self.rtc_enabled = bool(getattr(policy_specs, "rtc_enabled", False))

        #加载策略类 — RM65 local ckpts need sanitize + normalizer stats
        start = time.perf_counter()
        if self.policy_type == "smolvla" and Path(policy_specs.pretrained_name_or_path).is_dir():
            from lerobot.scripts.server.rm65_ckpt import load_smolvla_for_server

            self.policy = load_smolvla_for_server(
                policy_specs.pretrained_name_or_path, self.device
            )
        else:
            policy_class = get_policy_class(self.policy_type)
            self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
            self.policy.to(self.device)
        if self.rtc_enabled:
            if self.policy_type != "smolvla":
                raise ValueError(f"RTC is wired for smolvla, got {self.policy_type}")
            from lerobot.scripts.server.rm65_rtc import enable_smolvla_rtc

            enable_smolvla_rtc(
                self.policy,
                num_steps=int(getattr(policy_specs, "rtc_num_steps", 4)),
                execution_horizon=int(getattr(policy_specs, "rtc_execution_horizon", 13)),
                max_guidance_weight=float(getattr(policy_specs, "rtc_max_guidance_weight", 10.0)),
                ckpt_dir=policy_specs.pretrained_name_or_path,
            )
            chunk = int(self.policy.config.chunk_size)
            if self.actions_per_chunk < chunk:
                self.logger.info(
                    "RTC returns the full chunk (%s); client asked for %s",
                    chunk,
                    self.actions_per_chunk,
                )
                self.actions_per_chunk = chunk
        end = time.perf_counter()
        self.logger.info("policy loaded and put on device (%.2fs)", end - start)

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
        # 关节变化很小时也继续推理。RM65_SKIP_SIMILAR_OBS=1 时不因 L2<1° 丢帧。
        elif os.environ.get("RM65_SKIP_SIMILAR_OBS", "0").lower() not in (
            "1",
            "true",
            "yes",
        ) and observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
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
        # RawObservation from robot.get_observation() - wrong keys, wrong dtype, wrong image shape
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.device,
        )
        # RM65: robot may send 7-D (joints+gripper) while astra/usb ckpts expect 6-D joint state
        from lerobot.common.constants import OBS_STATE

        if OBS_STATE in observation and self.policy is not None:
            want = None
            try:
                want = int(self.policy.config.robot_state_feature.shape[0])
            except Exception:  # noqa: BLE001
                want = None
            state = observation[OBS_STATE]
            if want is not None and state.shape[-1] > want:
                observation[OBS_STATE] = state[..., :want]

        return observation

    def _rtc_call_kwargs(self, observation_t: TimedObservation) -> dict:
        """Leftover prefix for guided RTC. Empty when this is the first chunk."""
        if not getattr(self, "rtc_enabled", False):
            return {}
        leftover = observation_t.rtc_prev_chunk_left_over
        delay = int(observation_t.rtc_inference_delay or 0)
        if leftover is None or delay <= 0 or leftover.numel() == 0:
            return {}
        chunk = int(self.policy.config.chunk_size)
        if leftover.ndim == 3:
            leftover = leftover[0]
        if delay >= chunk - 1 or leftover.shape[0] < 2:
            self.logger.warning(
                "RTC skip guidance delay=%s leftover=%s chunk=%s",
                delay,
                tuple(leftover.shape),
                chunk,
            )
            return {}
        delay = min(delay, int(leftover.shape[0]))
        horizon = int(self.policy.config.rtc_config.execution_horizon)
        horizon = min(chunk - 1, max(horizon, delay + 2))
        self.logger.info(
            "RTC guide delay=%s horizon=%s leftover=%s",
            delay,
            horizon,
            tuple(leftover.shape),
        )
        return {
            "prev_chunk_left_over": leftover,
            "inference_delay": delay,
            "execution_horizon": horizon,
        }

    def _get_action_chunk(self, observation: dict[str, torch.Tensor], rtc_kwargs: dict | None = None) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        # self.logger.info(observation)
        chunk = self.policy.predict_action_chunk(observation, **(rtc_kwargs or {}))
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation"""
        # self.logger.info("Predicting action chunk")
        inference_starts = time.perf_counter()

        """1. Prepare observation"""
        start_time = time.perf_counter()
        observation = self._prepare_observation(observation_t)
        # self.logger.info("Observation prepared")
        preprocessing_time = time.perf_counter() - start_time

        self.last_processed_obs: TimedObservation = observation_t

        """2. Get action chunk"""
        start_time = time.perf_counter()
        action_tensor = self._get_action_chunk(observation, self._rtc_call_kwargs(observation_t))
        inference_time = time.perf_counter() - start_time

        """3. Post-inference processing"""
        start_time = time.perf_counter()
        # Move to CPU before serializing
        action_tensor = action_tensor.cpu().squeeze(0)

        # Gripper diagnostics (chunk-level): compare vs teleop demos with few open/close edges
        if os.environ.get("RM65_LOG_GRIPPER", "1").lower() not in ("0", "false", "no"):
            try:
                import numpy as np

                grip = action_tensor[:, -1].numpy() if action_tensor.ndim == 2 else action_tensor.numpy()[-1:]
                edges = int(np.sum(np.abs(np.diff((grip > 0.5).astype(np.int8)))))
                self.logger.info(
                    f"grip_chunk t={observation_t.get_timestep()} n={int(grip.shape[0])} "
                    f"first={float(grip[0]):.3f} last={float(grip[-1]):.3f} mean={float(grip.mean()):.3f} "
                    f"min={float(grip.min()):.3f} max={float(grip.max()):.3f} edges(>0.5)={edges} "
                    f"head={np.array2string(grip[:12], precision=2, separator=',')}"
                )
                log_path = os.environ.get(
                    "RM65_GRIPPER_LOG",
                    "/home/sdznkj/zkx_ws/sendlerobot/logs/gripper_chunk.log",
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"chunk\tt={observation_t.get_timestep()}\tn={grip.shape[0]}\t"
                        f"first={grip[0]:.4f}\tlast={grip[-1]:.4f}\tmean={grip.mean():.4f}\t"
                        f"min={grip.min():.4f}\tmax={grip.max():.4f}\tedges={edges}\n"
                    )
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("grip_chunk log failed: %s", exc)

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
