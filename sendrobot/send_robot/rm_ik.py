#!/usr/bin/env python3
"""RM65 逆解辅助：笛卡尔位姿 [x,y,z,rx,ry,rz] + 参考关节角 -> 6 关节角（度）。"""

from __future__ import annotations

import logging
from typing import List, Sequence

from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

logger = logging.getLogger(__name__)


class RM65Kinematics:
    def __init__(self, ip: str = "192.168.73.18", port: int = 8080):
        self.ip = ip
        self.port = port
        self._robot: RoboticArm | None = None
        self._handle_id: int | None = None

    @property
    def robot(self) -> RoboticArm | None:
        return self._robot

    def connect(self) -> None:
        if self._robot is not None:
            return
        self._robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self._robot.rm_create_robot_arm(self.ip, self.port)
        if handle.id < 0:
            raise RuntimeError(f"连接 RM65 失败: {self.ip}:{self.port}, handle={handle.id}")
        self._handle_id = handle.id
        logger.info("RM65 kinematics connected at %s:%s", self.ip, self.port)

    def disconnect(self) -> None:
        if self._robot is None:
            return
        try:
            if self._handle_id is not None and self._handle_id >= 0:
                self._robot.rm_delete_robot_arm()
        finally:
            self._robot = None
            self._handle_id = None

    def get_current_pose(self) -> List[float]:
        if self._robot is None:
            raise RuntimeError("RM65 kinematics SDK 未连接")
        code, state = self._robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"读取末端位姿失败, code={code}")
        return list(state["pose"])

    def get_current_joints_deg(self) -> List[float]:
        if self._robot is None:
            raise RuntimeError("RM65 kinematics SDK 未连接")
        code, state = self._robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"读取关节角失败, code={code}")
        return [float(q) for q in state["joint"][:6]]

    def inverse_kinematics(
        self,
        target_pose: Sequence[float],
        ref_joints_deg: Sequence[float],
    ) -> List[float]:
        result = self.try_inverse_kinematics(target_pose, ref_joints_deg)
        if result is None:
            raise RuntimeError(f"逆解失败, pose={list(target_pose)}")
        return result

    def work_delta_to_pose(
        self,
        current_pose: Sequence[float],
        dx: float,
        dy: float,
        dz: float,
    ) -> List[float]:
        """基坐标系（Work）下平移，姿态保持不变。"""
        assert self._robot is not None
        delta = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]
        return list(self._robot.rm_algo_pose_move(list(current_pose), delta, 0))

    def tool_delta_to_pose(
        self,
        ref_joints_deg: Sequence[float],
        dx: float,
        dy: float,
        dz: float,
    ) -> List[float]:
        assert self._robot is not None
        return list(
            self._robot.rm_algo_cartesian_tool(
                list(ref_joints_deg),
                float(dx),
                float(dy),
                float(dz),
                flag=1,
            )
        )

    def try_inverse_kinematics(
        self,
        target_pose: Sequence[float],
        ref_joints_deg: Sequence[float],
        *,
        base_pose: Sequence[float] | None = None,
        deltas: Sequence[float] | None = None,
    ) -> List[float] | None:
        """尝试逆解；失败时可选按单轴回退。"""
        assert self._robot is not None
        q = self._solve_ik(target_pose, ref_joints_deg)
        if q is not None:
            return q

        if base_pose is None or deltas is None:
            return None

        partial_pose = list(base_pose)
        ref = list(ref_joints_deg)
        changed = False
        for idx, delta in enumerate(deltas[:3]):
            if abs(delta) < 1e-9:
                continue
            candidate = list(partial_pose)
            candidate[idx] += delta
            q_axis = self._solve_ik(candidate, ref)
            if q_axis is not None:
                partial_pose = candidate
                ref = q_axis
                changed = True
        return ref if changed else None

    def _solve_ik(
        self,
        target_pose: Sequence[float],
        ref_joints_deg: Sequence[float],
    ) -> List[float] | None:
        assert self._robot is not None
        params = rm_inverse_kinematics_params_t(
            q_in=list(ref_joints_deg),
            q_pose=list(target_pose),
            flag=1,
        )
        ik_ret, q_sol = self._robot.rm_algo_inverse_kinematics(params)
        if ik_ret != 0:
            return None
        return [float(q) for q in q_sol[:6]]

    def try_inverse_kinematics_quat(
        self,
        xyz: Sequence[float],
        quat_xyzw: Sequence[float],
        ref_joints_deg: Sequence[float],
    ) -> List[float] | None:
        """用四元数逆解，避开侧装 ry≈±90° 的欧拉翻解。"""
        assert self._robot is not None
        q_in = [float(v) for v in list(ref_joints_deg[:6])]
        while len(q_in) < 7:
            q_in.append(0.0)
        qw = float(quat_xyzw[3])
        qx = float(quat_xyzw[0])
        qy = float(quat_xyzw[1])
        qz = float(quat_xyzw[2])
        params = rm_inverse_kinematics_params_t(
            q_in=q_in,
            q_pose=[float(xyz[0]), float(xyz[1]), float(xyz[2]), qw, qx, qy, qz],
            flag=0,
        )
        ik_ret, q_sol = self._robot.rm_algo_inverse_kinematics(params)
        if ik_ret != 0:
            return None
        return [float(q) for q in q_sol[:6]]
