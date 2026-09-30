#!/usr/bin/env python3
"""RM65 急停/掉使能恢复：清错 + 全关节上使能 + 可选微小自检运动。"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description="RM65 清错并使能恢复")
    p.add_argument("--ip", default="192.168.73.18")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--probe-move", action="store_true", help="做 ±1° 微小 movej 自检")
    args = p.parse_args()

    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    h = arm.rm_create_robot_arm(args.ip, args.port)
    if h.id < 0:
        print(f"连接失败 handle={h.id}", file=sys.stderr)
        return 1
    print(f"已连接 {args.ip}:{args.port}")

    try:
        code, st = arm.rm_get_current_arm_state()
        print(f"状态 code={code} err={st.get('err')} joints={[round(x,2) for x in st['joint'][:6]]}")
        all_st = arm.rm_get_arm_all_state()[1]
        print(f"使能 joint_en_flag={all_st['joint_en_flag'][:6]}")
        print(f"报错/抱闸 {arm.rm_get_joint_err_flag()}")

        print("停止拖动/示教…")
        arm.rm_stop_drag_teach()
        arm.rm_set_stop_teach()
        arm.rm_set_arm_delete_trajectory()

        print("清除系统/关节错误…")
        print("  clear_system_err =", arm.rm_clear_system_err())
        for j in range(1, 7):
            arm.rm_set_joint_clear_err(j)

        print("全关节上使能…")
        for j in range(1, 7):
            r = arm.rm_set_joint_en_state(j, 1)
            print(f"  J{j} enable -> {r}")
        time.sleep(1.0)

        all_st = arm.rm_get_arm_all_state()[1]
        en = all_st["joint_en_flag"][:6]
        print(f"使能后 joint_en_flag={en}")
        print(f"抱闸 {arm.rm_get_joint_err_flag()}")
        if any(x == 0 for x in en):
            print("警告: 仍有关节未使能。请检查示教器/柜体急停按钮是否按下。", file=sys.stderr)
            return 2

        arm.rm_set_arm_continue()
        print("恢复完成（关节已上使能）。")

        if args.probe_move:
            code, st = arm.rm_get_current_arm_state()
            q = [float(x) for x in st["joint"][:6]]
            t = list(q)
            t[0] = q[0] + 1.0
            print(f"自检 movej: {[round(x,2) for x in q]} -> {[round(x,2) for x in t]}")
            ret = arm.rm_movej(t, 3, 0, 0, 1)
            code2, st2 = arm.rm_get_current_arm_state()
            print(f"movej ret={ret} joints={[round(x,2) for x in st2['joint'][:6]]}")
            if ret != 0:
                print("movej 仍返回失败；若关节角已变化可忽略，否则检查示教器急停。", file=sys.stderr)
                return 3
        return 0
    finally:
        arm.rm_delete_robot_arm()


if __name__ == "__main__":
    raise SystemExit(main())
