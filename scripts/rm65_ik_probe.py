#!/usr/bin/env python3
"""Quick IK probe at current pose."""

from sendrobot.send_robot.rm_ik import RM65Kinematics


def main() -> None:
    kin = RM65Kinematics(ip="192.168.73.18", port=8080)
    kin.connect()
    ref = kin.get_current_joints_deg()
    pose = kin.get_current_pose()
    step = 0.003 * 0.08
    print("joints:", [round(q, 2) for q in ref])
    print("pose:", [round(p, 4) for p in pose])
    print("teleop step m:", step)

    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        target = list(pose)
        target[idx] += step
        q = kin.try_inverse_kinematics(target, ref)
        if q:
            deltas = [round(q[i] - ref[i], 4) for i in range(6)]
            print(f"base +{axis}: OK joint_delta={deltas}")
        else:
            print(f"base +{axis}: FAIL")

    for name, dx, dy, dz in [("tool x", step, 0, 0), ("tool y", 0, step, 0), ("tool z", 0, 0, step)]:
        target = kin.tool_delta_to_pose(ref, dx, dy, dz)
        q = kin.try_inverse_kinematics(target, ref)
        print(f"{name}: {'OK' if q else 'FAIL'}")

    kin.disconnect()


if __name__ == "__main__":
    main()
