#!/usr/bin/env python3
"""读取 RM65 当前关节角（度），用于配置 rm65_home_joints.yaml。"""

from sendrobot.send_robot.rm_ik import RM65Kinematics


def main() -> None:
    kin = RM65Kinematics()
    kin.connect()
    joints = kin.get_current_joints_deg()
    pose = kin.get_current_pose()
    kin.disconnect()

    rounded = [round(q, 3) for q in joints]
    print("当前关节角 (deg):", rounded)
    print("当前末端 pose [x,y,z,rx,ry,rz]:", [round(v, 4) for v in pose])
    print()
    print("请将下列内容写入 scripts/rm65_home_joints.yaml:")
    print(f"home_joints_deg: {rounded}")


if __name__ == "__main__":
    main()
