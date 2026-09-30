#!/usr/bin/env python3
"""RM65 DAgger-lite collector (Path B).

HG-DAgger clutch: hold intervene button → record expert actions; release → policy.

Mock dry-run (no robot / no VLA)::

    docker exec rm65_dev2 bash -lc 'cd /home/sdznkj/zkx_ws/sendlerobot && \\
      PYTHONPATH=. python3 scripts/dagger_rm65_collect.py \\
        --backend mock --policy mock --max_episodes 3 --steps_per_episode 12 \\
        --intervene_steps 4,5,6 --corrections_only \\
        --output_dir datasets/dagger_rm65_mock'

Real robot + SmolVLA (prefer the wrapper script)::

    scripts/rm65_run_dagger_collect.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="RM65 DAgger-lite collect")
    p.add_argument("--backend", choices=["mock", "send_robot"], default="mock")
    p.add_argument("--policy", choices=["mock", "smolvla"], default="mock")
    p.add_argument(
        "--camera_set",
        default="stack",
        choices=["desk", "slot", "stack", "eval", "stack_eval", "usb", "default"],
    )
    p.add_argument("--task", type=str, default="Stack three cylinder blocks. Grasp the left cylinder.")
    p.add_argument("--max_episodes", type=int, default=3)
    p.add_argument("--steps_per_episode", type=int, default=12)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument(
        "--corrections_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only save frames/episodes that contain human corrections (default)",
    )
    p.add_argument(
        "--intervene_steps",
        type=str,
        default="4,5,6",
        help="Mock: step indices with forced hold (per episode)",
    )
    p.add_argument(
        "--safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SafeActionGate on adapter send_action",
    )
    p.add_argument(
        "--require_deadman",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require deadman button (default: off for mock / dagger; on only if set)",
    )
    p.add_argument("--speed_scale", type=float, default=0.3)
    p.add_argument("--max_delta_joints", type=float, default=2.0)
    p.add_argument("--max_delta_gripper", type=float, default=0.08)
    p.add_argument("--watchdog_timeout_s", type=float, default=0.25)
    p.add_argument("--output_dir", type=str, default="datasets/dagger_rm65_mock")
    p.add_argument("--seed", type=int, default=0)
    # Real robot
    p.add_argument(
        "--send_robot_dry_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send_robot without ROS/hardware (default True for offline tests)",
    )
    p.add_argument("--ckpt", type=str, default=str(ROOT / "astra_smolvla"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--intervene_button", type=int, default=3, help="Hold to correct (default Y=3)")
    p.add_argument("--end_episode_button", type=int, default=1, help="Press to end episode (default B=1)")
    p.add_argument("--deadman_button", type=int, default=5)
    p.add_argument("--front_dev", type=int, default=0)
    p.add_argument("--astra_dev", type=int, default=2)
    p.add_argument("--rm_ip", type=str, default="192.168.73.18")
    p.add_argument("--rm_port", type=int, default=8080)
    return p.parse_args()


def _make_send_robot(args):
    import rclpy
    from lerobot.common.cameras.dummy import DummyCameraConfig
    from lerobot.common.cameras.opencv import OpenCVCameraConfig
    from sendrobot.send_robot import SendRobot
    from sendrobot.send_robot.config_send_robot import SendRobotConfig

    if not rclpy.ok():
        rclpy.init()

    cams = {
        "camera1": OpenCVCameraConfig(
            index_or_path=args.front_dev, width=640, height=480, fps=int(args.fps)
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=args.astra_dev, width=640, height=480, fps=int(args.fps)
        ),
        "camera3": DummyCameraConfig(width=256, height=256, fps=int(args.fps)),
        "empty_camera_0": DummyCameraConfig(width=640, height=480, fps=int(args.fps)),
    }
    cfg = SendRobotConfig(
        id="rm65_dagger",
        cameras=cams,
        motor_names=[
            "left_joint_1",
            "left_joint_2",
            "left_joint_3",
            "left_joint_4",
            "left_joint_5",
            "left_joint_6",
            "left_gripper",
        ],
        use_gripper=True,
        teleop=False,
        use_kalman_filter=True,
        initial_gripper_pos=float(os.environ.get("RM65_INITIAL_GRIPPER_POS", "1.0")),
    )
    robot = SendRobot(cfg)
    robot.connect()
    # Wait for first joint state from record_state_test
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            rclpy.spin_once(robot, timeout_sec=0.05)
        except Exception:
            pass
        obs = robot.get_observation()
        if obs and any(k.startswith("left_joint") for k in obs):
            logger.info("SendRobot joint feedback OK")
            return robot
        time.sleep(0.05)
    raise RuntimeError("SendRobot got no joint states — is record_state_test.py running?")


def _build_policy(args):
    from lerobot.common.policies.dagger_rm65.smolvla_actor import MockPolicy, SmolVLAActor

    if args.policy == "mock":
        return MockPolicy(seed=args.seed)
    return SmolVLAActor(args.ckpt, device=args.device, task=args.task)


def main() -> int:
    args = parse_args()
    from lerobot.common.policies.dagger_rm65 import (
        DAggerLiteController,
        save_episode_bundle,
        write_session_report,
    )
    from lerobot.common.policies.smolvla_rlt.robot_adapter import (
        MockRM65Adapter,
        SendRobotRM65Adapter,
    )
    from lerobot.common.policies.smolvla_rlt.safety import SafetyConfig

    # Real robot: prefer eval camera keys to match astra ckpt
    if args.backend == "send_robot" and not args.send_robot_dry_run and args.camera_set == "stack":
        args.camera_set = "eval"
        logger.info("camera_set auto → eval (camera1/camera2) for real send_robot")

    require_deadman = args.require_deadman
    if require_deadman is None:
        require_deadman = False

    intervene_steps = {int(x) for x in args.intervene_steps.split(",") if x.strip() != ""}
    safety_cfg = SafetyConfig(
        enabled=bool(args.safety) and not (args.backend == "send_robot" and not args.send_robot_dry_run),
        require_deadman=bool(require_deadman),
        max_delta_joints=args.max_delta_joints,
        max_delta_gripper=args.max_delta_gripper,
        speed_scale=args.speed_scale,
        watchdog_timeout_s=args.watchdog_timeout_s,
        action_dim=7,
    )

    teleop = None
    hw_robot = None
    if args.backend == "mock":
        robot = MockRM65Adapter(
            camera_set=args.camera_set,
            intervene_schedule=intervene_steps,
            seed=args.seed,
            safety=safety_cfg,
        )
    elif args.backend == "send_robot":
        if args.send_robot_dry_run:
            robot = SendRobotRM65Adapter(
                camera_set=args.camera_set,
                intervene_button=args.intervene_button,
                deadman_button=args.deadman_button,
                dry_run=True,
                seed=args.seed,
                safety=safety_cfg,
                require_deadman=require_deadman,
            )
        else:
            from lerobot.common.policies.dagger_rm65.gamepad_teleop import GamepadClutchTeleop

            teleop = GamepadClutchTeleop(
                intervene_button=args.intervene_button,
                end_episode_button=args.end_episode_button,
                ip=args.rm_ip,
                port=args.rm_port,
            )
            teleop.connect()
            hw_robot = _make_send_robot(args)
            robot = SendRobotRM65Adapter(
                camera_set=args.camera_set,
                intervene_button=args.intervene_button,
                deadman_button=args.deadman_button,
                dry_run=False,
                robot=hw_robot,
                human_action_provider=lambda: teleop.human_action(),
                seed=args.seed,
                safety=safety_cfg,
                require_deadman=require_deadman,
            )
            # Use gamepad clutch for intervene poll (shared stick/joystick)
            robot.intervention = teleop
            robot.deadman.backend = "mock"  # deadman off unless --require_deadman
            if require_deadman:
                robot.deadman.backend = "gamepad"
    else:
        raise SystemExit(f"unknown backend={args.backend}")

    policy = _build_policy(args)
    ctrl = DAggerLiteController(corrections_only=args.corrections_only, task=args.task)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robot.connect()
    target_dt = 1.0 / args.fps
    sources_seen: set[str] = set()
    saved = 0

    logger.info(
        "DAgger start backend=%s policy=%s dry_run=%s fps=%g intervene_btn=%d",
        args.backend,
        args.policy,
        getattr(args, "send_robot_dry_run", True),
        args.fps,
        args.intervene_button,
    )
    if args.backend == "send_robot" and not args.send_robot_dry_run:
        logger.info(
            "手柄: 按住 b%d(Y) 介入纠偏 | LB/RB 开合爪 | 点 b%d(B) 结束本集 | Ctrl+C 退出",
            args.intervene_button,
            args.end_episode_button,
        )

    try:
        for ep in range(1, args.max_episodes + 1):
            logger.info("=" * 60)
            logger.info("DAgger episode %d/%d", ep, args.max_episodes)
            if hasattr(robot, "reset_episode"):
                robot.reset_episode()
            if teleop is not None:
                teleop.reset()
            if hasattr(policy, "reset"):
                policy.reset()
            ctrl.reset_episode()

            prev_holding = False
            for step in range(args.steps_per_episode):
                t0 = time.time()
                # Spin ROS so joint callbacks keep flowing
                if hw_robot is not None:
                    try:
                        import rclpy

                        rclpy.spin_once(hw_robot, timeout_sec=0.0)
                    except Exception:
                        pass

                if args.backend == "send_robot" and args.send_robot_dry_run:
                    robot.set_force_intervention(step in intervene_steps)

                obs = robot.get_observation()
                holding = robot.poll_intervention()

                # Falling edge of intervene: flush SmolVLA action queue
                if prev_holding and not holding and hasattr(policy, "reset"):
                    policy.reset()

                if holding:
                    # Keep last autonomous action for logging; don't burn GPU / queue
                    policy_action = getattr(robot, "_last_policy_action", obs.state.copy())
                else:
                    policy_action = policy.act(obs.state, obs.images)
                    robot._last_policy_action = policy_action.copy()
                prev_holding = holding

                human = None
                if holding:
                    if teleop is not None:
                        human = teleop.human_action(obs.state)
                    else:
                        human = robot.human_action()

                action, source = ctrl.on_step(
                    state=obs.state,
                    policy_action=policy_action,
                    human_action=human,
                    images=obs.images,
                    holding=holding,
                )
                sources_seen.add(source)
                robot.send_action(action)

                if step % max(1, int(args.fps)) == 0:
                    logger.info(
                        "  step %d | %s | hold=%s | J2=%.1f J3=%.1f G=%.2f",
                        step,
                        source,
                        holding,
                        float(obs.state[1]),
                        float(obs.state[2]),
                        float(obs.state[6]) if obs.state.shape[0] > 6 else -1.0,
                    )

                if teleop is not None and teleop.consume_end_episode():
                    logger.info("  early end episode at step %d", step)
                    break

                sleep = target_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(min(sleep, 0.05))

            # If last frame was correcting→release mid-episode, reset policy queue for next ep
            if hasattr(policy, "reset"):
                policy.reset()

            finished = ctrl.end_episode()
            if finished is not None:
                path = save_episode_bundle(finished, out_dir, saved)
                saved += 1
                logger.info(
                    "  saved %s frames=%d intervention=%d",
                    path,
                    len(finished.frames),
                    finished.n_intervention,
                )
            else:
                logger.info("  episode not saved (no corrections?)")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if hw_robot is not None:
            try:
                hw_robot.disconnect()
            except Exception:
                pass
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    report = write_session_report(
        out_dir,
        ctrl.completed_episodes,
        extra={
            "sources_seen": sorted(sources_seen),
            "corrections_only": args.corrections_only,
            "backend": args.backend,
            "policy": args.policy,
            "send_robot_dry_run": bool(getattr(args, "send_robot_dry_run", True)),
            "pass_has_human": "HUMAN" in sources_seen,
            "pass_saved_episodes": saved > 0,
            "pass_intervention_frames": sum(e.n_intervention for e in ctrl.completed_episodes) > 0,
        },
    )
    report["pass"] = (
        report["pass_has_human"]
        and report["pass_saved_episodes"]
        and report["pass_intervention_frames"]
    )
    (out_dir / "session_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["pass"]:
        logger.info("PASS DAgger-lite collect")
        return 0
    # Real sessions may Ctrl+C before a full corrected episode — still exit 0 if any data
    if saved > 0:
        logger.warning("Partial session: saved=%d (not all pass flags)", saved)
        return 0
    logger.error("FAIL DAgger-lite collect")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
