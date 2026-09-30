---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - rm65
  - smolvla
---

# rm65_usb_insert

LeRobot v2.1, 20 fps. RM65 joint-space teleop. Idle / mis-operation segments trimmed.
`observation.images.mono` is paired with joints/front at original index `i+10` (500 ms) to correct RTSP lag.

**Task:** Insert the blue USB module into the white base.

Per-frame language is `{overall} {subtask}`:

1. Rotate the gripper and grasp the blue USB module.
2. Lift and align the USB with the port.
3. Insert into the white base and release.

Episode 4 is labeled `failed` (long stall, kept untrimmed). **Do not use it for SFT.**
`meta/info.json` splits: `train = 0:4,5:200`, `failed = 4:5`. When loading with LeRobot, pass all episode indices except 4.

- Episodes: 200 (199 train + 1 failed)
- Frames: 94944
- Cameras: `observation.images.front`, `observation.images.mono` (480×640)
- Action / state: 6 joints (deg) + gripper (0–1)
