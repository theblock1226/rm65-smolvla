---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - rm65
  - smolvla
---

# rm65_stacking_blocks

LeRobot v2.1, 20 fps. RM65 joint-space teleop. Idle / mis-operation segments trimmed.

**Task:** Stack three cylinder blocks.

Per-frame language is `{overall} {subtask}`:

1. Grasp the left cylinder.
2. Place it on the middle cylinder and release.
3. Grasp the right cylinder.
4. Place it on the stack and release.

- Episodes: 52
- Frames: 32749
- Cameras: `observation.images.front`, `observation.images.astra` (480×640)
- Action / state: 6 joints (deg) + gripper (0–1)
