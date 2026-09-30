---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - rm65
  - smolvla
---

# rm65_cleanup_the_desk

LeRobot v2.1, 20 fps. RM65 joint-space teleop. Idle / mis-operation segments trimmed.

**Task:** Put three objects from the desk into the box.

Per-frame language is `{overall} {subtask}` (left to right, not object identity):

1. Grasp the left object.
2. Place it in the box and release.
3. Grasp the middle object.
4. Place it in the box and release.
5. Grasp the right object.
6. Place it in the box and release.

- Episodes: 52
- Frames: 45717
- Cameras: `observation.images.front`, `observation.images.astra` (480×640)
- Action / state: 6 joints (deg) + gripper (0–1)
