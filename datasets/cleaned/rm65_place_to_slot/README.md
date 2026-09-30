---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - rm65
  - smolvla
---

# rm65_place_to_slot

LeRobot v2.1, 20 fps. RM65 joint-space teleop. Idle / mis-operation segments trimmed.

**Task:** Place three objects into matching slots on the silver card.

Per-frame language is `{overall} {subtask}` (grasp / lift-and-rotate-to-align / insert, per object):

1. Rotate the gripper and grasp the cylinder.
2. Lift and rotate the cylinder to align with the round slot.
3. Insert into the round slot and release.
4. Rotate the gripper and grasp the T-shaped wooden block.
5. Lift and rotate the T-shaped wooden block to align with the T-shaped slot.
6. Insert into the T-shaped slot and release.
7. Rotate the gripper and grasp the blue box.
8. Lift and rotate the blue box to align with the rectangular slot.
9. Insert into the rectangular slot and release.

- Episodes: 52
- Frames: 73839
- Cameras: `observation.images.front`, `observation.images.astra` (480×640)
- Action / state: 6 joints (deg) + gripper (0–1)
