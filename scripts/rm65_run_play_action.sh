#!/usr/bin/env bash
# Start RM65 joint follow bridge (sent_actions → movej_canfd, rate-limited).
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
export RM65_MAX_JOINT_STEP_DEG="${RM65_MAX_JOINT_STEP_DEG:-0.4}"
export RM65_SPEED_SCALE="${RM65_SPEED_SCALE:-0.3}"

echo ">>> play_action_rm65 max_step=${RM65_MAX_JOINT_STEP_DEG}° speed_scale=${RM65_SPEED_SCALE}"

docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  cd ${ROOT}
  export PYTHONPATH=${VENDOR}:${ROOT}:\${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${VENDOR}/Robotic_Arm/libs:\${LD_LIBRARY_PATH:-}
  export RM65_MAX_JOINT_STEP_DEG=${RM65_MAX_JOINT_STEP_DEG}
  export RM65_SPEED_SCALE=${RM65_SPEED_SCALE}
  exec python3 sendrobot/send_robot/play_action_rm65.py
"
