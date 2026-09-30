#!/usr/bin/env bash
# Replay the smoothed open-loop trajectory and record both real cameras.
# Frame i of the recording is the live image at command i, aligned with dataset frame i.
# A later run reuses outputs/diag/stack_30ep_offline/ep*_s4_openloop_smooth.npy.
#
#   scripts/rm65_play_policy_openloop.sh
#   EPISODE=0 REPLAY_ONLY=1 scripts/rm65_play_policy_openloop.sh
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
EPISODE="${EPISODE:-0}"
FRONT_DEV="${FRONT_DEV:-0}"
ASTRA_DEV="${ASTRA_DEV:-2}"
EXTRA=()
if [[ "${REPLAY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--replay-only)
fi
if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--generate-only)
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

# Preview servers keep /dev/video* open, and OpenCV then gets no frames.
pkill -9 -f dabai_web_viewer.py 2>/dev/null || true
pkill -9 -f astra_pro_web_viewer.py 2>/dev/null || true
pkill -9 -f astra2_web_viewer.py 2>/dev/null || true
pkill -9 -f hikvision_web_viewer.py 2>/dev/null || true

exec docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  cd ${ROOT}
  for d in /dev/video${FRONT_DEV} /dev/video${ASTRA_DEV}; do
    [[ -e \$d ]] && chmod a+rw \$d 2>/dev/null || true
  done
  export PYTHONPATH=${ROOT}/vendor:${ROOT}:${ROOT}/scripts:\${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${ROOT}/vendor/Robotic_Arm/libs:\${LD_LIBRARY_PATH:-}
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export FRONT_DEV=${FRONT_DEV} ASTRA_DEV=${ASTRA_DEV}
  exec python3 scripts/rm65_play_policy_openloop.py --episode ${EPISODE} ${EXTRA[*]:-}
"
