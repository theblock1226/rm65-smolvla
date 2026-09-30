#!/usr/bin/env bash
# RM65 XY 平面画圆：半径 20cm（需 rm_driver；测试前停掉任何旧桥接，结束后不恢复旧栈）
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
SPEED="${XY_CIRCLE_SPEED:-3}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

echo ">>> 停止旧遥操/桥接进程（禁止自动恢复旧栈）..."
docker exec "${CONTAINER}" bash -lc '
  pkill -f "python3.*play_action_test.py" 2>/dev/null || true
  pkill -f "python3.*record_state_test.py" 2>/dev/null || true
  pkill -f "python3.*rm65_movep_teleop.py" 2>/dev/null || true
  pkill -f "python3 -m lerobot.teleoperate" 2>/dev/null || true
  sleep 0.5
'

echo ">>> XY 平面画圆 (r=20cm, movel 笛卡尔, speed=${SPEED})"
docker exec "${CONTAINER}" bash -lc "
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  export PYTHONPATH=${ROOT}/vendor:${ROOT}:\${PYTHONPATH:-}
  python3 ${ROOT}/scripts/rm65_xy_circle_test.py --speed ${SPEED} \"\$@\"
" -- "$@"
