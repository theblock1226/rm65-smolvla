#!/usr/bin/env bash
# 笛卡尔 movep_canfd 遥操（base.py 基线）
# --print-only：只测手柄（不动臂、不 movej）
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

PRINT_ONLY=false
for a in "$@"; do
  if [[ "$a" == "--print-only" ]]; then
    PRINT_ONLY=true
  fi
done

echo ">>> 清理残留遥操进程..."
docker exec "${CONTAINER}" bash -lc '
  pkill -f rm65_movep_teleop.py 2>/dev/null || true
  if [[ -e /dev/input/js0 ]]; then
    chmod a+rw /dev/input/js0 2>/dev/null || true
    echo "手柄: /dev/input/js0"
  else
    echo "警告: /dev/input/js0 不存在" >&2
  fi
' || true

if [[ "${PRINT_ONLY}" == "true" ]]; then
  EXTRA_ARGS=""
  echo ">>> 只测手柄 (--print-only)"
elif [[ "${TELEOP_NO_READY:-0}" == "1" ]]; then
  EXTRA_ARGS="--follow"
  echo ">>> 实机遥操（跳过就绪位）"
else
  EXTRA_ARGS="--go-ready --follow"
  echo ">>> 实机遥操（--go-ready + --follow）"
  echo "    就绪位: ${ROOT}/scripts/rm65_ready_pose.json"
fi

echo "    映射: 左左右→Y；左上下→Z；右上下→X；右左右→ry"
echo "          十字上下→夹爪滚转；十字左右→夹爪偏航"
echo ""

docker exec -it "${CONTAINER}" bash -lc "
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  export SDL_VIDEODRIVER=\${SDL_VIDEODRIVER:-dummy}
  export XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/tmp/runtime-cursor}
  mkdir -p \"\${XDG_RUNTIME_DIR}\"
  export SDL_JOYSTICK_DEVICE=\${SDL_JOYSTICK_DEVICE:-/dev/input/js0}
  export LD_LIBRARY_PATH=${VENDOR}/pygame.libs:\${LD_LIBRARY_PATH:-}
  export PYTHONPATH=${VENDOR}:${ROOT}:\${PYTHONPATH:-}
  export GAMEPAD_DEADZONE=\${GAMEPAD_DEADZONE:-0.15}
  export TELEOP_SENSITIVITY=\${TELEOP_SENSITIVITY:-0.001}
  export TELEOP_ORIENT_SENSITIVITY=\${TELEOP_ORIENT_SENSITIVITY:-0.003}
  export GAMEPAD_HAT_SINGLE_AXIS=\${GAMEPAD_HAT_SINGLE_AXIS:-1}
  export GAMEPAD_HAT_BLOCK_PITCH=\${GAMEPAD_HAT_BLOCK_PITCH:-1}
  python3 ${ROOT}/scripts/rm65_movep_teleop.py ${EXTRA_ARGS} \"\$@\"
" -- "$@"
