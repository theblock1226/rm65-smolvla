#!/usr/bin/env bash
# RM65 关节空间录制：movep 遥操 + 大白 USB + 旧 Astra USB → LeRobotDataset
# USB insert 任务请用 rm65_run_record_usb_insert.sh（cam2 单目 RTSP，不用 Astra）。
# 本脚本给其他任务用：cam1 旧 Astra /dev/video2，预览 8090/8093。
# 依赖：终端1 已启动 rm_driver
# USB 独占：会停掉独立网页预览；录制进程在 8090/8093 继续出预览
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"
USB_INSERT_TASK="Pick up the USB plug and insert it into the port"
TASK="${RECORD_TASK:-${USB_INSERT_TASK}}"
if [[ "${RECORD_FORCE_OLD_ASTRA:-0}" != "1" && "${TASK}" == "${USB_INSERT_TASK}" ]]; then
  echo "USB insert 任务改用 cam2 单目 RTSP，转去 rm65_run_record_usb_insert.sh"
  exec "${ROOT}/scripts/rm65_run_record_usb_insert.sh" "$@"
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

echo ">>> 清理残留遥操/录制进程与预览…"
pkill -f dabai_web_viewer.py 2>/dev/null || true
pkill -f astra_pro_web_viewer.py 2>/dev/null || true
pkill -f astra2_web_viewer.py 2>/dev/null || true
pkill -f hikvision_web_viewer.py 2>/dev/null || true
docker exec "${CONTAINER}" bash -lc '
  for pat in \
    "rm65_movep_teleop.py" \
    "rm65_record_movep.py" \
    "play_action_test.py" \
    "gripper_bridge.py"
  do
    pkill -f "$pat" 2>/dev/null || true
  done
  if [[ -e /dev/input/js0 ]]; then
    chmod a+rw /dev/input/js0 2>/dev/null || true
  fi
  if [[ -e /dev/video0 ]]; then
    chmod a+rw /dev/video0 2>/dev/null || true
  fi
  if [[ -e /dev/video2 ]]; then
    chmod a+rw /dev/video2 2>/dev/null || true
  fi
  echo "清理完成；camera=/dev/video0 astra=/dev/video2 joystick=/dev/input/js0"
' || true

REPO_ID="${RECORD_REPO_ID:-local/rm65_usb_insert_dual}"
DATA_ROOT="${RECORD_ROOT:-${ROOT}/datasets/rm65_usb_insert_dual}"
EPISODE_TIME_S="${RECORD_EPISODE_TIME_S:-120}"

echo ">>> 双相机同步录制（A=开始 B=结束 X=重录 LB/RB=夹爪）"
echo "    手柄: 左左右→Y(反向) 左上下→Z 十字上下→夹爪滚转 十字左右→夹爪偏航"
echo "    repo_id=${REPO_ID}"
echo "    root=${DATA_ROOT}"
echo "    task=${TASK}"
echo "    front=/dev/video0  astra=/dev/video2"
echo "    预览: 8090 大白 / 8093 Astra（录制进程出图）"
echo "    单集最长 ${EPISODE_TIME_S}s；Ctrl+C 退出不保存当前集；已保存集会 finalize"
echo ""

# 相机设备需传入容器：若容器未挂载 /dev/video0，在宿主机直接跑更稳
# 默认：容器内跑；可用 RECORD_HOST=1 改宿主机
if [[ "${RECORD_HOST:-0}" == "1" ]]; then
  export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-cursor}"
  mkdir -p "${XDG_RUNTIME_DIR}"
  export SDL_JOYSTICK_DEVICE="${SDL_JOYSTICK_DEVICE:-/dev/input/js0}"
  export LD_LIBRARY_PATH="${VENDOR}/pygame.libs:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${VENDOR}:${ROOT}:${PYTHONPATH:-}"
  export GAMEPAD_DEADZONE="${GAMEPAD_DEADZONE:-0.15}"
  export GAMEPAD_LEFT_Y_AXIS="${GAMEPAD_LEFT_Y_AXIS:-0}"
  export GAMEPAD_LEFT_Z_AXIS="${GAMEPAD_LEFT_Z_AXIS:-1}"
  export GAMEPAD_INVERT_Y="${GAMEPAD_INVERT_Y:-1}"
  export GAMEPAD_INVERT_Z="${GAMEPAD_INVERT_Z:-1}"
  export GAMEPAD_HAT_SINGLE_AXIS="${GAMEPAD_HAT_SINGLE_AXIS:-1}"
  export GAMEPAD_HAT_RY_FROM="${GAMEPAD_HAT_RY_FROM:-y}"
  export GAMEPAD_HAT_RZ_FROM="${GAMEPAD_HAT_RZ_FROM:-x}"
  export GAMEPAD_HAT_BLOCK_PITCH="${GAMEPAD_HAT_BLOCK_PITCH:-1}"
  export GAMEPAD_GRIPPER_OPEN="${GAMEPAD_GRIPPER_OPEN:-6}"
  export GAMEPAD_GRIPPER_CLOSE="${GAMEPAD_GRIPPER_CLOSE:-7}"
  export TELEOP_FOLLOW="${TELEOP_FOLLOW:-1}"
  export RECORD_FRONT_KEY="${RECORD_FRONT_KEY:-front}"
  export RECORD_ASTRA_KEY="${RECORD_ASTRA_KEY:-astra}"
  export RECORD_ASTRA_CAMERA="${RECORD_ASTRA_CAMERA:-/dev/video2}"
  export RECORD_EPISODE_TIME_S="${EPISODE_TIME_S}"
  export RECORD_EMBODIMENT_ID="${RECORD_EMBODIMENT_ID:-rm65_dahuan}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  python3 "${ROOT}/scripts/rm65_record_movep.py" \
    --go-ready \
    --repo-id "${REPO_ID}" \
    --root "${DATA_ROOT}" \
    --task "${TASK}" \
    --episode-time-s "${EPISODE_TIME_S}" \
    "$@"
else
  docker exec -it "${CONTAINER}" bash -lc "
    source /opt/ros/humble/setup.bash
    source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
    export SDL_VIDEODRIVER=\${SDL_VIDEODRIVER:-dummy}
    export XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/tmp/runtime-cursor}
    mkdir -p \"\${XDG_RUNTIME_DIR}\"
    export SDL_JOYSTICK_DEVICE=\${SDL_JOYSTICK_DEVICE:-/dev/input/js0}
    export LD_LIBRARY_PATH=${VENDOR}/pygame.libs:\${LD_LIBRARY_PATH:-}
    export PYTHONPATH=${VENDOR}:${ROOT}:${ROOT}/scripts:\${PYTHONPATH:-}
    export GAMEPAD_DEADZONE=\${GAMEPAD_DEADZONE:-0.15}
    export GAMEPAD_LEFT_Y_AXIS=\${GAMEPAD_LEFT_Y_AXIS:-0}
    export GAMEPAD_LEFT_Z_AXIS=\${GAMEPAD_LEFT_Z_AXIS:-1}
    export GAMEPAD_INVERT_Y=\${GAMEPAD_INVERT_Y:-1}
    export GAMEPAD_INVERT_Z=\${GAMEPAD_INVERT_Z:-1}
    export GAMEPAD_HAT_SINGLE_AXIS=\${GAMEPAD_HAT_SINGLE_AXIS:-1}
    export GAMEPAD_HAT_RY_FROM=\${GAMEPAD_HAT_RY_FROM:-y}
    export GAMEPAD_HAT_RZ_FROM=\${GAMEPAD_HAT_RZ_FROM:-x}
    export GAMEPAD_HAT_BLOCK_PITCH=\${GAMEPAD_HAT_BLOCK_PITCH:-1}
    export GAMEPAD_GRIPPER_OPEN=\${GAMEPAD_GRIPPER_OPEN:-6}
    export GAMEPAD_GRIPPER_CLOSE=\${GAMEPAD_GRIPPER_CLOSE:-7}
    export TELEOP_FOLLOW=\${TELEOP_FOLLOW:-1}
    export RECORD_FRONT_KEY=\${RECORD_FRONT_KEY:-front}
    export RECORD_ASTRA_KEY=\${RECORD_ASTRA_KEY:-astra}
    export RECORD_ASTRA_CAMERA=\${RECORD_ASTRA_CAMERA:-/dev/video2}
    export RECORD_EPISODE_TIME_S=${EPISODE_TIME_S}
    export RECORD_EMBODIMENT_ID=\${RECORD_EMBODIMENT_ID:-rm65_dahuan}
    export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
    export HF_DATASETS_OFFLINE=\${HF_DATASETS_OFFLINE:-1}
    if [[ ! -e /dev/video0 ]]; then
      echo '容器内无 /dev/video0。请用: RECORD_HOST=1 ${ROOT}/scripts/rm65_run_record.sh' >&2
      exit 2
    fi
    if [[ ! -e /dev/video2 ]]; then
      echo '容器内无 /dev/video2（Astra）。请确认设备已挂入容器。' >&2
      exit 2
    fi
    python3 ${ROOT}/scripts/rm65_record_movep.py --go-ready \
      --repo-id '${REPO_ID}' \
      --root '${DATA_ROOT}' \
      --task '${TASK}' \
      --episode-time-s ${EPISODE_TIME_S} \
      \"\$@\"
  " -- "$@"
fi
