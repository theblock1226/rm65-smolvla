#!/usr/bin/env bash
# USB insert 专用录制。相机编号：
#   cam0  front  /dev/video0              大白（本任务启用）
#   cam1  astra  /dev/video2              旧 Astra Pro（本任务停用）
#   cam2  mono   RTSP 192.168.73.102      单目子码流 800x448@~20fps（仅本任务启用）
# 其他任务请用 rm65_run_record.sh（cam0+cam1）。
# 依赖：终端1 已启动 rm_driver
# 预览：8090 大白 / 8094 cam2；落盘缩到 640x480 @20Hz。
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"
USB_INSERT_TASK="Pick up the USB plug and insert it into the port"
CAM2_ID="cam2"
CAM2_KEY="mono"
DEFAULT_RTSP="rtsp://admin:@192.168.73.102:554/user=admin&password=&channel=1&stream=1.sdp"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

TASK="${RECORD_TASK:-${USB_INSERT_TASK}}"
if [[ "${TASK}" != "${USB_INSERT_TASK}" ]]; then
  echo "本脚本只用于 USB insert 任务（启用 ${CAM2_ID}/${CAM2_KEY} 单目）。" >&2
  echo "当前 task=${TASK}" >&2
  echo "其他任务请用: ${ROOT}/scripts/rm65_run_record.sh" >&2
  exit 2
fi

CAM2_RTSP="${RECORD_CAM2_RTSP:-${RECORD_ASTRA_RTSP:-${DEFAULT_RTSP}}}"

echo ">>> 清理残留遥操/录制进程与预览（含 hikvision RTSP，避免双客户端卡顿）…"
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
  echo "清理完成；cam0=/dev/video0 cam2=RTSP（cam1 /dev/video2 不打开） joystick=/dev/input/js0"
' || true

REPO_ID="${RECORD_REPO_ID:-local/rm65_usb_insert_dual}"
DATA_ROOT="${RECORD_ROOT:-${ROOT}/datasets/rm65_usb_insert_dual}"
EPISODE_TIME_S="${RECORD_EPISODE_TIME_S:-120}"
RTSP_SHOW="${CAM2_RTSP#*@}"

echo ">>> USB insert 双相机同步录制（cam0 大白 + cam2 单目，停用 cam1 旧 Astra）"
echo "    手柄: 左左右→Y(反向) 左上下→Z 十字上下→夹爪滚转 十字左右→夹爪偏航"
echo "    repo_id=${REPO_ID}"
echo "    root=${DATA_ROOT}"
echo "    task=${TASK}"
echo "    cam0 front=/dev/video0"
echo "    cam2 ${CAM2_KEY}=RTSP ${RTSP_SHOW}"
echo "    cam1 astra=/dev/video2 停用"
echo "    预览: 8090 cam0 / 8094 cam2（录制进程出图，落盘 640x480 @20Hz）"
echo "    单集最长 ${EPISODE_TIME_S}s；Ctrl+C 退出不保存当前集；已保存集会 finalize"
echo ""

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
  export RECORD_ASTRA_KEY="${CAM2_KEY}"
  export RECORD_ASTRA_RTSP="${CAM2_RTSP}"
  export RECORD_ASTRA_PREVIEW_PORT="${RECORD_ASTRA_PREVIEW_PORT:-8094}"
  export RECORD_ASTRA_PREVIEW_TITLE="${RECORD_ASTRA_PREVIEW_TITLE:-cam2 mono RTSP 800x448}"
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
    --astra-key "${CAM2_KEY}" \
    --astra-rtsp "${CAM2_RTSP}" \
    --astra-preview-port "${RECORD_ASTRA_PREVIEW_PORT}" \
    --astra-preview-title "${RECORD_ASTRA_PREVIEW_TITLE}" \
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
    export RECORD_ASTRA_KEY='${CAM2_KEY}'
    export RECORD_ASTRA_RTSP='${CAM2_RTSP}'
    export RECORD_ASTRA_PREVIEW_PORT=\${RECORD_ASTRA_PREVIEW_PORT:-8094}
    export RECORD_ASTRA_PREVIEW_TITLE=\${RECORD_ASTRA_PREVIEW_TITLE:-'cam2 mono RTSP 800x448'}
    export RECORD_EPISODE_TIME_S=${EPISODE_TIME_S}
    export RECORD_EMBODIMENT_ID=\${RECORD_EMBODIMENT_ID:-rm65_dahuan}
    export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
    export HF_DATASETS_OFFLINE=\${HF_DATASETS_OFFLINE:-1}
    if [[ ! -e /dev/video0 ]]; then
      echo '容器内无 /dev/video0。请用: RECORD_HOST=1 ${ROOT}/scripts/rm65_run_record_usb_insert.sh' >&2
      exit 2
    fi
    python3 ${ROOT}/scripts/rm65_record_movep.py --go-ready \
      --repo-id '${REPO_ID}' \
      --root '${DATA_ROOT}' \
      --task '${TASK}' \
      --episode-time-s ${EPISODE_TIME_S} \
      --astra-key '${CAM2_KEY}' \
      --astra-rtsp '${CAM2_RTSP}' \
      --astra-preview-port \${RECORD_ASTRA_PREVIEW_PORT} \
      --astra-preview-title \"\${RECORD_ASTRA_PREVIEW_TITLE}\" \
      \"\$@\"
  " -- "$@"
fi
