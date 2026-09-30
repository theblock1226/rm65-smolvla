#!/usr/bin/env bash
# RM65 DAgger-lite 真机采集（策略自治 + 手柄 clutch 纠偏）
#
# 前置：
#   1) 容器 rm65_dev2 已运行，rm_driver 已启动
#   2) 手柄已插入（/dev/input/js0）；本脚本会 chmod
#   3) 手持急停
#
# 手柄：
#   启动后有 ~8s 标定：按住 Y 或扣住 LT 完成绑定
#   之后按住该键 = 介入纠偏；LB/RB = 夹爪；点 B = 结束本集
#   Ctrl+C = 退出
#
# 若介入无效，先标定/查编号：
#   scripts/rm65_probe_gamepad.sh
#   DAGGER_INTERVENE_BUTTON=3 scripts/rm65_run_dagger_collect.sh

# 用法：
#   scripts/rm65_run_dagger_collect.sh
#   MAX_EPISODES=5 STEPS_PER_EPISODE=600 scripts/rm65_run_dagger_collect.sh
#   SKIP_PREP=1 scripts/rm65_run_dagger_collect.sh
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"
CKPT="${ASTRA_CKPT:-${ROOT}/astra_smolvla}"
FRONT_DEV="${FRONT_DEV:-0}"
ASTRA_DEV="${ASTRA_DEV:-2}"
FPS="${FPS:-20}"
MAX_EPISODES="${MAX_EPISODES:-5}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-600}"
OUT_DIR="${OUT_DIR:-${ROOT}/datasets/dagger_rm65_$(date +%Y%m%d_%H%M%S)}"
TASK_PROMPT="${TASK_PROMPT:-Stack three cylinder blocks. Grasp the left cylinder.}"

# DAgger 纠偏需要更大单步，否则摇杆几乎不动
export RM65_MAX_JOINT_STEP_DEG="${RM65_MAX_JOINT_STEP_DEG:-5.0}"
export RM65_SPEED_SCALE="${RM65_SPEED_SCALE:-1.0}"
export RM65_POLICY_STATE_DIM="${RM65_POLICY_STATE_DIM:-6}"
export RM65_SAFETY="${RM65_SAFETY:-1}"
export RM65_WATCHDOG_S="${RM65_WATCHDOG_S:-10}"
export RM65_GRIPPER_HYSTERESIS="${RM65_GRIPPER_HYSTERESIS:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DAGGER_INTERVENE_BUTTON="${DAGGER_INTERVENE_BUTTON:-3}"
export DAGGER_END_EPISODE_BUTTON="${DAGGER_END_EPISODE_BUTTON:-1}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi
if [[ ! -d "${CKPT}" ]]; then
  echo "缺少 checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "=============================================="
echo " DAgger collect (send_robot + SmolVLA)"
echo " prompt: ${TASK_PROMPT}"
echo " ckpt:   ${CKPT}"
echo " out:    ${OUT_DIR}"
echo " eps=${MAX_EPISODES}  steps/ep=${STEPS_PER_EPISODE}  fps=${FPS}"
echo " limit:  max_step=${RM65_MAX_JOINT_STEP_DEG}°  speed_scale=${RM65_SPEED_SCALE}"
echo " 手柄: 启动后按住 Y/LT 完成标定；之后按住该键介入 | LB/RB=爪 | B=结束本集"
echo "=============================================="

ARM_IP="${RM65_ARM_IP:-192.168.73.18}"
echo ">>> 预检: ping ${ARM_IP}"
if ! ping -c 2 -W 1 "${ARM_IP}" >/dev/null 2>&1; then
  echo "错误: 机械臂 ${ARM_IP} 不通（ping 失败）。" >&2
  echo "  movej 就绪位会失败。请检查：臂上电、网线、急停是否松开、IP 是否仍是 ${ARM_IP}" >&2
  echo "  当前本机: $(ip -br addr show enP1p3s0 2>/dev/null | awk '{print $3}')" >&2
  echo "  恢复后可先: scripts/rm65_recover_estop.sh" >&2
  exit 2
fi
echo "  arm OK"

# Warn on duplicate rm_driver (can make movej_result=False)
n_drv=$(docker exec "${CONTAINER}" bash -lc 'pgrep -c -f "/rm_driver/rm_driver" || true' 2>/dev/null || echo 0)
if [[ "${n_drv}" -gt 1 ]]; then
  echo "警告: 检测到 ${n_drv} 个 rm_driver 进程，建议只留一个后再跑。" >&2
fi

cleanup_stack() {
  echo ">>> 清理 DAgger / eval 进程…"
  pkill -9 -f dabai_web_viewer.py 2>/dev/null || true
  pkill -9 -f astra_pro_web_viewer.py 2>/dev/null || true
  docker exec "${CONTAINER}" bash -lc '
    for pat in \
      "dagger_rm65_collect.py" \
      "robot_client.py" \
      "policy_server.py" \
      "play_action_rm65.py" \
      "record_state_test.py" \
      "rm65_movep_teleop.py" \
      "rm65_eval_prep.py"
    do
      pkill -9 -f "$pat" 2>/dev/null || true
    done
    sleep 0.5
    echo cleaned
  ' || true
}

trap 'cleanup_stack; exit 130' INT TERM
trap 'cleanup_stack' EXIT

cleanup_stack

docker exec "${CONTAINER}" bash -lc "
  for d in /dev/video${FRONT_DEV} /dev/video${ASTRA_DEV} /dev/input/js0; do
    [[ -e \$d ]] && chmod a+rw \$d 2>/dev/null || true
  done
  if [[ -e /dev/input/js0 ]]; then
    echo \"手柄: /dev/input/js0\"
  else
    echo \"警告: /dev/input/js0 不存在 — 请先插上手柄\" >&2
  fi
"

COMMON_ENV="
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  cd ${ROOT}
  export PYTHONPATH=${VENDOR}:${ROOT}:${ROOT}/scripts:\${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${VENDOR}/Robotic_Arm/libs:${VENDOR}/pygame.libs:\${LD_LIBRARY_PATH:-}
  export PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
  export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}
  export RM65_MAX_JOINT_STEP_DEG=${RM65_MAX_JOINT_STEP_DEG}
  export RM65_SPEED_SCALE=${RM65_SPEED_SCALE}
  export RM65_POLICY_STATE_DIM=${RM65_POLICY_STATE_DIM}
  export RM65_SAFETY=${RM65_SAFETY}
  export RM65_WATCHDOG_S=${RM65_WATCHDOG_S}
  export RM65_GRIPPER_HYSTERESIS=${RM65_GRIPPER_HYSTERESIS}
  export SDL_VIDEODRIVER=\${SDL_VIDEODRIVER:-dummy}
  export XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/tmp/runtime-cursor}
  mkdir -p \"\${XDG_RUNTIME_DIR}\"
  export SDL_JOYSTICK_DEVICE=\${SDL_JOYSTICK_DEVICE:-/dev/input/js0}
  export DAGGER_INTERVENE_BUTTON=${DAGGER_INTERVENE_BUTTON}
  export DAGGER_END_EPISODE_BUTTON=${DAGGER_END_EPISODE_BUTTON}
  export TELEOP_SENSITIVITY=\${TELEOP_SENSITIVITY:-0.001}
  export TELEOP_ORIENT_SENSITIVITY=\${TELEOP_ORIENT_SENSITIVITY:-0.003}
  export GAMEPAD_DEADZONE=\${GAMEPAD_DEADZONE:-0.15}
"

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo ">>> [0] eval prep: ready + open gripper"
  docker exec "${CONTAINER}" bash -lc "${COMMON_ENV}
    python3 ${ROOT}/scripts/rm65_eval_prep.py --speed 5 --gripper-norm 1.0
  "
fi

launch_bg() {
  local name="$1"; shift
  local cmd="$1"
  docker exec -d "${CONTAINER}" bash -lc "bash -c $(printf '%q' "${COMMON_ENV}
${cmd}") > $(printf '%q' "${OUT_DIR}/${name}.log") 2>&1"
}

mkdir -p "${OUT_DIR}" 2>/dev/null || {
  echo ">>> OUT_DIR 本机无写权限，改用容器 root 创建…"
  docker exec "${CONTAINER}" bash -lc "mkdir -p $(printf '%q' "${OUT_DIR}") && chown -R $(id -u):$(id -g) $(printf '%q' "${OUT_DIR}")"
}
if [[ ! -d "${OUT_DIR}" ]] || [[ ! -w "${OUT_DIR}" ]]; then
  echo "错误: 无法写入 ${OUT_DIR}" >&2
  exit 1
fi

echo ">>> [1/3] record_state_test"
launch_bg record_state "exec python3 sendrobot/send_robot/record_state_test.py"
sleep 2

echo ">>> [2/3] play_action_rm65"
launch_bg play_action "exec python3 sendrobot/send_robot/play_action_rm65.py"
sleep 2

echo ">>> [3/3] dagger_rm65_collect (前台，接手柄)"
# -it so pygame/joystick stays responsive
set +e
docker exec -it "${CONTAINER}" bash -lc "${COMMON_ENV}
  exec python3 ${ROOT}/scripts/dagger_rm65_collect.py \\
    --backend send_robot --no-send_robot_dry_run \\
    --policy smolvla --ckpt ${CKPT} --device cuda \\
    --camera_set eval \\
    --task $(printf '%q' "${TASK_PROMPT}") \\
    --max_episodes ${MAX_EPISODES} \\
    --steps_per_episode ${STEPS_PER_EPISODE} \\
    --fps ${FPS} \\
    --corrections_only \\
    --intervene_button ${DAGGER_INTERVENE_BUTTON} \\
    --end_episode_button ${DAGGER_END_EPISODE_BUTTON} \\
    --front_dev ${FRONT_DEV} --astra_dev ${ASTRA_DEV} \\
    --speed_scale ${RM65_SPEED_SCALE} \\
    --max_delta_joints ${RM65_MAX_JOINT_STEP_DEG} \\
    --watchdog_timeout_s ${RM65_WATCHDOG_S} \\
    --no-require_deadman \\
    --output_dir ${OUT_DIR}
"
rc=$?
set -e

echo ">>> 输出目录: ${OUT_DIR}"
if [[ -f "${OUT_DIR}/session_report.json" ]]; then
  echo ">>> session_report:"
  cat "${OUT_DIR}/session_report.json"
fi
exit "${rc}"
