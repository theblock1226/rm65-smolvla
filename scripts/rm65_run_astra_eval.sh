#!/usr/bin/env bash
# Astra multitask real-robot eval (stack / desk / slot) with rate-limited follow.
#
# Prerequisites (separate terminals):
#   1) rm_driver already running
#   2) This script starts: record_state_test → play_action_rm65 → policy_server → robot_client
#
# Speed limits (match EE teleop max_joint_step_deg=0.4 + first-run scale):
#   RM65_MAX_JOINT_STEP_DEG=0.4  RM65_SPEED_SCALE=0.3
#
# Usage:
#   scripts/rm65_run_astra_eval.sh                  # stack: prep(ready+开爪) → 闭环
#   TASK_NAME=desk scripts/rm65_run_astra_eval.sh
#   SKIP_PREP=1 scripts/rm65_run_astra_eval.sh      # 跳过就绪位/开爪
#   TASK_PROMPT='...' scripts/rm65_run_astra_eval.sh
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"
CKPT="${ASTRA_CKPT:-${ROOT}/smolvla_rm65_stack_30ep_s1000}"
FRONT_DEV="${FRONT_DEV:-0}"
ASTRA_DEV="${ASTRA_DEV:-2}"
FPS="${FPS:-20}"
SERVER_ADDR="${SERVER_ADDR:-127.0.0.1:8080}"
# Guided RTC: 4 denoise steps in memory, full 24-step chunk, no weighted blend.
RTC_ENABLED="${RTC_ENABLED:-1}"
RTC_NUM_STEPS="${RTC_NUM_STEPS:-4}"
RTC_HORIZON="${RTC_HORIZON:-13}"
RTC_DELAY_STEPS="${RTC_DELAY_STEPS:-14}"
if [[ "${RTC_ENABLED}" == "1" ]]; then
  ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-24}"
  RTC_FLAG=true
else
  ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
  RTC_FLAG=false
fi

export RM65_MAX_JOINT_STEP_DEG="${RM65_MAX_JOINT_STEP_DEG:-0.4}"
export RM65_SPEED_SCALE="${RM65_SPEED_SCALE:-0.3}"
export RM65_POLICY_STATE_DIM="${RM65_POLICY_STATE_DIM:-7}"
export RM65_SAFETY="${RM65_SAFETY:-1}"
export RM65_WATCHDOG_S="${RM65_WATCHDOG_S:-10}"
export RM65_GRIPPER_HYSTERESIS="${RM65_GRIPPER_HYSTERESIS:-1}"
export RM65_GRIPPER_DEBOUNCE="${RM65_GRIPPER_DEBOUNCE:-20}"
export RM65_GRIPPER_MIN_HOLD_S="${RM65_GRIPPER_MIN_HOLD_S:-2.5}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

TASK_NAME="${TASK_NAME:-stack}"
case "${TASK_NAME}" in
  stack)
    DEFAULT_TASK="Stack three cylinder blocks. Grasp the left cylinder."
    ;;
  desk)
    DEFAULT_TASK="Clean up the desk."
    ;;
  slot)
    DEFAULT_TASK="Place the object into the slot."
    ;;
  *)
    DEFAULT_TASK="${TASK_NAME}"
    ;;
esac
TASK_PROMPT="${TASK_PROMPT:-${DEFAULT_TASK}}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

if [[ ! -d "${CKPT}" ]]; then
  echo "缺少 checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "=============================================="
echo " Astra eval  task=${TASK_NAME}"
echo " prompt: ${TASK_PROMPT}"
echo " ckpt:   ${CKPT}"
echo " cams:   camera1=${FRONT_DEV}  camera2=${ASTRA_DEV}"
echo " limit:  max_step=${RM65_MAX_JOINT_STEP_DEG}°  speed_scale=${RM65_SPEED_SCALE}"
echo " fps=${FPS}  state_dim=${RM65_POLICY_STATE_DIM}"
echo " gripper: hysteresis=${RM65_GRIPPER_HYSTERESIS:-1} debounce=${RM65_GRIPPER_DEBOUNCE:-20} min_hold=${RM65_GRIPPER_MIN_HOLD_S:-2.5}s"
echo " 手持急停；Ctrl+C 会自动停掉桥接/server"
echo "=============================================="

cleanup_eval_stack() {
  echo ">>> 清理 eval 进程（play/server/record/client）…"
  # 预览会独占 /dev/video*，不关掉则 OpenCVCamera 报 width=0 / errno 16
  pkill -9 -f dabai_web_viewer.py 2>/dev/null || true
  pkill -9 -f astra_pro_web_viewer.py 2>/dev/null || true
  pkill -9 -f astra2_web_viewer.py 2>/dev/null || true
  pkill -9 -f hikvision_web_viewer.py 2>/dev/null || true
  # 本 shell 的命令行里含有这些文件名。pkill -f 会先杀掉清理进程自己，旧 server 就留在 8080 上。
  docker exec "${CONTAINER}" bash -lc '
    self=$$
    parent=$PPID
    for pat in robot_client.py policy_server.py play_action_rm65.py record_state_test.py rm65_eval_prep.py; do
      ps -eo pid=,args= | grep -F "$pat" | grep -v grep | while read -r pid rest; do
        if [[ "$pid" != "$self" && "$pid" != "$parent" ]]; then
          echo "kill $pid $rest"
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
    done
    sleep 0.5
    left=""
    while read -r pid rest; do
      if [[ -n "$pid" && "$pid" != "$self" && "$pid" != "$parent" ]]; then
        left+="$pid $rest"$'\n'
      fi
    done < <(ps -eo pid=,args= | grep -E "play_action_rm65[.]py|policy_server[.]py|record_state_test[.]py|robot_client[.]py" | grep -v grep || true)
    if [[ -n "$left" ]]; then
      echo "仍有残留:"
      echo "$left"
    else
      echo "cleaned"
    fi
  ' || true
}

# Ctrl+C / 异常退出也必须清栈，避免双 play 抢 movej_canfd 导致狂抖
trap 'cleanup_eval_stack; exit 130' INT TERM
trap 'cleanup_eval_stack' EXIT

# Cleanup leftover eval processes (not rm_driver)
cleanup_eval_stack

# Camera permissions (integer index → /dev/videoN)
docker exec "${CONTAINER}" bash -lc "
  for d in /dev/video${FRONT_DEV} /dev/video${ASTRA_DEV} /dev/video0 /dev/video2; do
    [[ -e \$d ]] && chmod a+rw \$d 2>/dev/null || true
  done
"

COMMON_ENV="
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  cd ${ROOT}
  export PYTHONPATH=${VENDOR}:${ROOT}:${ROOT}/scripts:\${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${VENDOR}/Robotic_Arm/libs:\${LD_LIBRARY_PATH:-}
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export RM65_MAX_JOINT_STEP_DEG=${RM65_MAX_JOINT_STEP_DEG}
  export RM65_SPEED_SCALE=${RM65_SPEED_SCALE}
  export RM65_POLICY_STATE_DIM=${RM65_POLICY_STATE_DIM}
  export RM65_SAFETY=${RM65_SAFETY}
  export RM65_WATCHDOG_S=${RM65_WATCHDOG_S:-10}
  export RM65_INITIAL_GRIPPER_POS=${RM65_INITIAL_GRIPPER_POS:-1.0}
  export RM65_LOG_GRIPPER=${RM65_LOG_GRIPPER:-1}
  export RM65_GRIPPER_LOG=${ROOT}/logs/gripper_chunk.log
  export RM65_GRIPPER_EXEC_LOG=${ROOT}/logs/gripper_exec.log
  export RM65_GRIPPER_HYSTERESIS=${RM65_GRIPPER_HYSTERESIS:-1}
  export RM65_GRIPPER_DEBOUNCE=${RM65_GRIPPER_DEBOUNCE:-20}
  export RM65_GRIPPER_MIN_HOLD_S=${RM65_GRIPPER_MIN_HOLD_S:-2.5}
  export RM65_SKIP_SIMILAR_OBS=${RM65_SKIP_SIMILAR_OBS:-1}
"

# 推理起点用堆叠数据第 0 帧均值，不用 9 月 8 日录制就绪位
STACK_START_POSE="${STACK_START_POSE:-${ROOT}/scripts/rm65_stack_start_pose.json}"
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo ">>> [0/4] eval prep: stack start pose + open gripper (norm=1.0)"
  echo "    pose: ${STACK_START_POSE}"
  docker exec "${CONTAINER}" bash -lc "${COMMON_ENV}
    python3 ${ROOT}/scripts/rm65_eval_prep.py --speed \${PREP_SPEED:-5} --gripper-norm \${PREP_GRIPPER_NORM:-1.0} --pose ${STACK_START_POSE}
  "
else
  echo ">>> [0/4] SKIP_PREP=1，跳过就绪位/开爪"
fi

echo ">>> [1/4] record_state_test (joint feedback + gripper)"
docker exec -d "${CONTAINER}" bash -lc "${COMMON_ENV}
  exec python3 sendrobot/send_robot/record_state_test.py
"
sleep 2

echo ">>> [2/4] play_action_rm65 (rate-limited movej_canfd)"
docker exec -d "${CONTAINER}" bash -lc "${COMMON_ENV}
  exec python3 sendrobot/send_robot/play_action_rm65.py
"
sleep 2

echo ">>> [3/4] policy_server"
docker exec -d "${CONTAINER}" bash -lc "${COMMON_ENV}
  exec python3 lerobot/scripts/server/policy_server.py --host=127.0.0.1 --port=8080 --fps=${FPS}
"
sleep 3

CAMERAS="{
  camera1: {type: opencv, index_or_path: ${FRONT_DEV}, width: 640, height: 480, fps: ${FPS}},
  camera2: {type: opencv, index_or_path: ${ASTRA_DEV}, width: 640, height: 480, fps: ${FPS}}
}"

MOTORS="[left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_joint_6, left_gripper]"

echo ">>> [4/4] robot_client (foreground — Ctrl+C to stop)"
# shellcheck disable=SC2086
docker exec -it "${CONTAINER}" bash -lc "${COMMON_ENV}
  exec python3 lerobot/scripts/server/robot_client.py \\
    --server_address=${SERVER_ADDR} \\
    --robot.type=send_robot \\
    --robot.id=rm65_astra \\
    --robot.use_gripper=true \\
    --robot.motor_names='${MOTORS}' \\
    --robot.teleop=false \\
    --robot.use_kalman_filter=true \\
    --robot.cameras='${CAMERAS}' \\
    --task=\"${TASK_PROMPT}\" \\
    --fps=${FPS} \\
    --policy_type=smolvla \\
    --pretrained_name_or_path=${CKPT} \\
    --policy_device=cuda \\
    --actions_per_chunk=${ACTIONS_PER_CHUNK} \\
    --chunk_size_threshold=0.5 \\
    --aggregate_fn_name=${AGGREGATE_FN:-latest_only} \\
    --rtc_enabled=${RTC_FLAG} \\
    --rtc_num_steps=${RTC_NUM_STEPS} \\
    --rtc_execution_horizon=${RTC_HORIZON} \\
    --rtc_delay_steps=${RTC_DELAY_STEPS} \\
    --verify_robot_cameras=true \\
    --debug_visualize_queue_size=false
"

# EXIT trap 会再清一次；这里去掉重复 pkill 以免双 trap 混乱
echo ">>> client 正常退出"
trap - INT TERM
cleanup_eval_stack
trap - EXIT
echo "done"
