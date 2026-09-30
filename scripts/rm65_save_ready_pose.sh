#!/usr/bin/env bash
# 把当前机械臂关节角存为遥操就绪位（可反复执行覆盖）
# 用法：
#   1) 先遥操/示教摆到合适姿态（或 TELEOP_NO_READY=1 启动后摆好）
#   2) Ctrl+C 停掉遥操
#   3) 运行本脚本
#   4) 再开遥操，--go-ready 会去新位姿
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"
RM_IP="${RM_IP:-192.168.73.18}"

run_on_host() {
  echo ">>> 宿主机执行（容器 ${CONTAINER} 未运行）"
  pkill -f rm65_movep_teleop.py 2>/dev/null || true
  pkill -f rm65_record_movep.py 2>/dev/null || true
  sleep 0.5
  export PYTHONPATH="${VENDOR}:${ROOT}:${PYTHONPATH:-}"
  echo ">>> 读取当前关节并写入 rm65_ready_pose.json…"
  python3 "${ROOT}/scripts/rm65_save_ready_pose.py" --ip "${RM_IP}" "$@"
  echo ">>> 查看配置:"
  python3 "${ROOT}/scripts/rm65_save_ready_pose.py" --show
}

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo ">>> 停止遥操/录制，避免与 SDK 抢 TCP…"
  docker exec "${CONTAINER}" bash -lc '
    pkill -f rm65_movep_teleop.py 2>/dev/null || true
    pkill -f rm65_record_movep.py 2>/dev/null || true
    sleep 0.5
    echo 已清理
  ' || true

  echo ">>> 读取当前关节并写入 rm65_ready_pose.json…"
  docker exec "${CONTAINER}" bash -lc "
    export PYTHONPATH=${VENDOR}:${ROOT}:\${PYTHONPATH:-}
    python3 ${ROOT}/scripts/rm65_save_ready_pose.py --ip ${RM_IP} \"\$@\"
  " -- "$@"

  echo ">>> 查看配置:"
  docker exec "${CONTAINER}" bash -lc "
    export PYTHONPATH=${VENDOR}:${ROOT}:\${PYTHONPATH:-}
    python3 ${ROOT}/scripts/rm65_save_ready_pose.py --show
  "
else
  echo "警告: 容器 ${CONTAINER} 未运行，改在宿主机直连机械臂 SDK。" >&2
  run_on_host "$@"
fi
