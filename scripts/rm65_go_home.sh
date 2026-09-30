#!/usr/bin/env bash
# 缓慢将 RM65 移回控制器初始位姿（movej，block=true）
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"
# 越小越慢；默认 2，约为常见 movej speed=10 的 1/5
HOME_SPEED="${HOME_SPEED:-2}"
RM_IP="${RM_IP:-192.168.73.18}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

echo ">>> 停止遥操与 CANFD 透传（避免冲突）..."
docker exec "${CONTAINER}" bash -lc "
  ps -eo pid=,args= | while read -r pid cmd; do
    case \"\$cmd\" in
      python3*play_action_test.py*) kill -TERM \"\$pid\" 2>/dev/null || true ;;
      python3*lerobot.teleoperate*) kill -TERM \"\$pid\" 2>/dev/null || true ;;
    esac
  done
  sleep 0.5
  echo '遥操/透传已停止'
"

echo ">>> 缓慢回到初始位姿 (speed='"${HOME_SPEED}"')..."
docker exec "${CONTAINER}" bash -lc "
  source /opt/ros/humble/setup.bash
  source /home/sdznkj/zkx_ws/ros2_ws/install/setup.bash
  export LD_LIBRARY_PATH=${VENDOR}/pygame.libs:\${LD_LIBRARY_PATH:-}
  export PYTHONPATH=${VENDOR}:${ROOT}:\${PYTHONPATH:-}
  python3 ${ROOT}/scripts/rm65_go_home.py --ip ${RM_IP} --speed ${HOME_SPEED} \"\$@\"
" -- "$@"

echo ">>> 完成。可重新启动桥接栈与遥操。"
