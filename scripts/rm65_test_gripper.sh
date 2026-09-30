#!/usr/bin/env bash
# 大寰夹爪单独测试：初始化 -> 全开(1000) -> 全合(0) -> 再开
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
RM_IP="${RM_IP:-192.168.73.18}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

echo ">>> 大寰夹爪 Modbus 测试（请目视夹爪）..."
docker exec "${CONTAINER}" bash -lc "
  export PYTHONPATH=${ROOT}/vendor:${ROOT}:\${PYTHONPATH:-}
  python3 ${ROOT}/sendrobot/send_robot/rm65_gripper_test.py --ip ${RM_IP}
"
