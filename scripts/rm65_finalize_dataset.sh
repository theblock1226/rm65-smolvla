#!/usr/bin/env bash
# 对已有 LeRobot 数据集写入 meta/stats.json + meta/clean_report.json
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
DATA_ROOT="${RECORD_ROOT:-${ROOT}/datasets/rm65_usb_insert}"

docker exec "${CONTAINER}" bash -lc "
  export PYTHONPATH=${ROOT}/vendor:${ROOT}:${ROOT}/scripts:\${PYTHONPATH:-}
  python3 ${ROOT}/scripts/rm65_dataset_finalize.py --root '${DATA_ROOT}' \"\$@\"
" -- "$@"
