#!/usr/bin/env bash
# 容器内若存在多个 rm_driver，保留最新一组，杀掉旧实例（避免 CANFD 指令冲突）
set -euo pipefail

CONTAINER="${RM65_CONTAINER:-rm65_dev2}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行" >&2
  exit 1
fi

mapfile -t lines < <(
  docker exec "${CONTAINER}" bash -lc "ps -eo pid=,lstart=,args= | grep rm_driver/lib/rm_driver/rm_driver | grep -v grep || true"
)

count=${#lines[@]}
if [[ "${count}" -le 1 ]]; then
  echo "rm_driver 实例数: ${count}（无需清理）"
  exit 0
fi

echo "检测到 ${count} 个 rm_driver，保留最新进程，终止其余…"
# 按 ps lstart 排序后保留最后一个
to_kill=()
while IFS= read -r line; do
  pid=$(echo "${line}" | awk '{print $1}')
  to_kill+=("${pid}")
done < <(printf '%s\n' "${lines[@]}" | head -n $((count - 1)))

for pid in "${to_kill[@]}"; do
  echo "  kill ${pid}"
  docker exec "${CONTAINER}" kill -TERM "${pid}" 2>/dev/null || true
done

sleep 1
remaining=$(docker exec "${CONTAINER}" bash -lc "pgrep -cf rm_driver/lib/rm_driver/rm_driver || true")
echo "清理后 rm_driver 实例数: ${remaining}"
