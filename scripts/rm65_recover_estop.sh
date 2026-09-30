#!/usr/bin/env bash
# 恢复 RM65：停抢 TCP 的进程 → 清错上使能（释放急停/掉使能卡死）
#
# 注意：不要对「当前 bash 命令行里出现过的」字符串做 pkill -f，
# 否则会误杀本脚本的 docker exec shell，表现为只打印第一行就退出。
set -euo pipefail
CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"

echo ">>> 停止可能抢 TCP 的进程..."
# 用 pgrep 取 PID 再 kill，避免 pkill -f 匹配到本 shell 命令行
docker exec "${CONTAINER}" bash -lc '
  kill_pat() {
    local pat="$1"
    local pids
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
      echo "  kill ${pat}: ${pids}"
      # shellcheck disable=SC2086
      kill -9 ${pids} 2>/dev/null || true
    fi
  }
  kill_pat "[r]m65_movep_teleop.py"
  kill_pat "[p]lay_action_rm65.py"
  kill_pat "[p]lay_action_test.py"
  kill_pat "[r]ecord_state_test.py"
  kill_pat "[d]agger_rm65_collect.py"
  kill_pat "[r]obot_client.py"
  sleep 0.5
  echo "  leftover:"
  pgrep -af "record_state_test|play_action_rm65|movep_teleop|dagger_rm65" || echo "  (none)"
' || true

echo ">>> 清错 + 全关节使能..."
docker exec "${CONTAINER}" bash -lc "
  export PYTHONPATH=${ROOT}/vendor:\${PYTHONPATH:-}
  export LD_LIBRARY_PATH=${ROOT}/vendor/Robotic_Arm/libs:\${LD_LIBRARY_PATH:-}
  python3 ${ROOT}/scripts/rm65_recover_estop.py \"\$@\"
" -- "$@"
rc=$?

echo ""
if [[ "${rc}" -eq 0 ]]; then
  echo "恢复成功。若 movej 仍失败：确认只跑一个 rm_driver，再试。"
else
  echo "恢复脚本退出码=${rc}。若提示连接失败：先停掉多余 rm_driver 再重试。" >&2
fi
exit "${rc}"
