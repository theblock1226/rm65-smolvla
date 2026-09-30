#!/usr/bin/env bash
# 汇总最近一次评测的夹爪 chunk / 执行 log
set -euo pipefail
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
CHUNK="${RM65_GRIPPER_LOG:-${ROOT}/logs/gripper_chunk.log}"
EXEC="${RM65_GRIPPER_EXEC_LOG:-${ROOT}/logs/gripper_exec.log}"

echo "=== chunk log: ${CHUNK} ==="
if [[ -f "${CHUNK}" ]]; then
  echo "lines=$(wc -l < "${CHUNK}")"
  echo "--- last 20 ---"
  tail -20 "${CHUNK}"
  python3 - <<PY
from pathlib import Path
p=Path("${CHUNK}")
edges=[]; means=[]
for line in p.read_text().splitlines():
  if not line.startswith("chunk"): continue
  parts=dict(x.split("=",1) for x in line.split("\t")[1:] if "=" in x)
  if "edges" in parts: edges.append(int(float(parts["edges"])))
  if "mean" in parts: means.append(float(parts["mean"]))
print(f"chunks={len(edges)} edges_per_chunk: mean={sum(edges)/len(edges) if edges else 0:.2f} max={max(edges) if edges else 0}")
print(f"grip_mean across chunks: mean={sum(means)/len(means) if means else 0:.3f}")
# demo reference: stacking ep typically ~1 open→close edge in first approach
print("参考: 数采接近段通常 edges≈0~1（开着接近），整集开合次数很少")
PY
else
  echo "(missing — 跑一轮 eval 后生成)"
fi

echo ""
echo "=== exec log: ${EXEC} ==="
if [[ -f "${EXEC}" ]]; then
  echo "lines=$(wc -l < "${EXEC}")"
  echo "--- last 20 ---"
  tail -20 "${EXEC}"
  python3 - <<PY
from pathlib import Path
p=Path("${EXEC}")
edges=0
for line in p.read_text().splitlines():
  if "edges=" in line:
    edges=int(line.split("edges=")[-1].split()[0])
print(f"final open/close edges (threshold 0.5) = {edges}")
print("参考: 数采单次抓取通常 1 次合 + 1 次开，edges 很小")
PY
else
  echo "(missing)"
fi
