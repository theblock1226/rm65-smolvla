#!/usr/bin/env bash
# Build a GitHub-slim tree for theblock1226/rm65-smolvla (no weights / cleaned data / VLM).
set -euo pipefail
SRC="${SRC:-/home/sdznkj/zkx_ws/sendlerobot}"
DST="${DST:-/home/sdznkj/zkx_ws/rm65-smolvla-gh}"
rm -rf "$DST"
mkdir -p "$DST"

rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.cache/' \
  --exclude '*.egg-info/' --exclude '.pytest_cache/' \
  "$SRC/lerobot/" "$DST/lerobot/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$SRC/sendrobot/" "$DST/sendrobot/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$SRC/scripts/" "$DST/scripts/"
rsync -a "$SRC/docs/" "$DST/docs/"
rsync -a "$SRC/docker/" "$DST/docker/" 2>/dev/null || true

mkdir -p "$DST/third_party"
for d in lerobot_rtc dagger_rm65 RL-Token-SmolVLA; do
  if [[ -d "$SRC/third_party/$d" ]]; then
    rsync -a --exclude '__pycache__/' "$SRC/third_party/$d/" "$DST/third_party/$d/"
  fi
done

# vendor SDK (small) — skip pygame wheels
if [[ -d "$SRC/vendor/Robotic_Arm" ]]; then
  mkdir -p "$DST/vendor"
  rsync -a --exclude '__pycache__/' "$SRC/vendor/Robotic_Arm/" "$DST/vendor/Robotic_Arm/"
fi

# ckpt metadata only (no *.safetensors)
for ckpt in smolvla_rm65_stack_30ep_s1000 smolvla_rm65_usb_100ep_s40000 \
            smolvla_rm65_cleanup_47ep_s40000 smolvla_rm65_place_47ep_s40000; do
  if [[ -d "$SRC/$ckpt" ]]; then
    mkdir -p "$DST/$ckpt"
    rsync -a \
      --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.pth' \
      --exclude '__pycache__/' \
      "$SRC/$ckpt/" "$DST/$ckpt/"
  fi
done

# diag charts (exclude weight backups + live video dump)
mkdir -p "$DST/outputs/diag"
rsync -a \
  --exclude 'smolvla_rm65_stack_30ep_s1000_backup/' \
  --exclude 'smolvla_rm65_stack_30ep_s1000_fp16/' \
  --exclude 'stack_30ep_offline/ep000_live/' \
  --exclude '*.log' \
  "$SRC/outputs/diag/" "$DST/outputs/diag/"

mkdir -p "$DST/outputs"
[[ -d "$SRC/outputs/usb_eval" ]] && rsync -a "$SRC/outputs/usb_eval/" "$DST/outputs/usb_eval/"
[[ -d "$SRC/outputs/profiling" ]] && rsync -a --include '*/' --include '*.json' --exclude '*' \
  "$SRC/outputs/profiling/" "$DST/outputs/profiling/" || true

# small dagger mocks
mkdir -p "$DST/datasets"
for d in dagger_rm65_mock dagger_rm65_mock_e2e dagger_lerobot; do
  [[ -d "$SRC/datasets/$d" ]] && rsync -a --exclude '__pycache__/' "$SRC/datasets/$d/" "$DST/datasets/$d/"
done
# cleaned metadata only (not videos/parquet)
if [[ -d "$SRC/datasets/cleaned" ]]; then
  mkdir -p "$DST/datasets/cleaned"
  rsync -a --include '*/' --include '*.md' --include '*.html' --include '*.json' --include '*.jsonl' \
    --exclude '*' "$SRC/datasets/cleaned/" "$DST/datasets/cleaned/" || true
fi

# root files
for f in pyproject.toml Makefile README.md CODE_OF_CONDUCT.md CONTRIBUTING.md LICENSE start_server.sh start_client.sh; do
  [[ -f "$SRC/$f" ]] && cp -a "$SRC/$f" "$DST/$f"
done

# logs samples (tiny)
mkdir -p "$DST/logs"
shopt -s nullglob
for f in "$SRC"/logs/gripper_*.log "$SRC"/logs/astra_stack_*; do
  [[ -f "$f" ]] || continue
  # keep only small text logs (<2MB)
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if (( sz < 2000000 )); then
    cp -a "$f" "$DST/logs/"
  fi
done
[[ -d "$SRC/logs/rtc_live" ]] && rsync -a --max-size=2m "$SRC/logs/rtc_live/" "$DST/logs/rtc_live/" || true
[[ -d "$SRC/logs/ood_live_probe" ]] && rsync -a --max-size=5m "$SRC/logs/ood_live_probe/" "$DST/logs/ood_live_probe/" || true

du -sh "$DST"
find "$DST" -type f | wc -l
