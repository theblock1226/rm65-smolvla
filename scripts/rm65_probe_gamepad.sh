#!/usr/bin/env bash
# 打印 GameSir / 手柄按键编号：按住各键看 DOWN 行
set -euo pipefail
CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROOT="/home/sdznkj/zkx_ws/sendlerobot"
VENDOR="${ROOT}/vendor"

echo ">>> 8 秒内依次按住: Y / A / B / X / LB / RB / LT / RT"
docker exec -it "${CONTAINER}" bash -lc "
  export SDL_VIDEODRIVER=\${SDL_VIDEODRIVER:-dummy}
  export XDG_RUNTIME_DIR=\${XDG_RUNTIME_DIR:-/tmp/runtime-cursor}
  mkdir -p \"\${XDG_RUNTIME_DIR}\"
  unset SDL_JOYSTICK_DEVICE
  export LD_LIBRARY_PATH=${VENDOR}/pygame.libs:\${LD_LIBRARY_PATH:-}
  export PYTHONPATH=${VENDOR}:\${PYTHONPATH:-}
  chmod a+rw /dev/input/js0 2>/dev/null || true
  python3 - <<'PY'
import time, pygame
pygame.init(); pygame.joystick.init()
assert pygame.joystick.get_count()>=1, 'no joystick'
js=pygame.joystick.Joystick(0); js.init()
print(f'name={js.get_name()} axes={js.get_numaxes()} buttons={js.get_numbuttons()}')
base=[js.get_axis(i) for i in range(js.get_numaxes())]
pygame.event.pump()
seen=set(); t0=time.time()
while time.time()-t0<8:
    pygame.event.pump()
    for i in range(js.get_numbuttons()):
        if js.get_button(i) and ('b',i) not in seen:
            print(f'DOWN button[{i}]', flush=True); seen.add(('b',i))
    for i in range(js.get_numaxes()):
        v=js.get_axis(i)
        if abs(v-base[i])>0.55 and ('a',i) not in seen:
            print(f'AXIS[{i}] {base[i]:.2f}->{v:.2f}', flush=True); seen.add(('a',i))
    time.sleep(0.03)
print('seen', sorted(seen))
pygame.quit()
PY
"
