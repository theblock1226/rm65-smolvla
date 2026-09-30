#!/usr/bin/env bash
# 在 rm65_dev2 容器内加载 ROS2 + rm65 工作空间环境
# 用法: source scripts/docker_ros_env.sh
CONTAINER="${RM65_CONTAINER:-rm65_dev2}"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="/home/sdznkj/zkx_ws/ros2_ws/install/setup.bash"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "容器 ${CONTAINER} 未运行。请先: docker start ${CONTAINER}" >&2
  return 1 2>/dev/null || exit 1
fi

# 在宿主机上封装 docker exec，后续命令都在容器内执行
docker_exec() {
  docker exec -it "${CONTAINER}" bash -lc "
    source ${ROS_SETUP}
    source ${WS_SETUP}
    $*
  "
}

# 供 source 后直接启动交互 shell
docker_shell() {
  docker exec -it "${CONTAINER}" bash -lc "
    source ${ROS_SETUP}
    source ${WS_SETUP}
    exec bash
  "
}

echo "已配置 docker_exec / docker_shell（容器: ${CONTAINER}）"
echo "示例: docker_exec 'ros2 launch rm_bringup rm_65_bringup.launch.py'"
