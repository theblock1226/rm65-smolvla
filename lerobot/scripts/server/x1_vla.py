import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Int32, Float64MultiArray
from std_msgs.msg import UInt32, UInt64
from geometry_msgs.msg import PoseStamped
import json
import math
import threading


class VlaTaskRos:
    def __init__(self):
        """初始化 ROS2 节点和话题"""
        if not rclpy.ok():
            rclpy.init()

        self.node = Node('x1_vla_task_node')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # === Publisher ===
        # 上肢任务控制
        self.vla_task_pub = self.node.create_publisher(String, '/vla_task_switch', qos_profile)
        # 左手臂关节控制
        self.movej_left_pub = self.node.create_publisher(Float64MultiArray, '/movej_left_angle', qos_profile)
        # 右手臂关节控制
        self.movej_right_pub = self.node.create_publisher(Float64MultiArray, '/movej_right_angle', qos_profile)
        # 身体升降控制
        self.lift_goal_pub = self.node.create_publisher(Int32, '/lift/cmd_position', qos_profile)
        # 手爪控制
        self.gripper_pub = self.node.create_publisher(Int32, '/joystick_info', qos_profile)

        # 导航目标点
        self.nav_goal_pub = self.node.create_publisher(String, '/start_task_v2', qos_profile)
        # 直线后退导航（和导航目标点一样的反馈）
        self.move_distance_pub = self.node.create_publisher(UInt64, '/move_backward_distance', qos_profile)

        # 导航强制停止（暂不启用）
        self.nav_stop_pub = self.node.create_publisher(String, '/task_ctrl_v2', qos_profile)

        # === Subscriber ===
        # 上肢任务控制反馈
        self.vla_task_status_sub = self.node.create_subscription(
            Int32,
            '/vla_task_status',
            self.vla_task_status_callback,
            qos_profile
        )
        # 左（右）手臂状态反馈
        self.arm_move_status_sub = self.node.create_subscription(
            Int32,
            '/arm_move_info',
            self.arm_move_status_callback,
            qos_profile
        )
        # 手臂升降状态反馈
        self.lift_status_sub = self.node.create_subscription(
            Int32,
            '/lift/status_position',
            self.lift_status_callback,
            qos_profile
        )

        # 导航收到任务反馈
        self.wheel2control_sub = self.node.create_subscription(
            UInt32,
            '/wheel2control',
            self.wheel2control_callback,
            qos_profile
        )
        # 导航结果反馈
        self.nav_status_sub = self.node.create_subscription(
            String,
            '/task_status_v2',
            self.nav_status_callback,
            qos_profile
        )
        
        # # slam位姿反馈（暂不启用）
        # qos_pose = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        # self.location_status_sub = self.node.create_subscription(
        #     PoseStamped,
        #     '/slam/pose',
        #     self.location_status_callback,
        #     qos_pose
        # )

        # === 内部状态 ===

        # 订阅手柄信息
        self.joystick_subscription = self.node.create_subscription(
            Int32,
            '/joystick_info',
            self._joystick_callback,
            qos_profile
        )
        self.joystick_task_yes = False  # 手柄触发执行当前任务
        self.joystick_task_no = False  # 手柄触发跳过当前任务

        self.wheel_nav_status = 0
        self.lift_goal_position = 0

        self._events = set()
        self._event_lock = threading.Lock()

        self.node.get_logger().info("VLA Task manager initialized.")

        self._wait_for_all_subscribers()  # 等待订阅者准备好

    def _wait_for_all_subscribers(self):
        """
        无限等待，直到所有订阅者准备好。
        会实时打印缺失的订阅者列表。
        """
        required = {
            # "/vla_task_switch": self.vla_task_pub,
            "/movej_left_angle": self.movej_left_pub,
            "/movej_right_angle": self.movej_right_pub,
            "/lift/cmd_position": self.lift_goal_pub,
            "/start_task_v2": self.nav_goal_pub,
            "/move_backward_distance": self.move_distance_pub,
            "/task_ctrl_v2": self.nav_stop_pub,
        }

        self.node.get_logger().info("等待所有订阅者准备就绪...")

        while rclpy.ok():
            missing = []
            for topic, pub in required.items():
                if self.node.count_subscribers(topic) == 0:
                    missing.append(topic)

            if not missing:
                self.node.get_logger().info("所有订阅者已准备就绪 ✅")
                return
            else:
                self.node.get_logger().warn(f"仍在等待以下订阅者连接: {missing}")

            rclpy.spin_once(self.node, timeout_sec=1.0)


    # === Publisher 方法 ===
    def pub_vla_task(self, task: str):
        """发布上肢控制任务"""
        msg = String()
        msg.data = task
        self.vla_task_pub.publish(msg)
        self.node.get_logger().info(f"下发任务: {task}")

    def pub_movej_left(self, angles: list[float]):
        """发布左手臂关节角度
        angles: 最多7个关节角度值（弧度），不足补0，多余会截断。
        """
        # 若angles=[0.8, 0.0, 0.0, -2.1, 0.0, 0.5, 0.0]
        # 补全后，angles=[0.8, 0.0, 0.0, -2.1, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 14位，前7个是关节角度（弧度），后7个是关节速度，可补齐为0.0
        msg = Float64MultiArray()
        safe_angles = (angles[:7] + [0.0] * 7)[:7]  # 截断或补齐到7个角度
        full_data = safe_angles + [0.0] * 7        # 14位（7角度 + 7速度）
        msg.data = full_data
        self.movej_left_pub.publish(msg)
        self.node.get_logger().info(f"  -- 发布左手臂角度: {full_data}")

    def pub_movej_right(self, angles: list[float]):
        """发布右手臂关节角度
        angles: 最多7个关节角度值（弧度），不足补0，多余会截断。
        """
        msg = Float64MultiArray()
        safe_angles = (angles[:7] + [0.0] * 7)[:7]
        full_data = safe_angles + [0.0] * 7
        msg.data = full_data
        self.movej_right_pub.publish(msg)
        self.node.get_logger().info(f"  -- 发布右手臂角度: {full_data}")

    def pub_lift_goal_position(self, position: int):
        """发布升降位置"""
        msg = Int32()
        msg.data = position
        self.lift_goal_pub.publish(msg)
        self.lift_goal_position = position # 更新内部状态
        self.node.get_logger().info(f"  -- 发布升降目标位置: {self.lift_goal_position}")
        time.sleep(0.1) # 等待 100ms 确保消息发送
        self.lift_goal_pub.publish(msg) # 再发布一次

    def pub_gripper_left_pos(self, left_gripper_pos: float):
        """发布左手爪控制命令
        left_gripper_pos: 0.99 左手打开, 0.0 左手闭合
        """
        msg = Int32()
        msg.data = int(left_gripper_pos * 100.0 + 100.0)
        self.gripper_pub.publish(msg)
        self.node.get_logger().info(f"  -- 发布左手爪控制命令: {left_gripper_pos}")

    def pub_gripper_right_pos(self, right_gripper_pos: float):
        """发布右手爪控制命令
        right_gripper_pos: 0.99 右手打开, 0.0 右手闭合
        """
        msg = Int32()
        msg.data = int(right_gripper_pos * 100.0 + 200.0)
        self.gripper_pub.publish(msg)
        self.node.get_logger().info(f"  -- 发布右手爪控制命令: {right_gripper_pos}")


    def pub_moving_goal(self, x: float, y: float, yaw: float):
        """发布导航目标点 (x, y, yaw)，单位 (m, m, rad)"""
        self.wheel_nav_status = 0 # 重置轮式导航状态
        goal_dict = {
            "task_id": "ca76d33f-7f26-4ae8-b01b-b78de484854e",
            "type": "goto_pose",
            "detail": {
                "pose": [x, y, yaw],
                "is_accurate_pose": True
            }
        }
        msg = String()
        msg.data = json.dumps(goal_dict)
        self.nav_goal_pub.publish(msg)
        self.node.get_logger().info(f"  -- 发布导航目标点: x={x}, y={y}, yaw={yaw}")

        # first = True
        # while rclpy.ok() and self.wheel_nav_status == 0:
        #     self.nav_goal_pub.publish(msg)
        #     if first:
        #         self.node.get_logger().info(f"  -- 发布导航目标点: x={x}, y={y}, yaw={yaw}")
        #         first = False
        #     time.sleep(1.0)
        #     if self.wheel_nav_status != 0:
        #         self.node.get_logger().info(f"导航任务已被接收 ✅")
        #         break

    def pub_move_distance(self, distance: int):
        """发布直线后退移动距离，单位: 厘米"""
        msg = UInt64()
        msg.data = distance
        self.move_distance_pub.publish(msg)
        self.node.get_logger().info(f"  -- 发布后退移动距离: {distance}")


    # 发布停止导航指令（暂不启用）
    def pub_stop_navigation(self):
        """发布停止导航指令"""
        msg_dict = {
            "task_id": "ca76d33f-7f26-4ae8-b01b-b78de484854e",
            "cmd": "stop"
        }
        msg = String()
        msg.data = json.dumps(msg_dict)
        # 发布两次，确保消息被接收
        self.nav_stop_pub.publish(msg)
        self.node.get_logger().info("  -- 发布停止导航指令")
        time.sleep(0.1)
        self.nav_stop_pub.publish(msg)


    # === Subscriber 回调 ===
    def vla_task_status_callback(self, msg: Int32):
        """上肢任务控制反馈回调"""
        if msg.data == 1:
            self.node.get_logger().info("上肢任务已完成 ✅")
            self._add_event("VlaTaskCompleted")
        elif msg.data == 0:
            self.node.get_logger().info("上肢任务失败 ❌")
            self._add_event("VlaTaskFailed")

    def arm_move_status_callback(self, msg: Int32):
        """
        左（右）手臂状态反馈回调
        10: 左臂失败
        11: 左臂完成
        20: 右臂失败
        21: 右臂完成
        """
        if msg.data == 11:
            self.node.get_logger().info("左手臂状态反馈: 动作完成 ✅")
            self._add_event("LeftArmDone")
        elif msg.data == 10:
            self.node.get_logger().error("左手臂状态反馈: 动作失败 ❌")
            self._add_event("LeftArmFailed")
        elif msg.data == 21:
            self.node.get_logger().info("右手臂状态反馈: 动作完成 ✅")
            self._add_event("RightArmDone")
        elif msg.data == 20:
            self.node.get_logger().error("右手臂状态反馈: 动作失败 ❌")
            self._add_event("RightArmFailed")

    def lift_status_callback(self, msg: Int32):
        """升降状态反馈回调"""
        position = msg.data
        self.node.get_logger().info(f"Received /lift_status: {position}")
        if abs(position - getattr(self, "lift_goal_position", 0)) < 200:
            self.node.get_logger().info("升降到达目标位置 ✅")
            self._add_event("LiftArrived")
        else:
            self.node.get_logger().info("升降未到达目标位置 ❌")
            self._add_event("LiftFailed")


    def wheel2control_callback(self, msg: UInt32):
        """导航收到任务反馈回调"""
        self.wheel_nav_status = msg.data
        self.node.get_logger().info(f"导航状态更新 ✅: {self.wheel_nav_status}")
        # self._add_event("NavTaskReceived")
    def nav_status_callback(self, msg: String):
        """导航结果反馈回调"""
        self.node.get_logger().info(f"Received /task_status_v2: {msg.data}")
        try:
            json_msg = json.loads(msg.data)
            task_id = json_msg.get("task_id", "")
            status = json_msg.get("status", "")
            fail_reason = json_msg.get("fail_reason", "")
            succ_reason = json_msg.get("succ_reason", "")

            self.node.get_logger().info(f"Task ID: {task_id}, Status: {status}")

            if task_id == "ca76d33f-7f26-4ae8-b01b-b78de484854e":
                if status == "success":
                    self.node.get_logger().info(f"Task succeeded. Reason: {succ_reason}")
                    self._add_event("NavSuccess")
                elif status == "fail":
                    self.node.get_logger().error(f"Task failed. Reason: {fail_reason}")
                    self._add_event("NavFailed")
        except Exception as e:
            self.node.get_logger().error(f"Failed to parse /task_status_v2 JSON: {e}")


    # SLAM 位姿反馈回调（暂不启用）
    # def location_status_callback(self, msg: PoseStamped):
    #     """SLAM 位姿反馈回调"""
    #     x = msg.pose.position.x
    #     y = msg.pose.position.y

    #     # 四元数转 yaw
    #     q = msg.pose.orientation
    #     siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    #     cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    #     yaw = math.atan2(siny_cosp, cosy_cosp)

    #     self.node.get_logger().info(f"Current location: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    # 手柄信息回调（控制程序）
    def _joystick_callback(self, msg):
        """手柄信息回调函数"""
        if msg.data == 18:
            self.joystick_task_yes = True
            self.node.get_logger().info('收到手柄信号: 18，执行当前任务')
        elif msg.data == 19:
            self.joystick_task_no = True
            self.node.get_logger().info('收到手柄信号: 19，跳过当前任务')

    # === 通用事件处理 ===
    def _add_event(self, event_type: str):
        with self._event_lock:
            if event_type not in self._events:
                self._events.add(event_type)
                self.node.get_logger().info(f"事件触发: {event_type}")

    def wait_for(self, event_types, timeout: float = 30.0) -> str | None:
        """
        阻塞等待指定事件，支持单个字符串或字符串列表。
        返回捕获到的事件类型。
        超时后会自动补发 Fail 事件（如果 event_types 中包含 xxxFailed），避免卡死。
        """
        if isinstance(event_types, str):
            event_types = [event_types]

        self.node.get_logger().info(f"等待事件: {event_types} (timeout={timeout}s)")
        start = time.time()
        while rclpy.ok():
            with self._event_lock:
                for e in event_types:
                    if e in self._events:
                        self._events.remove(e)
                        self.node.get_logger().info(f"事件已捕获: {e}")
                        return e
            if time.time() - start > timeout:
                self.node.get_logger().warn(f"等待事件 {event_types} 超时")

                # === 统一兜底 Fail 事件 ===
                for e in event_types:
                    if e.endswith("Failed"):
                        self._add_event(e)
                        self.node.get_logger().warn(f"触发兜底事件: {e}")
                        return e

                return None
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def shutdown(self):
        """关闭 ROS 节点"""
        self.node.get_logger().info("Shutting down VlaTask...")
        self.node.destroy_node()
        rclpy.shutdown()

# 步骤函数定义
def init_step(task):
    # 初始状态（左手、右手、身体、夹爪、导航位置）
    task.pub_movej_left([0.8, 0.0, 0.0, -2.1, 0.0, 0.5, 0.0]) # 左手臂收回初始点
    task.wait_for(["LeftArmDone", "LeftArmFailed"], timeout=20)
    time.sleep(0.5)
    task.pub_movej_right([-0.8, 0.0, 0.0, 2.1, 0.0, -0.5, 0.0]) # 右手臂收回初始点
    task.wait_for(["RightArmDone", "RightArmFailed"], timeout=20)
    task.pub_lift_goal_position(0) # 升回身体初始点
    task.wait_for(["LiftArrived", "LiftFailed"], timeout=20)
    task.pub_gripper_left_pos(0.0)  # 左手闭合
    task.pub_gripper_right_pos(0.0) # 右手闭合
    task.pub_moving_goal(0.0, 0.0, 0.0) # 导航初始点
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def sofa_step(task):
    # 到沙发衣服点
    task.pub_moving_goal(-0.761, -0.306, -3.134)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def pickup_sofa_step(task):
    # 抓取沙发衣服(右手)vla_task
    task.pub_vla_task("pick up the clothes from sofa")
    task.wait_for(["VlaTaskCompleted", "VlaTaskFailed"], timeout=60)

def goto_place_step(task):
    # 到放置衣服点
    task.pub_moving_goal(-0.590, 1.704, 2.964)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def place_clothes_step(task):
    # 放衣服（右手）
    task.pub_movej_right([-0.1, -0.3, 0.65, 1.8, 0.0, 0.95, 0.2]) # 右手臂前伸
    task.wait_for(["RightArmDone", "RightArmFailed"], timeout=20)
    task.pub_gripper_right_pos(0.99)  # 右手打开
    time.sleep(1.5)
    task.pub_gripper_right_pos(0.0)  # 右手闭合
    task.pub_movej_right([-0.8, 0.0, 0.0, 2.1, 0.0, -0.5, 0.0]) # 右手臂收回初始点
    task.wait_for(["RightArmDone", "RightArmFailed"], timeout=20)

def open_door_step(task):
    # 开洗衣机门（右手）vla_task
    task.pub_vla_task("open the washing machine door")
    task.wait_for(["VlaTaskCompleted", "VlaTaskFailed"], timeout=60)

def goto_basket_step(task):
    # 到篮子衣服点
    task.pub_moving_goal(-0.592, 1.712, 1.601)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def pickup_basket_step(task):
    # 抓取篮子衣服（左手）vla_task
    task.pub_vla_task("pick up the clothes from the storage basket")
    task.wait_for(["VlaTaskCompleted", "VlaTaskFailed"], timeout=60)

def goto_push_point_step(task):
    # 到洗衣机推门点
    task.pub_lift_goal_position(-60000)  # 身体降低高度
    task.wait_for(["LiftArrived", "LiftFailed"], timeout=20)
    task.pub_moving_goal(-1.05, 1.671, -3.134)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def goto_door_point_step(task):
    # 到洗衣机门口点
    task.pub_movej_left([1.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0]) # 左手臂后缩
    task.wait_for(["LeftArmDone", "LeftArmFailed"], timeout=20)
    time.sleep(0.5)
    task.pub_movej_right([-1.0, 0.0, 0.0, 1.5, 0.0, -0.75, 0.0]) # 右手臂后缩
    task.wait_for(["RightArmDone", "RightArmFailed"], timeout=20)
    time.sleep(1)
    task.pub_moving_goal(-0.999, 1.671, 1.652)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def place_left_clothes_step(task):
    # 放衣服（左手）
    task.pub_movej_left([-0.7, 0.0, 0.0, -0.75, 0.0, 0.25, 0.0]) # 左手臂前伸
    task.wait_for(["LeftArmDone", "LeftArmFailed"], timeout=20)
    task.pub_movej_left([-0.7, 0.0, 0.0, -0.75, 0.0, -0.25, 0.0]) # 左手臂前伸
    task.wait_for(["LeftArmDone", "LeftArmFailed"], timeout=20)
    task.pub_gripper_left_pos(0.99)  # 左手打开
    time.sleep(1.5)
    task.pub_gripper_left_pos(0.0)  # 左手闭合
    task.pub_movej_left([0.8, 0.0, 0.0, -2.1, 0.0, 0.5, 0.0]) # 左手臂收回初始点
    task.wait_for(["LeftArmDone", "LeftArmFailed"], timeout=20)
    time.sleep(0.5)
    task.pub_movej_right([-0.8, 0.0, 0.0, 2.1, 0.0, -0.5, 0.0]) # 右手臂收回初始点
    task.wait_for(["RightArmDone", "RightArmFailed"], timeout=20)
    task.pub_lift_goal_position(0)  # 升回原点
    task.wait_for(["LiftArrived", "LiftFailed"], timeout=20)

def goto_close_point_step(task):
    # 到洗衣机关门点
    task.pub_move_distance(30)  # 直线后退导航（单位厘米 ）
    task.wait_for(["NavSuccess", "NavFailed"], timeout=20)
    task.pub_moving_goal(-1.108, 1.415, 2.263)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

def close_door_step(task):
    # 关洗衣机门（左手、右手）vla_task
    task.pub_vla_task("close the washing machine door")
    task.wait_for(["VlaTaskCompleted", "VlaTaskFailed"], timeout=60)

def goto_home_step(task):
    # 回导航初始点
    task.pub_moving_goal(0.0, 0.0, 0.0)
    task.wait_for(["NavSuccess", "NavFailed"], timeout=60)

# 手动控制模式
def main():
    task = VlaTaskRos()

    # 定义所有步骤
    steps = [
        ("初始状态", lambda: init_step(task)),
        ("到沙发衣服点", lambda: sofa_step(task)),
        ("抓取沙发衣服(右手)vla_task", lambda: pickup_sofa_step(task)),
        ("到放置衣服点", lambda: goto_place_step(task)),
        ("放衣服（右手）", lambda: place_clothes_step(task)),
        ("开洗衣机门（右手）vla_task", lambda: open_door_step(task)),
        ("到篮子衣服点", lambda: goto_basket_step(task)),
        ("抓取篮子衣服（左手）vla_task", lambda: pickup_basket_step(task)),
        ("到洗衣机推门点", lambda: goto_push_point_step(task)),
        ("到洗衣机门口点", lambda: goto_door_point_step(task)),
        ("放衣服（左手）", lambda: place_left_clothes_step(task)),
        ("到洗衣机关门点", lambda: goto_close_point_step(task)),
        ("关洗衣机门vla_task", lambda: close_door_step(task)),
        ("回导航初始点", lambda: goto_home_step(task))
    ]

    try:
        task.node.get_logger().info("程序启动，等待手柄触发执行步骤...")
        task.node.get_logger().info("手柄18: 执行当前步骤 | 手柄19: 跳过当前步骤")
        
        current_step = 0
        
        while rclpy.ok() and current_step < len(steps):
            # 持续监听手柄
            rclpy.spin_once(task.node, timeout_sec=0.1)
            
            # 检查跳过信号（手柄19）
            if task.joystick_task_no:
                if current_step < len(steps):
                    task.node.get_logger().warn(f"收到跳过信号，跳过步骤: {steps[current_step][0]}")
                    current_step += 1  # 进入下一步
                    
                task.joystick_task_no = False  # 重置跳过标志
                
                if current_step < len(steps):
                    next_step_name = steps[current_step][0]
                    task.node.get_logger().info(f"等待手柄触发下一步: {next_step_name}")
                else:
                    task.node.get_logger().info("所有步骤完成！按手柄重新开始")
                    current_step = 0  # 重置，可以重新开始
                
                continue
            
            # 收到手柄触发信号（手柄18）- 改为同步执行
            if task.joystick_task_yes:
                step_name, step_func = steps[current_step]
                task.node.get_logger().info(f"步骤 {current_step + 1}/{len(steps)}: {step_name}")
                
                task.joystick_task_yes = False  # 重置触发标志
                
                try:
                    # 直接在主线程中执行任务（同步）
                    step_func()  # 执行当前步骤
                    current_step += 1  # 进入下一步
                    
                    if current_step < len(steps):
                        next_step_name = steps[current_step][0]
                        task.node.get_logger().info(f"步骤完成！等待手柄触发下一步: {next_step_name}")
                    else:
                        task.node.get_logger().info("所有步骤完成！按手柄重新开始")
                        current_step = 0  # 重置，可以重新开始
                        
                except Exception as e:
                    task.node.get_logger().error(f"步骤执行失败: {e}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        task.node.get_logger().info("程序被中断")
    finally:
        task.shutdown()
 
# 自动控制模式
def main_auto():
    task = VlaTaskRos()

    # 定义所有步骤
    steps = [
        ("初始状态", lambda: init_step(task)),
        ("到沙发衣服点", lambda: sofa_step(task)),
        ("抓取沙发衣服(右手)vla_task", lambda: pickup_sofa_step(task)),
        ("到放置衣服点", lambda: goto_place_step(task)),
        ("放衣服（右手）", lambda: place_clothes_step(task)),
        ("开洗衣机门（右手）vla_task", lambda: open_door_step(task)),
        ("到篮子衣服点", lambda: goto_basket_step(task)),
        ("抓取篮子衣服（左手）vla_task", lambda: pickup_basket_step(task)),
        ("到洗衣机推门点", lambda: goto_push_point_step(task)),
        ("到洗衣机门口点", lambda: goto_door_point_step(task)),
        ("放衣服（左手）", lambda: place_left_clothes_step(task)),
        ("到洗衣机关门点", lambda: goto_close_point_step(task)),
        ("关洗衣机门vla_task", lambda: close_door_step(task)),
        ("回导航初始点", lambda: goto_home_step(task))
    ]

    def execute_all_steps():
        """执行所有步骤的函数"""
        task.node.get_logger().info("🚀 开始执行完整任务流程...")
        
        for i, (step_name, step_func) in enumerate(steps):
            try:
                task.node.get_logger().info(f"📋 步骤 {i + 1}/{len(steps)}: {step_name}")
                step_func()  # 执行当前步骤
                task.node.get_logger().info(f"✅ 步骤 {i + 1} 完成: {step_name}")
                
                # 步骤间短暂间隔
                time.sleep(0.5)
                
            except Exception as e:
                task.node.get_logger().error(f"❌ 步骤 {i + 1} 执行失败: {step_name} - {e}")
                # 可以选择继续执行下一步，或者中断整个流程
                user_choice = input("步骤失败，是否继续执行下一步？(y/n): ")
                if user_choice.lower() != 'y':
                    task.node.get_logger().warn("用户选择中断流程")
                    return False
        
        task.node.get_logger().info("🎉 所有步骤执行完成！")
        return True

    try:
        task.node.get_logger().info("自动执行模式启动")
        task.node.get_logger().info("手柄18: 开始执行完整流程 | 手柄19: 退出程序")
        
        while rclpy.ok():
            # 持续监听手柄
            rclpy.spin_once(task.node, timeout_sec=0.1)
            
            # 检查退出信号（手柄19）
            if task.joystick_task_no:
                task.node.get_logger().info("收到退出信号，程序结束")
                task.joystick_task_no = False
                break
            
            # 收到开始执行信号（手柄18）
            if task.joystick_task_yes:
                task.node.get_logger().info("🎯 收到手柄触发信号，开始执行完整任务流程...")
                task.joystick_task_yes = False  # 重置触发标志
                
                # 执行所有步骤
                success = execute_all_steps()
                
                if success:
                    task.node.get_logger().info("✨ 任务流程执行完成！等待下一次手柄触发...")
                else:
                    task.node.get_logger().warn("⚠️  任务流程执行中断！等待下一次手柄触发...")
                
                # 任务完成后的提示
                task.node.get_logger().info("手柄18: 重新执行完整流程 | 手柄19: 退出程序")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        task.node.get_logger().info("程序被中断")
    finally:
        task.shutdown()


if __name__ == '__main__':
    main() # 手动控制模式（每步需要手柄触发）
    # main_auto() # 自动执行模式（一次性执行所有步骤）

