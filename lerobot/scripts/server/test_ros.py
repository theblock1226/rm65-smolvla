import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class SimpleSubscriberNode(Node):
    def __init__(self):
        super().__init__('simple_subscriber_node')
        self.create_subscription(
            JointState,
            '/unix/recorded_joint_states',
            self.callback,
            10
        )
        self.get_logger().info("Subscribed to /unix/recorded_joint_states")

    def callback(self, msg):
        self.get_logger().info(f"Received joint states: {msg.name}, positions: {msg.position}")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
