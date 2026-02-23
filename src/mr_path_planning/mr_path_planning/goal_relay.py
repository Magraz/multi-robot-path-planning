"""Relay /goal_pose to each robot's namespaced goal_pose topic."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped


ROBOT_NAMES = ["robot_0", "robot_1"]


class GoalRelay(Node):
    def __init__(self):
        super().__init__("goal_relay")

        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.goal_pubs = {}
        for name in ROBOT_NAMES:
            self.goal_pubs[name] = self.create_publisher(
                PoseStamped, f"/{name}/goal_pose", qos
            )

        self.subscription = self.create_subscription(
            PoseStamped, "/goal_pose", self.on_goal, qos
        )

    def on_goal(self, msg: PoseStamped):
        for name, pub in self.goal_pubs.items():
            forwarded = PoseStamped()
            forwarded.header = msg.header
            # Rewrite frame_id to the robot's prefixed map frame
            forwarded.header.frame_id = f"{name}/map"
            forwarded.pose = msg.pose
            pub.publish(forwarded)
            self.get_logger().info(
                f"Relayed goal to /{name}/goal_pose "
                f"(x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GoalRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
