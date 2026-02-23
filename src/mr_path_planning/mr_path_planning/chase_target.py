"""Read target_0 position and send it as a goal to robot_0 and robot_1 every 5s.

Service ~/enable (std_srvs/Trigger) toggles chasing on/off.
Starts disabled by default.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


CHASERS = ["robot_0", "robot_1"]


class ChaseTarget(Node):
    def __init__(self):
        super().__init__("chase_target")

        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("period", 5.0)
        self.declare_parameter("enabled", False)

        target = self.get_parameter("target_name").value
        period = self.get_parameter("period").value
        self.enabled = self.get_parameter("enabled").value

        goal_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.goal_pubs = {}
        for name in CHASERS:
            self.goal_pubs[name] = self.create_publisher(
                PoseStamped, f"/{name}/goal_pose", goal_qos
            )

        self.latest_odom = None
        self.create_subscription(
            Odometry, f"/{target}/ground_truth", self.on_odom, 10
        )

        self.create_service(Trigger, "~/enable", self.on_enable)

        self.timer = self.create_timer(period, self.send_goal)
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(
            f"Chase {state} | /{target}/ground_truth -> "
            f"{[f'/{n}/goal_pose' for n in CHASERS]} every {period}s"
        )

    def on_enable(self, _request, response):
        self.enabled = not self.enabled
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(f"Chase {state}")
        response.success = True
        response.message = f"Chase {state} (toggle service)"
        return response

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def send_goal(self):
        if not self.enabled:
            return

        if self.latest_odom is None:
            self.get_logger().warn("No odom from target yet, skipping")
            return

        pos = self.latest_odom.pose.pose.position
        ori = self.latest_odom.pose.pose.orientation

        for name, pub in self.goal_pubs.items():
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = f"{name}/map"
            goal.pose.position = pos
            goal.pose.orientation = ori
            pub.publish(goal)

        self.get_logger().info(
            f"Sent chase goal (x={pos.x:.2f}, y={pos.y:.2f}) to {CHASERS}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ChaseTarget()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
