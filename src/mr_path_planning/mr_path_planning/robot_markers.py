"""Publish live cylinder + label markers for each robot in RViz.

Subscribes to /{name}/ground_truth (Odometry) for every robot and
re-publishes a MarkerArray on /robot_markers at 10 Hz so all robots
are always visible regardless of TF tree issues.

Marker colours:
  - Searcher robots → blue
  - Target robot    → red

Labels cycle through: r_1, r_2, ... for searchers; t_1 for target.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

# RGBA tuples
_BLUE = (0.20, 0.55, 1.00, 0.85)
_RED  = (1.00, 0.20, 0.20, 0.85)


class RobotMarkersNode(Node):
    """Publish cylinder + text markers that follow each robot."""

    def __init__(self) -> None:
        super().__init__("robot_markers")

        self.declare_parameter("searcher_names", ["robot_0", "robot_1"])
        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("frame_id", "robot_0/map")
        self.declare_parameter("cylinder_radius", 0.30)
        self.declare_parameter("cylinder_height", 0.30)
        self.declare_parameter("label_z_offset", 0.70)
        self.declare_parameter("label_size", 0.40)

        self.searchers = list(self.get_parameter("searcher_names").value)
        self.target    = str(self.get_parameter("target_name").value)
        self.frame_id  = str(self.get_parameter("frame_id").value)
        self.cyl_r     = float(self.get_parameter("cylinder_radius").value)
        self.cyl_h     = float(self.get_parameter("cylinder_height").value)
        self.label_z   = float(self.get_parameter("label_z_offset").value)
        self.label_sz  = float(self.get_parameter("label_size").value)

        # Build human-readable labels: robot_0 → r_1, robot_1 → r_2, target_0 → t_1
        self._labels: dict = {}
        for i, name in enumerate(self.searchers):
            self._labels[name] = f"r_{i + 1}"
        self._labels[self.target] = "t_1"

        # Latest (x, y) per robot
        self._positions: dict = {}

        # Odometry QoS – Stage publishes with Best Effort
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for name in self.searchers + [self.target]:
            self.create_subscription(
                Odometry,
                f"/{name}/ground_truth",
                lambda msg, n=name: self._odom_cb(n, msg),
                odom_qos,
            )

        self._pub = self.create_publisher(MarkerArray, "/robot_markers", 10)
        self.create_timer(0.1, self._publish)  # 10 Hz

        self.get_logger().info(
            f"robot_markers ready — tracking: {self.searchers + [self.target]}"
        )

    # ------------------------------------------------------------------
    def _odom_cb(self, name: str, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._positions[name] = (float(p.x), float(p.y))

    def _publish(self) -> None:
        if not self._positions:
            return

        now = self.get_clock().now().to_msg()
        array = MarkerArray()

        all_robots = self.searchers + [self.target]
        for robot_idx, name in enumerate(all_robots):
            if name not in self._positions:
                continue

            x, y = self._positions[name]
            label = self._labels[name]
            color = _RED if name == self.target else _BLUE

            # --- cylinder ---
            cyl = Marker()
            cyl.header.frame_id = self.frame_id
            cyl.header.stamp    = now
            cyl.ns              = "robot_cylinders"
            cyl.id              = robot_idx * 2
            cyl.type            = Marker.CYLINDER
            cyl.action          = Marker.ADD
            cyl.pose.position.x = x
            cyl.pose.position.y = y
            cyl.pose.position.z = self.cyl_h / 2.0
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = self.cyl_r * 2.0
            cyl.scale.y = self.cyl_r * 2.0
            cyl.scale.z = self.cyl_h
            cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = color

            # --- text label ---
            txt = Marker()
            txt.header.frame_id = self.frame_id
            txt.header.stamp    = now
            txt.ns              = "robot_labels"
            txt.id              = robot_idx * 2 + 1
            txt.type            = Marker.TEXT_VIEW_FACING
            txt.action          = Marker.ADD
            txt.pose.position.x = x
            txt.pose.position.y = y
            txt.pose.position.z = self.label_z
            txt.pose.orientation.w = 1.0
            txt.scale.z = self.label_sz
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = label

            array.markers.extend([cyl, txt])

        self._pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotMarkersNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
