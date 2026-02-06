#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

import math


def euler_to_quaternion(yaw):
    """Helper to convert yaw to quaternion (x, y, z, w)"""
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class WaypointPublisher(Node):
    def __init__(self):
        super().__init__("waypoint_publisher")

        # Declare parameters (shared config)
        self.declare_parameters(
            namespace="",
            parameters=[
                ("waypoints_x", [0.0]),
                ("waypoints_y", [0.0]),
                ("waypoints_yaw", [0.0]),
                ("frame_id", "map"),
            ],
        )

        # Get parameters
        waypoints_x = self.get_parameter("waypoints_x").value
        waypoints_y = self.get_parameter("waypoints_y").value
        waypoints_yaw = self.get_parameter("waypoints_yaw").value
        self.frame_id = self.get_parameter("frame_id").value

        # Parse waypoints from parameter
        self.waypoints = []
        for wx, wy, wyaw in zip(waypoints_x, waypoints_y, waypoints_yaw):
            self.waypoints.append([wx, wy, wyaw])

        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")

        # Publisher for PoseArray
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.poses_publisher = self.create_publisher(PoseArray, "/waypoints", qos)

        # Timer to publish all waypoints once (short delay to ensure connection)
        self.timer = self.create_timer(1.0, self.publish_all_waypoints)

        self.get_logger().info("Waypoint Publisher initialized")

    def publish_all_waypoints(self):
        """Publish all waypoints at once as a PoseArray"""
        # self.timer.cancel()  # Only run once

        current_time = self.get_clock().now().to_msg()

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = current_time
        pose_array.header.frame_id = self.frame_id

        for idx, waypoint in enumerate(self.waypoints):
            # Create Pose message
            pose = Pose()
            pose.position.x = waypoint[0]
            pose.position.y = waypoint[1]
            pose.position.z = 0.0

            # Convert yaw (waypoint[2]) to quaternion
            qx, qy, qz, qw = euler_to_quaternion(waypoint[2])

            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw

            pose_array.poses.append(pose)

        self.poses_publisher.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
