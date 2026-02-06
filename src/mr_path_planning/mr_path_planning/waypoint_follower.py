#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import math


def get_yaw_from_quaternion(q):
    """
    Convert orientation quaternion to Euler yaw angle.
    """
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class WaypointFollower(Node):
    def __init__(self):
        super().__init__("waypoint_follower")

        # Parameters
        self.declare_parameter("linear_speed", 0.5)
        self.declare_parameter("angular_speed", 1.0)
        self.declare_parameter(
            "go_to_goal_tolerance", 0.2
        )  # Distance radius to accept waypoint

        # Internal state
        self.waypoints = []
        self.current_waypoint_index = 0
        self.current_pose = None
        self.goal_received = False
        self.finished = False
        self.active = False

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, "/ground_truth", self.odom_callback, 10
        )
        self.waypoints_sub = self.create_subscription(
            PoseArray, "/waypoints", self.waypoints_callback, 10
        )

        # Service
        self.srv = self.create_service(
            SetBool, "start_follower", self.start_follower_callback
        )

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Control loop timer (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            "Waypoint Follower Node Initialized. Waiting for waypoints..."
        )

    def start_follower_callback(self, request, response):
        """Callback to enable/disable the follower."""
        self.active = request.data
        response.success = True
        if self.active:
            response.message = "Waypoint follower started."
        else:
            response.message = "Waypoint follower stopped."
            self.stop_robot()

        self.get_logger().info(response.message)
        return response

    def waypoints_callback(self, msg):
        """Store the received waypoints."""
        if not self.goal_received:
            self.waypoints = msg.poses
            self.current_waypoint_index = 0
            self.goal_received = True
            self.finished = False
            self.get_logger().info(f"Received {len(self.waypoints)} waypoints.")

    def odom_callback(self, msg):
        """Update robot pose from odometry."""
        self.current_pose = msg.pose.pose

    def control_loop(self):
        """Main controller logic."""
        if not self.goal_received or self.finished or not self.active:
            return

        if self.current_pose is None:
            self.get_logger().warn("Waiting for odometry...", throttle_duration_sec=2.0)
            return

        # Get current goal
        if self.current_waypoint_index >= len(self.waypoints):
            self.stop_robot()
            self.finished = True
            self.get_logger().info("All waypoints completed!")
            return

        target = self.waypoints[self.current_waypoint_index]

        # Calculate error
        dx = target.position.x - self.current_pose.position.x
        dy = target.position.y - self.current_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        # Check if we reached the waypoint
        tolerance = self.get_parameter("go_to_goal_tolerance").value
        if distance < tolerance:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}")
            self.current_waypoint_index += 1
            return  # Will pick up next point in next cycle

        # Calculate heading to goal
        desired_yaw = math.atan2(dy, dx)
        current_yaw = get_yaw_from_quaternion(self.current_pose.orientation)
        heading_error = normalize_angle(desired_yaw - current_yaw)

        # Simple p controller
        twist = Twist()

        # Angular control
        k_w = 2.0
        twist.angular.z = k_w * heading_error

        # Saturate angular speed
        max_w = self.get_parameter("angular_speed").value
        twist.angular.z = max(min(twist.angular.z, max_w), -max_w)

        # Linear control (slow down if turning sharply)
        k_v = 1.0
        if abs(heading_error) > 0.5:
            # Stop and turn if facing wrong way
            twist.linear.x = 0.0
        else:
            twist.linear.x = k_v * distance

            # Saturate linear speed
            max_v = self.get_parameter("linear_speed").value
            twist.linear.x = max(min(twist.linear.x, max_v), -max_v)

        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
