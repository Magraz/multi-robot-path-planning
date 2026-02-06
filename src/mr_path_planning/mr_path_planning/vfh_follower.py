#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseArray, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, PoseArray, Point
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


def get_yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class VFHFollower(Node):
    def __init__(self):
        super().__init__("vfh_follower")

        # Parameters
        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("angular_speed_max", 1.0)
        self.declare_parameter("goal_tolerance", 0.2)
        self.declare_parameter("robot_radius", 0.5)
        self.declare_parameter("safety_radius", 0.5)
        self.declare_parameter("sector_count", 72)

        self.declare_parameter("w_goal", 0.2)
        self.declare_parameter("w_smooth", 0.1)
        self.declare_parameter("w_clear", 1.0)
        self.declare_parameter("laser_scan_topic", "/base_scan")
        self.declare_parameter("ground_truth_topic", "/ground_truth")

        # Stuck recovery params
        self.declare_parameter("stuck_timeout", 4.0)  # seconds
        self.declare_parameter("stuck_distance", 0.05)  # meters
        self.declare_parameter("recovery_duration", 1.0)  # seconds

        self.stuck_timeout = self.get_parameter("stuck_timeout").value
        self.stuck_distance = self.get_parameter("stuck_distance").value
        self.recovery_duration = self.get_parameter("recovery_duration").value

        self.robot_radius = self.get_parameter("robot_radius").value
        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed_max = self.get_parameter("angular_speed_max").value
        self.goal_tolerance = self.get_parameter("goal_tolerance").value
        self.safety_radius = self.get_parameter("safety_radius").value
        self.sector_count = self.get_parameter("sector_count").value
        self.w_goal = self.get_parameter("w_goal").value
        self.w_smooth = self.get_parameter("w_smooth").value
        self.w_clear = self.get_parameter("w_clear").value

        scan_topic = self.get_parameter("laser_scan_topic").value
        ground_truth_topic = self.get_parameter("ground_truth_topic").value

        self.prev_steer_angle = 0.0

        # State
        self.waypoints = []
        self.current_waypoint_index = 0
        self.current_pose = None
        self.current_scan = None
        self.active = False
        self.goal_received = False
        self.finished = False

        # Stuck detection state
        self.last_progress_pose = None
        self.last_progress_time = 0.0
        self.recovery_until = 0.0

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, ground_truth_topic, self.odom_callback, 10
        )
        scan_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, scan_qos
        )

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.waypoints_sub = self.create_subscription(
            PoseArray, "/waypoints", self.waypoints_callback, qos
        )

        # Service
        self.srv = self.create_service(
            SetBool, "start_vfh_follower", self.start_follower_callback
        )

        # Publishercurrent_waypoint_index
        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.viz_pub = self.create_publisher(MarkerArray, "/vfh_histogram", 10)

        # Control Loop
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("VFH Follower ready. Call /start_follower to begin.")

        self._log_missing_active = False
        self._log_missing_goal = False
        self._log_missing_pose = False
        self._log_missing_scan = False
        self._log_finished = False

    def start_follower_callback(self, request, response):
        self.active = request.data
        response.success = True
        response.message = (
            "VFH follower started." if self.active else "VFH follower stopped."
        )
        if not self.active:
            self.stop_robot()
        self.get_logger().info(response.message)
        return response

    def waypoints_callback(self, msg):
        if not self.goal_received:
            self.waypoints = msg.poses
            self.current_waypoint_index = 0
            self.goal_received = True
            self.finished = False
            self.get_logger().info(f"Received {len(self.waypoints)} waypoints.")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        self.current_scan = msg

    def stop_robot(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def control_loop(self):
        # if not self.active or not self.goal_received or self.finished:
        #     return
        # if self.current_pose is None or self.current_scan is None:
        #     return

        if not self.active:
            if not self._log_missing_active:
                self.get_logger().warn("VFH inactive (start service not called).")
                self._log_missing_active = True
            return
        if not self.goal_received:
            if not self._log_missing_goal:
                self.get_logger().warn("No waypoints received yet.")
                self._log_missing_goal = True
            return
        if self.finished:
            if not self._log_finished:
                self.get_logger().warn("All waypoints completed.")
                self._log_finished = True
            return
        if self.current_pose is None:
            if not self._log_missing_pose:
                self.get_logger().warn("No odometry received yet.")
                self._log_missing_pose = True
            return
        if self.current_scan is None:
            if not self._log_missing_scan:
                self.get_logger().warn("No laser scan received yet.")
                self._log_missing_scan = True
            return

        # Progress / stuck detection
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_progress_pose is None:
            self.last_progress_pose = (
                self.current_pose.position.x,
                self.current_pose.position.y,
            )
            self.last_progress_time = now_sec
        else:
            dxp = self.current_pose.position.x - self.last_progress_pose[0]
            dyp = self.current_pose.position.y - self.last_progress_pose[1]
            moved = math.hypot(dxp, dyp)
            if moved > self.stuck_distance:
                self.last_progress_pose = (
                    self.current_pose.position.x,
                    self.current_pose.position.y,
                )
                self.last_progress_time = now_sec
                self.recovery_until = 0.0
            elif (
                now_sec - self.last_progress_time
            ) > self.stuck_timeout and now_sec > self.recovery_until:
                self.recovery_until = now_sec + self.recovery_duration

        target_pose = self.waypoints[self.current_waypoint_index]
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        dist_to_goal = math.sqrt(dx**2 + dy**2)

        if dist_to_goal < self.goal_tolerance:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.finished = True
                self.stop_robot()
                self.get_logger().info("All waypoints finished.")
                return
            target_pose = self.waypoints[self.current_waypoint_index]
            dx = target_pose.position.x - self.current_pose.position.x
            dy = target_pose.position.y - self.current_pose.position.y

        curr_yaw = get_yaw_from_quaternion(self.current_pose.orientation)
        target_heading = math.atan2(dy, dx)
        desired_yaw = normalize_angle(target_heading - curr_yaw)

        histogram = [0.0] * self.sector_count
        sector_size = (2 * math.pi) / self.sector_count

        angle = self.current_scan.angle_min
        inc = self.current_scan.angle_increment

        for r in self.current_scan.ranges:
            if math.isinf(r) or math.isnan(r):
                r = self.current_scan.range_max
            if r < self.safety_radius:
                norm_angle = normalize_angle(angle)
                sector_idx = int((norm_angle + math.pi) / sector_size)
                sector_idx = max(0, min(sector_idx, self.sector_count - 1))

                # Inflate by robot size
                inflate_angle = math.atan2(self.robot_radius, max(r, 0.05))
                inflate_sectors = int(math.ceil(inflate_angle / sector_size))

                for k in range(-inflate_sectors, inflate_sectors + 1):
                    histogram[(sector_idx + k) % self.sector_count] += 1

            angle += inc

        goal_sector = int((desired_yaw + math.pi) / sector_size)
        goal_sector = max(0, min(goal_sector, self.sector_count - 1))

        best_sector = -1
        found_safe = False

        best_cost = float("inf")
        for idx in range(self.sector_count):
            if histogram[idx] != 0:
                continue

            # Angle at center of this sector
            candidate_angle = (idx * sector_size) - math.pi + (sector_size / 2.0)

            # Cost terms
            cost_goal = abs(normalize_angle(candidate_angle - desired_yaw))
            cost_smooth = abs(normalize_angle(candidate_angle - self.prev_steer_angle))

            # Clearance: prefer sectors with more free neighbors
            # Count free neighbors in a small window
            window = 2
            free_neighbors = 0
            for k in range(-window, window + 1):
                if histogram[(idx + k) % self.sector_count] == 0:
                    free_neighbors += 1
            # Higher clearance => lower cost
            cost_clear = 1.0 / max(free_neighbors, 1)

            total_cost = (
                self.w_goal * cost_goal
                + self.w_smooth * cost_smooth
                + self.w_clear * cost_clear
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_sector = idx
                found_safe = True

        # --- VFH visualization ---
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        radius = self.safety_radius

        for i in range(self.sector_count):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0
            m.ns = "vfh_histogram"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.pose.orientation.w = 1.0

            # Color: green if free, red if blocked
            if histogram[i] == 0:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
            else:
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
            m.color.a = 0.8

            # Sector wedge as line strip (centered at origin)
            start_angle = -math.pi + i * sector_size
            end_angle = start_angle + sector_size

            p0 = Point(x=0.0, y=0.0, z=0.0)
            p1 = Point(
                x=radius * math.cos(start_angle),
                y=radius * math.sin(start_angle),
                z=0.0,
            )
            p2 = Point(
                x=radius * math.cos(end_angle), y=radius * math.sin(end_angle), z=0.0
            )

            m.points = [p0, p1, p2, p0]
            marker_array.markers.append(m)

        self.viz_pub.publish(marker_array)

        # Steering
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # # If stuck, rotate in place to escape corners
        if now_sec < self.recovery_until:
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = self.angular_speed_max
            self.cmd_vel_pub.publish(msg)
            return

        if found_safe:
            steer_angle = (best_sector * sector_size) - math.pi + (sector_size / 2.0)
            msg.twist.angular.z = max(
                min(steer_angle * 2.0, self.angular_speed_max),
                -self.angular_speed_max,
            )
            msg.twist.linear.x = 0.0 if abs(steer_angle) > 0.5 else self.linear_speed
            self.prev_steer_angle = steer_angle
        else:
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = self.angular_speed_max

        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VFHFollower()
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
