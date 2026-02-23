"""Continuously send a fixed waypoint loop to a Nav2 FollowWaypoints server."""

import math
import time
from typing import List, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
from rclpy.node import Node


DEFAULT_WAYPOINTS: List[float] = [
    4.0,
    4.0,
    0.0,
    4.0,
    -4.0,
    0.0,
    -4.0,
    -4.0,
    0.0,
    -4.0,
    4.0,
    0.0,
]


class TargetWaypointPatrol(Node):
    def __init__(self) -> None:
        super().__init__("target_waypoint_patrol")

        self.declare_parameter("robot_name", "target_0")
        self.declare_parameter("loop_patrol", True)
        self.declare_parameter("waypoints", DEFAULT_WAYPOINTS)

        self._robot_name = str(self.get_parameter("robot_name").value)
        self._loop_patrol = bool(self.get_parameter("loop_patrol").value)

        raw_waypoints = list(self.get_parameter("waypoints").value)
        self._waypoints = self._parse_waypoints(raw_waypoints)
        if not self._waypoints:
            self.get_logger().error(
                "No valid waypoints provided. Expected triplets [x, y, yaw_deg]."
            )
            raise RuntimeError("Invalid waypoint configuration")

        self._frame_id = f"{self._robot_name}/map"
        self._action_client = ActionClient(
            self,
            FollowWaypoints,
            f"/{self._robot_name}/follow_waypoints",
        )

        self._state = "idle"
        self._next_send_time = time.monotonic()
        self._last_wait_log_time = 0.0
        self._tick_timer = self.create_timer(0.5, self._tick)

        self.get_logger().info(
            f"Patrol client ready for {self._robot_name} with {len(self._waypoints)} waypoints."
        )

    def _parse_waypoints(
        self, raw_waypoints: Sequence[float]
    ) -> List[Tuple[float, float, float]]:
        if len(raw_waypoints) < 3 or len(raw_waypoints) % 3 != 0:
            return []

        parsed: List[Tuple[float, float, float]] = []
        for idx in range(0, len(raw_waypoints), 3):
            x = float(raw_waypoints[idx])
            y = float(raw_waypoints[idx + 1])
            yaw_deg = float(raw_waypoints[idx + 2])
            parsed.append((x, y, yaw_deg))
        return parsed

    def _tick(self) -> None:
        if self._state != "idle":
            return

        now = time.monotonic()
        if now < self._next_send_time:
            return

        if not self._action_client.wait_for_server(timeout_sec=0.0):
            if now - self._last_wait_log_time > 5.0:
                self._last_wait_log_time = now
                self.get_logger().info(
                    f"Waiting for /{self._robot_name}/follow_waypoints action server..."
                )
            return

        self._send_goal()

    def _send_goal(self) -> None:
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = [self._make_pose(x, y, yaw) for x, y, yaw in self._waypoints]

        # Keep one waypoint cycle per goal for predictable continuous patrol loops.
        if hasattr(goal_msg, "number_of_loops"):
            goal_msg.number_of_loops = 0
        if hasattr(goal_msg, "goal_index"):
            goal_msg.goal_index = 0

        self._state = "sending"
        send_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self._on_feedback
        )
        send_future.add_done_callback(self._on_goal_response)
        self.get_logger().info("Sent waypoint goal to Nav2 waypoint follower.")

    def _make_pose(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        yaw_rad = math.radians(yaw_deg)
        qz = math.sin(yaw_rad / 2.0)
        qw = math.cos(yaw_rad / 2.0)

        pose = PoseStamped()
        pose.header.frame_id = self._frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to send waypoint goal: {exc}")
            self._state = "idle"
            self._next_send_time = time.monotonic() + 2.0
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Waypoint goal was rejected by Nav2.")
            self._state = "idle"
            self._next_send_time = time.monotonic() + 2.0
            return

        self._state = "running"
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            wrapped_result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Error waiting for waypoint result: {exc}")
            wrapped_result = None

        if wrapped_result is None:
            self._next_send_time = time.monotonic() + 2.0
        else:
            missed_count = len(wrapped_result.result.missed_waypoints)
            self.get_logger().info(
                f"Waypoint cycle finished (status={wrapped_result.status}, missed={missed_count})."
            )
            self._next_send_time = time.monotonic() + 1.0

        if self._loop_patrol:
            self._state = "idle"
            return

        self.get_logger().info("Loop disabled; patrol node will stay idle.")
        self._state = "done"

    def _on_feedback(self, feedback_msg) -> None:
        current_idx = feedback_msg.feedback.current_waypoint
        self.get_logger().debug(f"Current waypoint index: {current_idx}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetWaypointPatrol()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

