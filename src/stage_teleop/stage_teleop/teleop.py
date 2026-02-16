#!/usr/bin/env python3
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

# Linux terminal raw key reading
import termios
import tty
import select


def get_key_nonblocking(timeout_s: float = 0.05) -> str:
    """
    Returns a single character if available within timeout, else ''.
    """
    dr, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if dr:
        return sys.stdin.read(1)
    return ""


class WASDTeleop(Node):
    def __init__(self):
        super().__init__("wasd_teleop")
        self.pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        # Parameters you can override:
        self.declare_parameter("linear_speed", 5.0)  # m/s
        self.declare_parameter("angular_speed", 3.0)  # rad/s
        self.declare_parameter("publish_rate", 20.0)  # Hz
        self.declare_parameter(
            "hold_timeout", 0.25
        )  # seconds to keep last cmd if no new key
        self.declare_parameter("frame_id", "base_link")

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.hold_timeout = float(self.get_parameter("hold_timeout").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self._lock = threading.Lock()
        self._last_twist = Twist()
        self._last_cmd_time = time.time()

        # Publisher timer
        period = 1.0 / max(self.publish_rate, 1e-6)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            "WASD teleop started: w/s linear, a/d angular, space stop, q quit.\n"
            f"Publishing TwistStamped to /cmd_vel @ {self.publish_rate} Hz "
            f"(linear_speed={self.linear_speed}, angular_speed={self.angular_speed})."
        )

    def set_cmd(self, lin_x: float, ang_z: float):
        with self._lock:
            twist = Twist()
            twist.linear.x = lin_x
            twist.angular.z = ang_z
            self._last_twist = twist
            self._last_cmd_time = time.time()

    def stop(self):
        self.set_cmd(0.0, 0.0)

    def _on_timer(self):
        # If we haven't received a key recently, publish stop (safety)
        with self._lock:
            age = time.time() - self._last_cmd_time
            twist = self._last_twist

        if age > self.hold_timeout:
            twist = Twist()  # zeros

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist = twist

        self.pub.publish(msg)


def main():
    rclpy.init()

    # Save terminal settings and set raw mode
    if not sys.stdin.isatty():
        print("This teleop node must be run in a terminal (TTY).")
        return

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    node = WASDTeleop()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            key = get_key_nonblocking(timeout_s=0.05)

            if not key:
                continue

            key = key.lower()

            lin = 0.0
            ang = 0.0

            if key == "w":
                lin = node.linear_speed
            elif key == "s":
                lin = -node.linear_speed
            elif key == "a":
                ang = node.angular_speed
            elif key == "d":
                ang = -node.angular_speed
            elif key == " ":
                node.stop()
                continue
            elif key == "q":
                node.get_logger().info("Quit.")
                break
            else:
                # ignore other keys
                continue

            node.set_cmd(lin, ang)

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.stop()
        # Give a moment to publish stop
        rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
