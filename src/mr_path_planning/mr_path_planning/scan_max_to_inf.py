#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanMaxToInf(Node):
    def __init__(self):
        super().__init__("scan_max_to_inf")

        self.declare_parameter("input_scan_topic", "/base_scan")
        self.declare_parameter("output_scan_topic", "/base_scan_inf")
        self.declare_parameter("epsilon", 1e-3)
        self.declare_parameter("no_return_intensity_threshold", 0.0)
        self.declare_parameter("log_every_n_scans", 10)

        self.input_scan_topic = self.get_parameter("input_scan_topic").value
        self.output_scan_topic = self.get_parameter("output_scan_topic").value
        self.epsilon = float(self.get_parameter("epsilon").value)
        self.no_return_intensity_threshold = float(
            self.get_parameter("no_return_intensity_threshold").value
        )
        self.log_every_n_scans = int(self.get_parameter("log_every_n_scans").value)
        self.scan_count = 0
        self.total_modified = 0
        self.total_modified_by_range = 0
        self.total_modified_by_intensity = 0

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.scan_pub = self.create_publisher(
            LaserScan,
            self.output_scan_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Converting max-range/zero-intensity LaserScan readings to inf: "
            f"{self.input_scan_topic} -> {self.output_scan_topic} "
            f"(epsilon={self.epsilon}, "
            f"no_return_intensity_threshold={self.no_return_intensity_threshold}, "
            f"log_every_n_scans={self.log_every_n_scans})"
        )

    def scan_callback(self, msg: LaserScan):
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max

        max_threshold = msg.range_max - self.epsilon
        has_intensities = len(msg.intensities) == len(msg.ranges)
        out.ranges = []
        out.intensities = list(msg.intensities) if has_intensities else []
        modified_this_scan = 0
        modified_by_range_this_scan = 0
        modified_by_intensity_this_scan = 0
        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r):
                out.ranges.append(r)
                continue

            no_return_by_range = r >= max_threshold
            no_return_by_intensity = (
                has_intensities
                and msg.intensities[i] <= self.no_return_intensity_threshold
            )

            converted_to_inf = no_return_by_range or no_return_by_intensity
            out.ranges.append(float("inf") if converted_to_inf else r)
            if converted_to_inf:
                modified_this_scan += 1
                if no_return_by_range:
                    modified_by_range_this_scan += 1
                if no_return_by_intensity:
                    modified_by_intensity_this_scan += 1
                if has_intensities:
                    out.intensities[i] = 1.0

        self.scan_pub.publish(out)

        self.scan_count += 1
        self.total_modified += modified_this_scan
        self.total_modified_by_range += modified_by_range_this_scan
        self.total_modified_by_intensity += modified_by_intensity_this_scan
        if self.log_every_n_scans > 0 and self.scan_count % self.log_every_n_scans == 0:
            self.get_logger().info(
                "scan %d: modified=%d (range=%d, intensity=%d) | total_modified=%d "
                "(range=%d, intensity=%d)"
                % (
                    self.scan_count,
                    modified_this_scan,
                    modified_by_range_this_scan,
                    modified_by_intensity_this_scan,
                    self.total_modified,
                    self.total_modified_by_range,
                    self.total_modified_by_intensity,
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = ScanMaxToInf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
