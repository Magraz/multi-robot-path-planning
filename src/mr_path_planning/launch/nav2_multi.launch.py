#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Launch Stage simulation with the ROS 2 Nav2 stack
for THREE robots (robot_0, robot_1 and target_0).

Stage is launched once with:
  enforce_prefixes = true   →  topics prefixed with robot name
                                e.g. /robot_0/base_scan, /robot_1/cmd_vel
  one_tf_tree      = true   →  all robots publish TF to global /tf and /tf_static
                                frame ids ARE prefixed: robot_0/odom, robot_0/base_link

Three independent Nav2 stacks are launched, each in its own namespace:
  /robot_0  →  nav2_params_multi.yaml (robot_0 section)
  /robot_1  →  nav2_params_multi.yaml (robot_1 section)
  /target_0 →  nav2_params_multi.yaml (target_0 section)

Topic mapping from Stage to Nav2 (per robot, relative names resolve in namespace):
  /<ns>/base_scan    → scan_topic / observation_sources
  /<ns>/ground_truth → odom_topic
  /<ns>/cmd_vel      → published by Nav2 controller (TwistStamped)
  /tf                → shared TF tree with prefixed frame ids

Initial poses are published to /<ns>/initialpose after 7 s so that AMCL can
start particle filtering and publish the map → odom transform on /tf.

Usage:
  ros2 launch mr_path_planning nav2_polkadot_multi.launch.py              # polkadot (default)
  ros2 launch mr_path_planning nav2_polkadot_multi.launch.py world:=graf201
"""
import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


NODENAME = "mr_path_planning"

# Per-world configuration: robot starting poses and target patrol waypoints.
WORLD_CONFIGS = {
    "polkadot": {
        "robots": [
            {"name": "robot_0", "x": -4.00, "y": -4.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": 4.00, "y": -4.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 0.00, "y": 0.00, "yaw_deg": 45.0},
        ],
    },
    "graf201": {
        "robots": [
            {"name": "robot_0", "x": -6.00, "y": -4.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": 6.00, "y": -4.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 0.00, "y": 0.00, "yaw_deg": 45.0},
        ],
    },
    "hospital": {
        "robots": [
            {"name": "robot_0", "x": -34.00, "y": -11.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": 34.00, "y": -17.00, "yaw_deg": 45.0},
            # {"name": "target_0", "x": 35.50, "y": 3.50, "yaw_deg": 45.0},
            # {"name": "target_0", "x": 24, "y": 14, "yaw_deg": 45.0},
            # {"name": "target_0", "x": -2.5, "y": -2.0, "yaw_deg": 45.0},
            # {"name": "target_0", "x": -17.0, "y": 3.0, "yaw_deg": 45.0},
            {"name": "target_0", "x": -32.0, "y": 14.5, "yaw_deg": 45.0},
        ],
    },
    "office_map": {
        "robots": [
            {"name": "robot_0", "x": -6.00, "y": -4.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": 6.00, "y": -4.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 0.00, "y": 0.00, "yaw_deg": 45.0},
        ],
    },
    "my_office": {
        "robots": [
            {"name": "robot_0", "x": -19.00, "y": 2.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": -5.70, "y": 2.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 7.60, "y": 0.00, "yaw_deg": 45.0},
        ],
    },
    "big_office": {
        "robots": [
            {"name": "robot_0", "x": -19.00, "y": 2.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": -5.70, "y": 2.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 15.00, "y": -10.00, "yaw_deg": 45.0},
        ],
    },
}


def make_initial_pose_yaml(ns: str, x: float, y: float, yaw_deg: float) -> str:
    yaw_rad = math.radians(yaw_deg)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return (
        "{"
        f"header: {{frame_id: {ns}/map}}, "
        "pose: {pose: {"
        f"position: {{x: {x}, y: {y}, z: 0.0}}, "
        f"orientation: {{x: 0.0, y: 0.0, z: {qz:.5f}, w: {qw:.5f}}}"
        "}, covariance: ["
        "0.25, 0.0, 0.0, 0.0, 0.0, 0.0, "
        "0.0, 0.25, 0.0, 0.0, 0.0, 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.06853"
        "]}"
        "}"
    )


def launch_setup(context):
    world = LaunchConfiguration("world").perform(context)
    config = WORLD_CONFIGS[world]
    robots = config["robots"]

    pkg_dir = get_package_share_directory(NODENAME)
    map_yaml = os.path.join(pkg_dir, "world", "bitmaps", f"{world}.yaml")

    # ------------------------------------------------------------------
    # Stage + RViz
    #   enforce_prefixes=true   → topics prefixed: /robot_N/base_scan etc.
    #   one_tf_tree=true        → shared TF tree on /tf with prefixed frames
    # ------------------------------------------------------------------
    stage_and_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "demo.launch.py")
        ),
        launch_arguments={
            "world": world,
            "enforce_prefixes": "true",
            "one_tf_tree": "true",
            "use_stamped_velocity": "true",
        }.items(),
    )

    # Relay /goal_pose from RViz to each robot's namespaced goal_pose topic
    goal_relay = Node(
        package=NODENAME,
        executable="goal_relay",
        name="goal_relay",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # Periodically send target_0's position as a goal to robot_0 and robot_1
    chase_target = Node(
        package=NODENAME,
        executable="chase_target",
        name="chase_target",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"target_name": "target_0"},
            {"period": 5.0},
        ],
    )

    # actions = [stage_and_rviz, goal_relay, target_patrol, chase_target]

    # Visualize the world's GML graph in RViz (if the .gml file exists)
    graph_file = os.path.join(pkg_dir, "world", "bitmaps", f"{world}.gml")
    graph_visualizer = Node(
        package=NODENAME,
        executable="graph_visualizer",
        name="graph_visualizer",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"graph_file": graph_file},
            {"map_yaml": map_yaml},
            {"frame_id": "robot_0/map"},
        ],
    )

    actions = [stage_and_rviz, goal_relay, chase_target, graph_visualizer]

    multi_params = os.path.join(pkg_dir, "config", "nav2_params_multi.yaml")

    for robot in robots:
        ns = robot["name"]

        nav2_min_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, "launch", "nav2_minimal.launch.py")
            ),
            launch_arguments={
                "map": map_yaml,
                "params_file": multi_params,
                "use_sim_time": "true",
            }.items(),
        )

        # Wrap both Nav2 sub-stacks in a namespace group so every spawned
        # node is placed in /robot_N/... automatically.
        nav2_group = GroupAction(
            [
                PushRosNamespace(ns),
                nav2_min_launch,
            ]
        )

        # Publish initial pose to /<ns>/initialpose after Nav2 has started.
        # AMCL (relative subscriber) sees this as its own /initialpose once
        # the namespace is pushed.
        initial_pose_pub = TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "topic",
                        "pub",
                        "--once",
                        f"/{ns}/initialpose",
                        "geometry_msgs/msg/PoseWithCovarianceStamped",
                        make_initial_pose_yaml(
                            ns, robot["x"], robot["y"], robot["yaw_deg"]
                        ),
                    ],
                    output="screen",
                )
            ],
        )

        actions += [nav2_group, initial_pose_pub]

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value="polkadot",
                description="World name (polkadot or graf201)",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
