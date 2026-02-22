#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Launch Stage simulation of the polkadot world with the ROS 2 Nav2 stack
for TWO robots (robot_0 and robot_1).

Stage is launched once with:
  enforce_prefixes = true   →  topics prefixed with robot name
                                e.g. /robot_0/base_scan, /robot_1/cmd_vel
  one_tf_tree      = false  →  each robot publishes TF to its own topic
                                /robot_N/tf and /robot_N/tf_static
                                frame ids are NOT prefixed: odom, base_link

Two independent Nav2 stacks are launched, each in its own namespace:
  /robot_0  →  nav2_params_robot_0.yaml
  /robot_1  →  nav2_params_robot_1.yaml

Topic mapping from Stage to Nav2 (per robot, relative names resolve in namespace):
  /<ns>/base_scan    → scan_topic / observation_sources
  /<ns>/ground_truth → odom_topic
  /<ns>/cmd_vel      → published by Nav2 controller (TwistStamped)
  /<ns>/tf           → TF from Stage; Nav2 remaps /tf→tf (relative) so it matches

Initial poses are published to /<ns>/initialpose after 7 s so that AMCL can
start particle filtering and publish the map → odom transform on /<ns>/tf.
"""
import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace


NODENAME = "mr_path_planning"
CONFIG = "polkadot"

# Starting poses from polkadot.world
ROBOTS = [
    {"name": "robot_0", "x": -4.00, "y": -4.00, "yaw_deg": 45.0},
    {"name": "robot_1", "x": 4.00, "y": -4.00, "yaw_deg": 45.0},
]


def make_initial_pose_yaml(x: float, y: float, yaw_deg: float) -> str:
    yaw_rad = math.radians(yaw_deg)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return (
        "{"
        "header: {frame_id: map}, "
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


def generate_launch_description():

    pkg_dir = get_package_share_directory(NODENAME)
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    map_yaml = os.path.join(pkg_dir, "world", "bitmaps", f"{CONFIG}.yaml")

    # ------------------------------------------------------------------
    # Stage + RViz
    #   enforce_prefixes=true   → topics prefixed: /robot_N/base_scan etc.
    #   one_tf_tree=false       → per-robot TF on /robot_N/tf
    # ------------------------------------------------------------------
    stage_and_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "demo.launch.py")
        ),
        launch_arguments={
            "world": CONFIG,
            "enforce_prefixes": "true",
            "one_tf_tree": "false",
            "use_stamped_velocity": "true",
        }.items(),
    )

    actions = [stage_and_rviz]

    multi_params = os.path.join(pkg_dir, "config", "nav2_params_multi.yaml")

    for robot in ROBOTS:
        ns = robot["name"]

        # # Localization: map_server + AMCL
        # localization = IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(nav2_bringup_dir, "launch", "localization_launch.py")
        #     ),
        #     launch_arguments={
        #         "namespace": "",  # empty → RewrittenYaml does no wrapping
        #         "map": map_yaml,
        #         "params_file": multi_params,
        #         "use_sim_time": "true",
        #     }.items(),
        # )

        # # Navigation: planner, controller, bt_navigator, behaviors, etc.
        # # Uses a local copy of navigation_launch.py with docking_server removed.
        # navigation = IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(pkg_dir, "launch", "navigation_custom.launch.py")
        #     ),
        #     launch_arguments={
        #         "namespace": "",  # empty → RewrittenYaml does no wrapping
        #         "params_file": multi_params,
        #         "use_sim_time": "true",
        #     }.items(),
        # )

        # Navigation: planner, controller, bt_navigator, behaviors, etc.
        # Uses a local copy of navigation_launch.py with docking_server removed.
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
                # localization,
                # navigation,
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
                            robot["x"], robot["y"], robot["yaw_deg"]
                        ),
                    ],
                    output="screen",
                )
            ],
        )

        # actions += [nav2_group, initial_pose_pub]

        actions += [nav2_group, initial_pose_pub]

    return LaunchDescription(actions)
