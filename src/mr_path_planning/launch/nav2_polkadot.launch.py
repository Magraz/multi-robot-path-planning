#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Launch Stage simulation of the polkadot world with the ROS 2 Nav2 stack.

  - nav2_bringup localization_launch  (map_server + AMCL)
  - nav2_bringup navigation_launch    (planner, controller, bt_navigator, etc.)
  - TimerAction publishes /initialpose after 7 s so AMCL can start particle filtering
    and take ownership of the map->odom TF (tf_broadcast: true in nav2_params.yaml).

Topic mapping from Stage to Nav2:
  /base_scan    -> scan_topic / observation_sources in nav2_params.yaml
  /ground_truth -> odom_topic in nav2_params.yaml
  /cmd_vel      -> published by Nav2 controller (TwistStamped, use_stamped_velocity=true)
"""
import os
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


NODENAME = "mr_path_planning"
CONFIG = "polkadot"


def generate_launch_description():

    pkg_dir = get_package_share_directory(NODENAME)
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    map_yaml = os.path.join(pkg_dir, "world", "bitmaps", "polkadot.yaml")
    nav2_params = os.path.join(pkg_dir, "config", "nav2_params.yaml")

    # Stage + RViz (reuse existing demo launch, no VFH nodes)
    stage_and_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "demo.launch.py")
        ),
        launch_arguments={
            "world": CONFIG,
            "use_stamped_velocity": "true",
        }.items(),
    )

    # Nav2 localization: map_server + AMCL
    nav2_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "localization_launch.py")
        ),
        launch_arguments={
            "map": map_yaml,
            "params_file": nav2_params,
            "use_sim_time": "true",
        }.items(),
    )

    # Nav2 navigation: planner, controller, bt_navigator, behaviors, etc.
    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "params_file": nav2_params,
            "use_sim_time": "true",
        }.items(),
    )

    # Robot starting pose from polkadot.world: pose [-6.59, -4.27, 0, 45 deg]
    # Quaternion for yaw=45°: z=sin(π/8), w=cos(π/8)
    yaw_deg = 45.0
    yaw_rad = math.radians(yaw_deg)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    initial_pose_yaml = (
        "{"
        "header: {frame_id: map}, "
        "pose: {pose: {"
        f"position: {{x: -6.59, y: -4.27, z: 0.0}}, "
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

    # Publish the robot's initial pose to AMCL after Nav2 has had time to start up.
    # AMCL (tf_broadcast: true) will then take ownership of the map->odom TF.
    initial_pose_pub = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "topic", "pub", "--once",
                    "/initialpose",
                    "geometry_msgs/msg/PoseWithCovarianceStamped",
                    initial_pose_yaml,
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            stage_and_rviz,
            nav2_localization,
            nav2_navigation,
            initial_pose_pub,
        ]
    )
