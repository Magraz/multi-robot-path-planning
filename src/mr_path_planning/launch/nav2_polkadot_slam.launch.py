#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Launch Stage simulation of the polkadot world with Nav2 + SLAM Toolbox.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


NODENAME = "mr_path_planning"
CONFIG = "polkadot"


def generate_launch_description():

    pkg_dir = get_package_share_directory(NODENAME)
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    slam_params = os.path.join(pkg_dir, "config", "slam_toolbox.yaml")
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

    nav2_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "slam_launch.py")
        ),
        launch_arguments={
            "params_file": slam_params,
            "use_sim_time": "true",
            "autostart": "true",
        }.items(),
    )

    scan_adapter = Node(
        package=NODENAME,
        executable="scan_max_to_inf",
        name="scan_max_to_inf",
        output="screen",
        parameters=[
            {
                "input_scan_topic": "/base_scan",
                "output_scan_topic": "/base_scan_inf",
                "epsilon": 0.005,
                "no_return_intensity_threshold": 0.0,
                "log_every_n_scans": 10,
            }
        ],
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

    return LaunchDescription(
        [
            stage_and_rviz,
            scan_adapter,
            nav2_slam,
            nav2_navigation,
        ]
    )
