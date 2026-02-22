#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
)
from launch_ros.actions import Node
import yaml

NODENAME = "mr_path_planning"
CONFIG = "polkadot"


def generate_launch_description():

    # Get directories
    pkg_dir = get_package_share_directory(NODENAME)
    pkg_launch_dir = os.path.join(pkg_dir, "launch")

    # Include stage_ros2 demo launch file
    stage_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_launch_dir, "demo.launch.py")),
        launch_arguments={
            "world": CONFIG,
            "use_stamped_velocity": "true",
        }.items(),
    )

    return LaunchDescription(
        [
            stage_demo,
        ]
    )
