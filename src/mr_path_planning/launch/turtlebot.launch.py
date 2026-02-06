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
    TimerAction,
)
from launch_ros.actions import Node
import yaml


def generate_launch_description():

    package_dir = get_package_share_directory("hw2")
    config_file = os.path.join(package_dir, "config", "turtlebot.yaml")
    rviz_config = os.path.join(package_dir, "config", "turtlebot.rviz")

    # Waypoint publisher node with shared config
    waypoint_pub_node = Node(
        package="hw2",
        executable="waypoint_publisher",
        name="waypoint_publisher",
        parameters=[config_file],
        output="screen",
    )

    vfh_follower_node = Node(
        package="hw2",
        executable="vfh_follower",
        name="vfh_follower",
        parameters=[config_file],
        output="screen",
        # prefix=["python3 -m debugpy --listen 5678 --wait-for-client"],
    )

    vfh_delayed = TimerAction(
        period=5.0,
        actions=[vfh_follower_node],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription(
        [
            waypoint_pub_node,
            rviz_node,
            vfh_delayed,
        ]
    )
