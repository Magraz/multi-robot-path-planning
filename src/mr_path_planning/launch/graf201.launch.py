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
CONFIG = "graf201"


def generate_launch_description():

    # Get directories
    pkg_dir = get_package_share_directory(NODENAME)
    pkg_launch_dir = os.path.join(pkg_dir, "launch")

    config_file = os.path.join(pkg_dir, "config", f"{CONFIG}.yaml")

    # Load YAML config to extract world_name
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        world_from_yaml = config["waypoint_publisher"]["ros__parameters"]["world_name"]

    # World can be overridden from command line, but defaults to YAML value
    world = LaunchConfiguration("world")
    declare_world_cmd = DeclareLaunchArgument(
        "world",
        default_value=world_from_yaml,
        description="World to load in stage (from config or override)",
    )

    # Waypoint publisher node with shared config
    waypoint_pub_node = Node(
        package=NODENAME,
        executable="waypoint_publisher",
        name="waypoint_publisher",
        parameters=[config_file],
        output="screen",
    )

    vfh_follower_node = Node(
        package=NODENAME,
        executable="vfh_follower",
        name="vfh_follower",
        parameters=[config_file],
        output="screen",
        # prefix=["python3 -m debugpy --listen 5678 --wait-for-client"],
    )

    map_pub_node = Node(
        package=NODENAME,
        executable="map_publisher",
        name="map_publisher",
        parameters=[config_file],
        output="screen",
    )

    # Include stage_ros2 demo launch file
    stage_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_launch_dir, "demo.launch.py")),
        launch_arguments={
            "world": world,
            "use_stamped_velocity": "true",
        }.items(),
    )

    return LaunchDescription(
        [
            declare_world_cmd,
            stage_demo,
            map_pub_node,
            waypoint_pub_node,
            vfh_follower_node,
            # waypoint_follower_node,
        ]
    )
