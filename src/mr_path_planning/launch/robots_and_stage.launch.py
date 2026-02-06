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
)  # <--- Add TimerAction here
from launch_ros.actions import Node
import yaml


def generate_launch_description():

    # Get directories
    stage_ros2_dir = get_package_share_directory("stage_ros2")
    stage_launch_dir = os.path.join(stage_ros2_dir, "launch")

    package_dir = get_package_share_directory("hw2")
    config_file = os.path.join(package_dir, "config", "params.yaml")

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
    waypoint_node = Node(
        package="hw2",
        executable="waypoint_publisher",
        name="waypoint_publisher",
        parameters=[config_file],
        output="screen",
    )

    map_file = os.path.join(get_package_share_directory("hw2"), "maps", "cave.yaml")

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": map_file},
            {"use_sim_time": True},
            {"frame_id": "map"},
            {"topic_name": "map"},
        ],
    )

    lifecycle_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    # DELAY the lifecycle manager by 2 seconds to let Stage and Map Server initialize
    delayed_lifecycle_manager = TimerAction(period=2.0, actions=[lifecycle_node])

    # Include stage_ros2 demo launch file
    stage_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(stage_launch_dir, "demo.launch.py")),
        launch_arguments={"world": world}.items(),
    )

    return LaunchDescription(
        [
            declare_world_cmd,
            stage_demo,
            waypoint_node,
            # map_server_node,
            # delayed_lifecycle_manager,
        ]
    )
