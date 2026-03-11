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
from launch.conditions import IfCondition
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
            {"name": "robot_0", "x": -8.00, "y": -6.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": 8.00, "y": -6.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 0.00, "y": 8.00, "yaw_deg": 45.0},
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
    "world_1": {
        "robots": [
            {"name": "robot_0", "x": -18.00, "y": -5.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": -4.00, "y": -5.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 0.00, "y": 8.00, "yaw_deg": 45.0},
        ],
    },
    "world_2": {
        "robots": [
            {"name": "robot_0", "x": -19.00, "y": 2.00, "yaw_deg": 45.0},
            {"name": "robot_1", "x": -5.70, "y": 2.00, "yaw_deg": 45.0},
            {"name": "target_0", "x": 15.00, "y": -10.00, "yaw_deg": 45.0},
        ],
    },
    "world_3": {
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
    enable_graph_viz = LaunchConfiguration("enable_graph_viz")
    enable_graph_markers = LaunchConfiguration("enable_graph_markers")
    graph_viz_rotation_deg = LaunchConfiguration("graph_viz_rotation_deg")
    config = WORLD_CONFIGS[world]
    robots = config["robots"]

    pkg_dir = get_package_share_directory(NODENAME)
    map_yaml = os.path.join(pkg_dir, "world", "bitmaps", f"{world}.yaml")
    graph_sparse = os.path.join(pkg_dir, "world", "bitmaps", f"{world}_sparse.gml")
    graph_dense = os.path.join(pkg_dir, "world", "bitmaps", f"{world}.gml")
    graph_path = graph_sparse if os.path.exists(graph_sparse) else graph_dense

    # ------------------------------------------------------------------
    # Stage simulator
    #   enforce_prefixes=true   → topics prefixed: /robot_N/base_scan etc.
    #   one_tf_tree=true        → shared TF tree on /tf with prefixed frames
    # ------------------------------------------------------------------
    stage = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "stage.launch.py")
        ),
        launch_arguments={
            "world": world,
            "enforce_prefixes": "true",
            "one_tf_tree": "true",
            "use_stamped_velocity": "true",
        }.items(),
    )

    # RViz with a single shared config for all worlds
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "rviz.launch.py")
        ),
        launch_arguments={
            "config": "nav2_multi",
        }.items(),
    )

    target_graph_uniform = Node(
        package=NODENAME,
        executable="target_graph_uniform",
        name="target_graph_uniform",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"target_name": "target_0"},
            {"graph_path": graph_path},
            {"map_yaml": map_yaml},
            {"period": 4.0},
            {"move_every_n_cycles": 2},
            {"seed": 0},
        ],
    )

    # High-level graph routing with MILP; Nav2 executes waypoint motion.
    # pkg_dir is <ws>/install/<pkg>/share/<pkg>; walk up to workspace root.
    ws_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir)))
    )
    mespp_code_path = os.path.join(ws_root, "src", "search_and_capture_algo", "code")

    milp_graph_search = Node(
        package=NODENAME,
        executable="milp_graph_search",
        name="milp_graph_search",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"enabled": True},
            {"graph_path": graph_path},
            {"map_yaml": map_yaml},
            {"horizon": 10},
            {"replan_period": 4.0},
            {"capture_distance": 1.0},
            {"searcher_names": ["robot_0", "robot_1"]},
            {"target_name": "target_0"},
            {"mespp_code_path": mespp_code_path},
            {"solver_time_limit": 5.0},
        ],
    )

    graph_viz = Node(
        package=NODENAME,
        executable="realtime_graph_visualizer",
        name="realtime_graph_visualizer",
        output="screen",
        condition=IfCondition(enable_graph_viz),
        parameters=[
            {"use_sim_time": True},
            {"graph_path": graph_path},
            {"map_yaml": map_yaml},
            {"searcher_names": ["robot_0", "robot_1"]},
            {"target_name": "target_0"},
            {"period": 0.5},
            {"rotate_graph_deg": graph_viz_rotation_deg},
            {"pixel_order": "rc"},
        ],
    )

    graph_markers = Node(
        package=NODENAME,
        executable="graph_visualizer",
        name="graph_visualizer",
        output="screen",
        condition=IfCondition(enable_graph_markers),
        parameters=[
            {"use_sim_time": True},
            {"graph_path": graph_path},
            {"map_yaml": map_yaml},
            {"frame_id": "robot_0/map"},
        ],
    )

    robot_markers = Node(
        package=NODENAME,
        executable="robot_markers",
        name="robot_markers",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"searcher_names": ["robot_0", "robot_1"]},
            {"target_name": "target_0"},
            {"frame_id": "robot_0/map"},
        ],
    )

    csv_path = os.path.join(ws_root, "results", f"search_metrics_{world}.csv")

    search_metrics_logger = Node(
        package=NODENAME,
        executable="search_metrics_logger",
        name="search_metrics_logger",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"graph_path": graph_path},
            {"map_yaml": map_yaml},
            {"metrics_csv": csv_path},
            {"searcher_names": ["robot_0", "robot_1"]},
            {"target_name": "target_0"},
        ],
    )

    actions = [
        stage,
        rviz,
        target_graph_uniform,
        milp_graph_search,
        search_metrics_logger,
        graph_viz,
        graph_markers,
        robot_markers,
    ]

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
                description="World name (polkadot, graf201, hospital, world_1, world_2, world_3)",
            ),
            DeclareLaunchArgument(
                "enable_graph_viz",
                default_value="false",
                description="Enable realtime map+graph+belief visualization window",
            ),
            DeclareLaunchArgument(
                "enable_graph_markers",
                default_value="true",
                description="Publish graph MarkerArray for RViz",
            ),
            DeclareLaunchArgument(
                "graph_viz_rotation_deg",
                default_value="0.0",
                description="Rotation (degrees) applied only to graph visualization",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
