# nav2_minimal.launch.py
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    map_yaml = LaunchConfiguration("map")

    # If you later add namespaces, this remap pattern is what Nav2 uses
    # so TF can live under the namespace (e.g., /robot_0/tf). :contentReference[oaicite:7]{index=7}
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    # --- localization lifecycle nodes (match actual node names) ---
    localization_nodes = ["map_server", "amcl"]

    # --- navigation lifecycle nodes (minimum for NavigateToPose) ---
    navigation_nodes = [
        "controller_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "params_file", description="Path to nav2 params yaml"
            ),
            DeclareLaunchArgument("map", description="Path to map yaml"),
            # Set sim time globally for all nodes in this group
            GroupAction(
                [
                    SetParameter(name="use_sim_time", value=use_sim_time),
                    # -------- Localization --------
                    Node(
                        package="nav2_map_server",
                        executable="map_server",
                        name="map_server",
                        output="screen",
                        parameters=[params_file, {"yaml_filename": map_yaml}],
                        remappings=remappings,
                    ),
                    Node(
                        package="nav2_amcl",
                        executable="amcl",
                        name="amcl",
                        output="screen",
                        parameters=[params_file],
                        remappings=remappings,
                    ),
                    Node(
                        package="nav2_lifecycle_manager",
                        executable="lifecycle_manager",
                        name="lifecycle_manager_localization",
                        output="screen",
                        parameters=[
                            {"autostart": autostart, "node_names": localization_nodes}
                        ],
                    ),
                    # -------- Navigation --------
                    Node(
                        package="nav2_planner",
                        executable="planner_server",
                        name="planner_server",
                        output="screen",
                        parameters=[params_file],
                        remappings=remappings,
                    ),
                    Node(
                        package="nav2_controller",
                        executable="controller_server",
                        name="controller_server",
                        output="screen",
                        parameters=[params_file],
                        # NOTE: no ('cmd_vel' -> 'cmd_vel_nav') remap since we are NOT launching velocity_smoother
                        remappings=remappings,
                    ),
                    Node(
                        package="nav2_behaviors",
                        executable="behavior_server",
                        name="behavior_server",
                        output="screen",
                        parameters=[params_file],
                        remappings=remappings,
                    ),
                    Node(
                        package="nav2_bt_navigator",
                        executable="bt_navigator",
                        name="bt_navigator",
                        output="screen",
                        parameters=[params_file],
                        remappings=remappings,
                    ),
                    Node(
                        package="nav2_lifecycle_manager",
                        executable="lifecycle_manager",
                        name="lifecycle_manager_navigation",
                        output="screen",
                        parameters=[
                            {"autostart": autostart, "node_names": navigation_nodes}
                        ],
                    ),
                ]
            ),
        ]
    )
