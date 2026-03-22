#!/bin/bash
# Helper script to activate robot_0's bt_navigator if it fails during launch

echo "Checking robot_0 bt_navigator state..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

STATE=$(ros2 lifecycle get /robot_0/bt_navigator 2>/dev/null)

if [ "$STATE" = "unconfigured [1]" ]; then
    echo "robot_0 bt_navigator is unconfigured. Activating..."
    ros2 lifecycle set /robot_0/bt_navigator configure
    sleep 1
    ros2 lifecycle set /robot_0/bt_navigator activate
    echo "robot_0 bt_navigator activated!"
elif [ "$STATE" = "inactive [2]" ]; then
    echo "robot_0 bt_navigator is inactive. Activating..."
    ros2 lifecycle set /robot_0/bt_navigator activate
    echo "robot_0 bt_navigator activated!"
else
    echo "robot_0 bt_navigator state: $STATE (already active or configured)"
fi
