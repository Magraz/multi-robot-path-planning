# Multi Robot Path Planning

## Setup

Make sure to place the ``src`` directory into a ros2 kilted workspace, then:

```bash
cd <ROS2_WORKSPACE>
colcon build --symlink-install
source install/setup.bash
```

### Run Nav2 on polkadot world

To run the robot navigation in the polkadot environment. This instantiates 2 robots, and a moving robot target.

```bash
ros2 launch hw2 nav2_multi.launch.py world:=graf201
```

To start the target chase call the ``/chase_target/enable`` service