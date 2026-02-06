# ROB599 Homework 2

This repository the code for assignment 2 of ROB599. I used the VFH algorithm.

## Writeup

The write up is located at:

`HW2_Individual Assignment_Writeup.pdf`

## Video Demonstration

Video submission can be found at:

`/videos/rob599_hw2.mp4`

## Setup

Make sure to place the ``src`` directory into a ros2 kilted workspace, then:

```bash
cd <ROS2_WORKSPACE>
colcon build
source install/setup.bash
```

## Running the Simulations

### Polkadot World

To run the robot navigation in the Polkadot environment:


```bash
ros2 launch hw2 polkadot.launch.py

```

### Graf201 World

To run the robot navigation in the Graf201 environment:

```bash
ros2 launch hw2 graf201.launch.py
```

## Starting the VFH Follower

After launching either world, start the VFH follower by calling the service:

```bash
ros2 service call /start_vfh_follower std_srvs/srv/SetBool "{data: true}"
```