# TurtleBot3 Burger robot definition for Stage

define turtlebot3_base position
(
  color "blue"
  drive "diff"
  gui_nose 1
  obstacle_return 1
  ranger_return 0.5
  blob_return 1
  fiducial_return 1

  localization "odom"
  odom_error [ 0.01 0.01 0.0 0.05 ]  # Lower error, as it's a small robot

  # four DOF kinematics limits
  # [ xmin xmax ymin ymax zmin zmax amin amax ]
  velocity_bounds [-0.22 0.22 0 0 0 0 -162.6 162.6]  # -/+0.22 m/s, -/+2.84 rad/s (deg)
  acceleration_bounds [-0.5 0.5 0 0 0 0 -162.6 162.6]

  # actual size
  size [0.178 0.138 0.192]  # length, width, height

  # center of rotation at geometric center
  origin [0 0 0 0]

  # estimated mass in KG
  mass 1.0
)

define turtlebot3_burger turtlebot3_base
(
  # simplified Body shape:
  block(
    points 4
    point[0] [-0.089 0.069]
    point[1] [-0.089 -0.069]
    point[2] [0.089 -0.069]
    point[3] [0.089 0.069]
    z [0 0.192]
  )
)