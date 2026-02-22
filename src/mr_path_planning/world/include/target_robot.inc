# Target robot definition for Stage
# No sensors — acts as a moving obstacle detectable by other robots' lidars.

define target_robot position
(
  color "red"
  drive "diff"
  gui_nose 1
  obstacle_return 1
  ranger_return 1.0

  localization "odom"
  odom_error [ 0.001 0.001 0.0 0.01 ]

  velocity_bounds [-0.22 0.22 0 0 0 0 -162.6 162.6]
  acceleration_bounds [-0.5 0.5 0 0 0 0 -162.6 162.6]

  size [0.178 0.138 0.192]
  origin [0 0 0 0]
  mass 1.0

  block(
    points 4
    point[0] [-0.089  0.069]
    point[1] [-0.089 -0.069]
    point[2] [ 0.089 -0.069]
    point[3] [ 0.089  0.069]
    z [0 0.192]
  )
)
