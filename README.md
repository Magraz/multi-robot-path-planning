# Multi-Robot Search and Capture

Two searcher robots (`robot_0`, `robot_1`) use a MILP planner to capture a mobile target (`target_0`) in a mapped environment. The environment is represented as a GML graph (nodes = waypoints, edges = traversable connections). ROS 2 Nav2 executes all physical motion.

---

## How it works

### Graph and map
Each world has a bitmap map (`.pgm`) and a GML graph of waypoints derived from it. The graph captures the topology of free space — corridors, junctions, rooms.

### Searcher planning (MILP)
Every replan cycle (~12 s), `milp_graph_search.py` solves a Mixed-Integer Linear Program (MILP) using the MESPP formulation:

- **Decision variables**: binary presence `x[t][s][v]` — is searcher `s` at node `v` at time step `t`?
- **Constraints**:
  - Each searcher occupies exactly one node per time step.
  - Movement is only allowed along graph edges (one hop per step).
  - The target moves with uniform probability over its graph neighbours.
- **Objective**: maximise the probability that at least one searcher shares a node with the target within a planning horizon of `H` steps (default `H=10`).

The MILP returns an optimal route for each searcher. Routes are converted to world coordinates and dispatched to Nav2's `FollowWaypoints` action server.

### Target motion
`target_graph_uniform.py` drives `target_0` by picking a random graph neighbour every 2 cycles and issuing a `NavigateToPose` goal. The target moves slower than the searchers to keep the problem tractable.

### Capture and stop conditions
After each replan tick the planner checks:

| Condition | Trigger | Action |
|---|---|---|
| **Capture** | A searcher is within 0.6 m of the target | Publish `/search_capture/stop`; cancel all goals; send the non-capturing searcher to the capture location |
| **Block** | All graph neighbours of the target node are occupied by searchers | Publish `/search_capture/stop`; all robots halt |

### Metrics
`search_metrics_logger.py` tracks every run and appends one row to `~/.ros/search_metrics.csv`:

| Column | Description |
|---|---|
| `timestamp` | ISO timestamp of the run |
| `outcome` | `CAPTURED` or `STOPPED` |
| `elapsed_sec` | Time from start to stop |
| `unique_nodes_all` | Distinct graph nodes visited across all searchers |
| `nodes_robot_0/1` | Nodes visited per searcher |

View results:
```bash
column -t -s',' ~/.ros/search_metrics.csv
```

---

## Setup

```bash
cd mrpp_ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

Available worlds: `polkadot`, `graf201`, `hospital`, `my_office`, `big_office`, `more_office`

```bash
ros2 launch mr_path_planning nav2_multi.launch.py world:=my_office
```

The graph overlay in RViz is on by default. To disable it:
```bash
ros2 launch mr_path_planning nav2_multi.launch.py world:=my_office enable_graph_markers:=false
```

### Kill all processes
```bash
pkill -9 -f "mr_path_planning"; pkill -9 -f "nav2"; pkill -9 stage_ros2; pkill -9 rviz2
```

### Standalone MILP (no ROS 2)
```bash
cd src/search_and_capture_algo
python3 code/main.py --graph ../mr_path_planning/world/bitmaps/my_office_sparse.gml --horizon 10 --searchers 2 --motion uniform
python3 animate_results.py --results-dir results --output results/my_office.mp4 --fps 3
```
