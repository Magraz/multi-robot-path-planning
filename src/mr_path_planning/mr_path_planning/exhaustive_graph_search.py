import math
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


class ExhaustiveGraphSearch(Node):
    """Coordinated exhaustive graph search: two robots sweep all nodes to find a target."""

    def __init__(self) -> None:
        super().__init__("exhaustive_graph_search")

        self.declare_parameter("enabled", True)
        self.declare_parameter("graph_path", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("replan_period", 4.0)
        self.declare_parameter("capture_distance", 1.0)
        self.declare_parameter("searcher_names", ["robot_0", "robot_1"])
        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("node_reach_distance", 1.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.graph_path = str(self.get_parameter("graph_path").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.replan_period = float(self.get_parameter("replan_period").value)
        self.capture_distance = float(self.get_parameter("capture_distance").value)
        self.searchers = list(self.get_parameter("searcher_names").value)
        self.target_name = str(self.get_parameter("target_name").value)
        self.node_reach_distance = float(self.get_parameter("node_reach_distance").value)

        self.latest_odom: Dict[str, Optional[Odometry]] = {name: None for name in self.searchers}
        self.latest_odom[self.target_name] = None

        self.graph: Optional[ig.Graph] = None
        self.node_world_xy: Dict[int, Tuple[float, float]] = {}

        # Shared visited set and per-robot current goal.
        self.visited: Set[int] = set()
        self.current_goal_node: Dict[str, Optional[int]] = {name: None for name in self.searchers}
        self.sweep_count = 0

        self.follow_clients: Dict[str, ActionClient] = {
            name: ActionClient(self, FollowWaypoints, f"/{name}/follow_waypoints")
            for name in self.searchers
        }
        self.active_goal_handles: Dict[str, object] = {}

        self.stop_pub = self.create_publisher(Bool, "/search_capture/stop", 10)
        self.capture_pose_pub = self.create_publisher(PoseStamped, "/search_capture/location", 10)

        self.cmd_vel_pubs: Dict[str, object] = {}
        for name in list(self.searchers) + [self.target_name]:
            self.cmd_vel_pubs[name] = self.create_publisher(
                TwistStamped, f"/{name}/cmd_vel", 10
            )

        self._load_graph_and_coordinates()

        for name in self.latest_odom.keys():
            self.create_subscription(Odometry, f"/{name}/ground_truth", self._odom_cb(name), 10)

        # Use a wall-clock timer so it fires regardless of sim clock state.
        wall_clock = rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        self.timer = self.create_timer(self.replan_period, self._tick, clock=wall_clock)
        self.stopped = False

        self.get_logger().info(
            f"Exhaustive graph search ready | graph={self.graph_path} "
            f"| nodes={self.graph.vcount()} | searchers={self.searchers} | target={self.target_name}"
        )

    # ------------------------------------------------------------------
    # Odometry & fast capture
    # ------------------------------------------------------------------

    def _odom_cb(self, name: str):
        got_first = [False]

        def cb(msg: Odometry):
            if not got_first[0]:
                got_first[0] = True
                self.get_logger().info(f"First odom received for {name}")
            self.latest_odom[name] = msg
            if not self.stopped and self.graph is not None:
                self._fast_capture_check()
                if name in self.searchers:
                    self._mark_visited_near(name)

        return cb

    def _fast_capture_check(self) -> None:
        if any(self.latest_odom[n] is None for n in self.latest_odom):
            return

        target_xy = self._xy_from_odom(self.latest_odom[self.target_name])
        for name in self.searchers:
            sx, sy = self._xy_from_odom(self.latest_odom[name])
            if math.dist((sx, sy), target_xy) <= self.capture_distance:
                self.get_logger().info(f"Capture condition met by {name}")
                self._handle_capture(name)
                return

    def _mark_visited_near(self, robot: str) -> None:
        """Mark the nearest graph node as visited when a robot is close enough."""
        odom = self.latest_odom.get(robot)
        if odom is None:
            return
        xy = self._xy_from_odom(odom)
        nearest = self._nearest_node(xy)
        nx, ny = self.node_world_xy[nearest]
        if math.dist(xy, (nx, ny)) <= self.node_reach_distance:
            if nearest not in self.visited:
                self.visited.add(nearest)
                self.get_logger().info(
                    f"{robot} visited node {nearest} | "
                    f"{len(self.visited)}/{self.graph.vcount()} explored"
                )

    # ------------------------------------------------------------------
    # Planning tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if not self.enabled:
            return
        if self.stopped:
            return

        if self.graph is None:
            return

        missing = [name for name in self.latest_odom if self.latest_odom[name] is None]
        if missing:
            self.get_logger().info(f"Waiting for odometry from: {missing}")
            return

        event, capturer = self._check_capture_or_blocked()
        if event == "capture":
            self._handle_capture(capturer)
            return
        if event == "blocked":
            self._handle_blocked()
            return

        # Check if full sweep is done; if so, reset.
        if len(self.visited) >= self.graph.vcount():
            self.sweep_count += 1
            self.get_logger().info(
                f"Sweep {self.sweep_count} complete — all {self.graph.vcount()} nodes visited. Resetting."
            )
            self.visited.clear()

        self.get_logger().info(
            f"Tick | visited={len(self.visited)}/{self.graph.vcount()}"
        )

        # Assign next target node to each searcher.
        for robot in self.searchers:
            xy = self._xy_from_odom(self.latest_odom[robot])
            current_node = self._nearest_node(xy)
            self.visited.add(current_node)

            next_node = self._pick_next_node(robot, current_node)
            if next_node is None:
                self.get_logger().info(f"{robot}: no unvisited nodes to assign")
                continue

            # Skip if already navigating to the same node.
            if (next_node == self.current_goal_node.get(robot)
                    and robot in self.active_goal_handles):
                continue

            path_nodes = self._shortest_path_nodes(current_node, next_node)
            if len(path_nodes) <= 1:
                continue

            self.current_goal_node[robot] = next_node
            poses = [self._node_to_pose(robot, n) for n in path_nodes[1:]]
            self._send_follow_waypoints(robot, poses)

    def _pick_next_node(self, robot: str, current_node: int) -> Optional[int]:
        """Pick the closest unvisited node via graph shortest path, avoiding the other robot's goal."""
        other_goals = set()
        for name in self.searchers:
            if name != robot:
                g = self.current_goal_node.get(name)
                if g is not None and g not in self.visited:
                    other_goals.add(g)

        all_nodes = set(range(self.graph.vcount()))
        unvisited = all_nodes - self.visited - {current_node}

        # Prefer nodes not targeted by the other robot.
        candidates = unvisited - other_goals
        if not candidates:
            candidates = unvisited
        if not candidates:
            return None

        # BFS distances from current_node.
        distances = self.graph.shortest_paths(source=current_node, weights=None)[0]

        best_node = None
        best_dist = float("inf")
        for n in candidates:
            d = distances[n]
            if d < best_dist:
                best_dist = d
                best_node = n

        return best_node

    def _shortest_path_nodes(self, src: int, dst: int) -> List[int]:
        """Return the list of node indices on the shortest graph path from src to dst."""
        path = self.graph.get_shortest_paths(src, to=dst, weights=None, output="vpath")
        if path and path[0]:
            return path[0]
        return [src]

    # ------------------------------------------------------------------
    # Nav2 action interface (same as milp_graph_search)
    # ------------------------------------------------------------------

    def _send_follow_waypoints(self, robot: str, poses: List[PoseStamped]) -> None:
        client = self.follow_clients[robot]
        if not client.server_is_ready():
            self.get_logger().warn(f"/{robot}/follow_waypoints action server not ready")
            return

        # Send directly; Nav2's SimpleActionServer preempts the old goal.
        goal = FollowWaypoints.Goal()
        goal.poses = poses
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f, name=robot: self._on_goal_response(name, f))
        self.get_logger().info(f"Sent {len(poses)} waypoints to {robot}")

    def _on_goal_response(self, robot: str, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to send waypoint goal to {robot}: {exc}")
            self.current_goal_node[robot] = None
            return

        if not goal_handle.accepted:
            self.get_logger().warn(f"Waypoint goal rejected by {robot}")
            self.current_goal_node[robot] = None
            return

        self.active_goal_handles[robot] = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, name=robot, h=goal_handle: self._on_goal_result(name, h)
        )

    def _on_goal_result(self, robot: str, original_handle) -> None:
        """Clear active state when a goal finishes, but only if the handle is still current."""
        if self.active_goal_handles.get(robot) is original_handle:
            self.active_goal_handles.pop(robot, None)
            self.current_goal_node[robot] = None

    # ------------------------------------------------------------------
    # Capture / blocked / stop
    # ------------------------------------------------------------------

    def _check_capture_or_blocked(self) -> Tuple[str, Optional[str]]:
        target_xy = self._xy_from_odom(self.latest_odom[self.target_name])

        for name in self.searchers:
            sx, sy = self._xy_from_odom(self.latest_odom[name])
            if math.dist((sx, sy), target_xy) <= self.capture_distance:
                return "capture", name

        target_node = self._nearest_node(target_xy)
        neighbors = self.graph.neighbors(target_node)
        if not neighbors:
            return "none", None

        occupied = set()
        for name in self.searchers:
            occupied.add(self._nearest_node(self._xy_from_odom(self.latest_odom[name])))

        if all(n in occupied for n in neighbors):
            self.get_logger().info("Target blocked in graph (all neighbor nodes occupied)")
            return "blocked", None

        return "none", None

    def _send_zero_velocity(self, robot: str) -> None:
        pub = self.cmd_vel_pubs.get(robot)
        if pub is None:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{robot}/base_link"
        pub.publish(msg)

    def _publish_stop(self) -> None:
        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

    def _publish_capture_pose(self, x: float, y: float) -> None:
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = "map"
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.orientation.w = 1.0
        self.capture_pose_pub.publish(p)

    def _handle_capture(self, capturer: Optional[str]) -> None:
        if self.stopped:
            return
        self.stopped = True
        target_xy = self._xy_from_odom(self.latest_odom[self.target_name])

        if capturer is not None:
            self._send_zero_velocity(capturer)
        self._publish_stop()
        self._publish_capture_pose(*target_xy)
        for robot, handle in self.active_goal_handles.items():
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception:  # noqa: BLE001
                    pass
            self.get_logger().info(f"Stopped {robot}")
        self.active_goal_handles.clear()

        if capturer is not None:
            for robot in self.searchers:
                if robot == capturer:
                    continue
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = f"{robot}/map"
                pose.pose.position.x = float(target_xy[0])
                pose.pose.position.y = float(target_xy[1])
                pose.pose.orientation.w = 1.0
                self._send_follow_waypoints(robot, [pose])
                self.get_logger().info(f"Regroup command sent to {robot}")

        self.get_logger().info("Capture event handled")

    def _handle_blocked(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        for name in self.cmd_vel_pubs:
            self._send_zero_velocity(name)
        self._publish_stop()
        for robot, handle in self.active_goal_handles.items():
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception:  # noqa: BLE001
                    pass
            self.get_logger().info(f"Stopped {robot}")
        self.get_logger().info("Blocked event handled")

    # ------------------------------------------------------------------
    # Geometry helpers (same as milp_graph_search)
    # ------------------------------------------------------------------

    @staticmethod
    def _xy_from_odom(msg: Optional[Odometry]) -> Tuple[float, float]:
        p = msg.pose.pose.position
        return float(p.x), float(p.y)

    def _nearest_node(self, xy: Tuple[float, float]) -> int:
        x, y = xy
        best_idx = 0
        best_d2 = float("inf")
        for idx, (nx, ny) in self.node_world_xy.items():
            d2 = (x - nx) ** 2 + (y - ny) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        return best_idx

    def _node_to_pose(self, robot: str, node_idx: int) -> PoseStamped:
        x, y = self.node_world_xy[node_idx]
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = f"{robot}/map"
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.position.z = 0.0
        p.pose.orientation.w = 1.0
        return p

    # ------------------------------------------------------------------
    # Graph & map loading (same as milp_graph_search)
    # ------------------------------------------------------------------

    def _load_graph_and_coordinates(self) -> None:
        if not self.graph_path or not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"graph_path not found: {self.graph_path}")

        if not self.map_yaml or not os.path.exists(self.map_yaml):
            raise FileNotFoundError(f"map_yaml not found: {self.map_yaml}")

        self.graph = ig.Graph.Read_GML(self.graph_path)
        if self.graph.is_directed():
            self.graph = self.graph.as_undirected(combine_edges="first")

        res, origin_x, origin_y, image_path = self._read_map_yaml(self.map_yaml)
        img_h = self._read_pgm_height(image_path)

        id_to_pixel_xy = self._read_gml_pixels(self.graph_path)
        self.node_world_xy = {}
        for v in self.graph.vs:
            vid = int(float(v["id"])) if "id" in self.graph.vertex_attributes() else v.index
            row, col = id_to_pixel_xy.get(vid, (0.0, 0.0))
            wx = origin_x + col * res
            wy = origin_y + (img_h - 1 - row) * res
            self.node_world_xy[v.index] = (wx, wy)

    @staticmethod
    def _read_map_yaml(path: str) -> Tuple[float, float, float, str]:
        image = None
        resolution = None
        origin = None

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("image:"):
                    image = line.split(":", 1)[1].strip()
                elif line.startswith("resolution:"):
                    resolution = float(line.split(":", 1)[1].strip())
                elif line.startswith("origin:"):
                    vals = line.split(":", 1)[1].strip().strip("[]")
                    origin = [float(v.strip()) for v in vals.split(",")]

        if image is None or resolution is None or origin is None:
            raise RuntimeError(f"Could not parse map yaml: {path}")

        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(path), image)
        return resolution, float(origin[0]), float(origin[1]), image

    @staticmethod
    def _read_pgm_height(path: str) -> int:
        with open(path, "rb") as f:
            magic = f.readline().strip()
            if magic not in (b"P5", b"P2"):
                raise RuntimeError(f"Unsupported PGM format: {magic!r}")

            def next_token() -> bytes:
                while True:
                    line = f.readline()
                    if not line:
                        return b""
                    line = line.strip()
                    if line.startswith(b"#") or len(line) == 0:
                        continue
                    return line

            dims = next_token().split()
            if len(dims) != 2:
                extra = next_token().split()
                dims += extra
            width = int(dims[0])
            height = int(dims[1])
            _ = width
            return height

    @staticmethod
    def _read_gml_pixels(path: str) -> Dict[int, Tuple[float, float]]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        blocks = re.findall(r"node\s*\[(.*?)\]", text, flags=re.DOTALL)
        out = {}
        for b in blocks:
            id_match = re.search(r"\bid\s+(-?\d+)", b)
            pixels = re.findall(r"\bpixel\s+(-?\d+(?:\.\d+)?)", b)
            if id_match and len(pixels) >= 2:
                vid = int(id_match.group(1))
                out[vid] = (float(pixels[0]), float(pixels[1]))
        return out


def main(args=None):
    rclpy.init(args=args)
    node = ExhaustiveGraphSearch()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
