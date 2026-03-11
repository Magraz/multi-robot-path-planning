import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import igraph as ig
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


class MilpGraphSearch(Node):
    """Use MILP on a GML graph to generate high-level searcher waypoint routes."""

    def __init__(self) -> None:
        super().__init__("milp_graph_search")

        self.declare_parameter("enabled", True)
        self.declare_parameter("graph_path", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("horizon", 10)
        self.declare_parameter("replan_period", 12.0)
        self.declare_parameter("capture_distance", 0.6)
        self.declare_parameter("searcher_names", ["robot_0", "robot_1"])
        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("mespp_code_path", "")

        self.enabled = bool(self.get_parameter("enabled").value)
        self.graph_path = str(self.get_parameter("graph_path").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.horizon = int(self.get_parameter("horizon").value)
        self.replan_period = float(self.get_parameter("replan_period").value)
        self.capture_distance = float(self.get_parameter("capture_distance").value)
        self.searchers = list(self.get_parameter("searcher_names").value)
        self.target_name = str(self.get_parameter("target_name").value)
        self.mespp_code_path = str(self.get_parameter("mespp_code_path").value)

        self.latest_odom: Dict[str, Optional[Odometry]] = {name: None for name in self.searchers}
        self.latest_odom[self.target_name] = None

        self.graph: Optional[ig.Graph] = None
        self.node_world_xy: Dict[int, Tuple[float, float]] = {}

        self.follow_clients: Dict[str, ActionClient] = {
            name: ActionClient(self, FollowWaypoints, f"/{name}/follow_waypoints")
            for name in self.searchers
        }
        self.active_goal_handles = {}

        self.stop_pub = self.create_publisher(Bool, "/search_capture/stop", 10)
        self.capture_pose_pub = self.create_publisher(PoseStamped, "/search_capture/location", 10)

        self.cmd_vel_pubs: Dict[str, any] = {}
        for name in list(self.searchers) + [self.target_name]:
            self.cmd_vel_pubs[name] = self.create_publisher(
                TwistStamped, f"/{name}/cmd_vel", 10
            )

        self._load_graph_and_coordinates()
        self._import_mespp()

        for name in self.latest_odom.keys():
            self.create_subscription(Odometry, f"/{name}/ground_truth", self._odom_cb(name), 10)

        self.timer = self.create_timer(self.replan_period, self._tick)
        self.stopped = False
        self._solving = False

        self.get_logger().info(
            f"MILP graph search ready | graph={self.graph_path} | searchers={self.searchers} | target={self.target_name}"
        )

    def _import_mespp(self) -> None:
        self.Mespp = None
        if self.mespp_code_path:
            code_path = Path(self.mespp_code_path)
        else:
            code_path = Path(__file__).resolve().parents[2] / "search_and_capture_algo" / "code"
        if not code_path.exists():
            self.get_logger().error(f"search_and_capture_algo code path not found: {code_path}")
            return

        sys.path.insert(0, str(code_path))
        try:
            from main import Mespp  # type: ignore

            self.Mespp = Mespp
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to import Mespp from search_and_capture_algo: {exc}")

    def _odom_cb(self, name: str):
        def cb(msg: Odometry):
            self.latest_odom[name] = msg
            if not self.stopped and self.graph is not None:
                self._fast_capture_check()

        return cb

    def _fast_capture_check(self) -> None:
        """Check capture at odom frequency so we never miss a close pass."""
        if any(self.latest_odom[n] is None for n in self.latest_odom):
            return

        target_xy = self._xy_from_odom(self.latest_odom[self.target_name])
        for name in self.searchers:
            sx, sy = self._xy_from_odom(self.latest_odom[name])
            if math.dist((sx, sy), target_xy) <= self.capture_distance:
                self.get_logger().info(f"Capture condition met by {name}")
                self._handle_capture(name)
                return

    def _tick(self) -> None:
        if not self.enabled or self.stopped:
            return

        if self.Mespp is None or self.graph is None:
            return

        if any(self.latest_odom[name] is None for name in self.latest_odom):
            self.get_logger().info("Waiting for all odometry topics...")
            return

        event, _ = self._check_capture_or_blocked()
        if event == "blocked":
            self._handle_blocked()
            return

        if self._solving:
            self.get_logger().info("MILP solve still in progress, skipping this tick")
            return

        # Snapshot current positions for the solver thread.
        target_xy = self._xy_from_odom(self.latest_odom[self.target_name])
        target_start = self._nearest_node(target_xy)
        searcher_starts = []
        for name in self.searchers:
            xy = self._xy_from_odom(self.latest_odom[name])
            searcher_starts.append(self._nearest_node(xy))

        self._solving = True
        thread = threading.Thread(
            target=self._solve_and_dispatch,
            args=(target_start, searcher_starts),
            daemon=True,
        )
        thread.start()

    def _solve_and_dispatch(self, target_start: int, searcher_starts: List[int]) -> None:
        """Run the MILP solver in a background thread, then dispatch routes."""
        try:
            solver = self.Mespp(
                graph_path=self.graph_path,
                horizon=self.horizon,
                num_searchers=len(self.searchers),
                target_start_vertex=target_start,
                searcher_starts=searcher_starts,
                motion="uniform",
            )
            solver.build_milp_variables()
            solver.m.update()
            solver.build_milp_constraints()
            solver.configure_objective()
            solver.plan()

            if self.stopped:
                return

            if solver.m.status != 2:  # GRB.OPTIMAL
                self.get_logger().warn("MILP did not return optimal solution; skipping dispatch")
                return

            routes = self._extract_routes(solver)
            for idx, robot in enumerate(self.searchers):
                node_route = routes[idx]
                waypoint_nodes = [n for i, n in enumerate(node_route) if i == 0 or n != node_route[i - 1]]
                if len(waypoint_nodes) <= 1:
                    continue
                poses = [self._node_to_pose(robot, n) for n in waypoint_nodes[1:]]
                self._send_follow_waypoints(robot, poses)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"MILP planning/dispatch failed: {exc}")
        finally:
            self._solving = False

    def _extract_routes(self, solver) -> List[List[int]]:
        routes = [[] for _ in self.searchers]
        for s in range(len(self.searchers)):
            last_v = 0
            for t in range(self.horizon + 1):
                chosen = None
                for v, var in solver.presence[t][s].items():
                    val = var.X if not hasattr(var.X, "item") else var.X.item()
                    if val > 0.5:
                        chosen = v
                        break
                if chosen is None:
                    chosen = last_v
                routes[s].append(int(chosen))
                last_v = int(chosen)
        return routes

    def _send_follow_waypoints(self, robot: str, poses: List[PoseStamped]) -> None:
        client = self.follow_clients[robot]
        if not client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn(f"/{robot}/follow_waypoints action server unavailable")
            return

        prev_handle = self.active_goal_handles.pop(robot, None)
        if prev_handle is not None:
            try:
                cancel_future = prev_handle.cancel_goal_async()
                cancel_future.add_done_callback(
                    lambda f, r=robot, p=poses: self._on_cancel_done(r, p)
                )
                return  # New goal sent after cancel completes.
            except Exception:  # noqa: BLE001
                pass

        self._send_new_goal(robot, poses)

    def _on_cancel_done(self, robot: str, poses: List[PoseStamped]) -> None:
        if self.stopped:
            return
        self._send_new_goal(robot, poses)

    def _send_new_goal(self, robot: str, poses: List[PoseStamped]) -> None:
        client = self.follow_clients[robot]
        goal = FollowWaypoints.Goal()
        goal.poses = poses

        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f, name=robot: self._on_goal_response(name, f))
        self.get_logger().info(f"Sent {len(poses)} MILP waypoints to {robot}")

    def _on_goal_response(self, robot: str, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to send waypoint goal to {robot}: {exc}")
            return

        if not goal_handle.accepted:
            self.get_logger().warn(f"Waypoint goal rejected by {robot}")
            return

        self.active_goal_handles[robot] = goal_handle

    def _check_capture_or_blocked(self) -> Tuple[str, Optional[str]]:
        target_xy = self._xy_from_odom(self.latest_odom[self.target_name])

        # Capture condition: any searcher physically close to target.
        for name in self.searchers:
            sx, sy = self._xy_from_odom(self.latest_odom[name])
            if math.dist((sx, sy), target_xy) <= self.capture_distance:
                self.get_logger().info(f"Capture condition met by {name}")
                return "capture", name

        # Block condition: all target graph neighbors are occupied by searchers.
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
        """Send zero cmd_vel to a specific robot for an immediate physical stop."""
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

        # Immediately halt the capturer, then cancel Nav2 goals.
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

        # Practical regroup: non-capturing searcher moves to capture location.
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

    def _stop_searchers(self, reason: str) -> None:
        self.stopped = True
        for robot, handle in self.active_goal_handles.items():
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception:  # noqa: BLE001
                    pass
            self.get_logger().info(f"Stopped {robot}")
        self.get_logger().info(reason)

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
            # draw_graph_on_map stores pixel values as row,col by default.
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
    node = MilpGraphSearch()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
