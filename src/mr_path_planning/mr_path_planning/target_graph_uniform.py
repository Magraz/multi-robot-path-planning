import os
import random
import re
from typing import Dict, Optional, Tuple

import igraph as ig
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


class TargetGraphUniform(Node):
    """Move target robot with uniform random-walk over graph nodes using Nav2."""

    def __init__(self) -> None:
        super().__init__("target_graph_uniform")

        self.declare_parameter("graph_path", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("period", 6.0)
        self.declare_parameter("move_every_n_cycles", 2)
        self.declare_parameter("seed", 0)

        self.graph_path = str(self.get_parameter("graph_path").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.target_name = str(self.get_parameter("target_name").value)
        self.period = float(self.get_parameter("period").value)
        self.move_every_n_cycles = int(self.get_parameter("move_every_n_cycles").value)
        seed = int(self.get_parameter("seed").value)
        random.seed(seed)

        self.graph = None
        self.node_world_xy: Dict[int, Tuple[float, float]] = {}
        self._load_graph_and_coordinates()

        self.latest_odom: Optional[Odometry] = None
        self.create_subscription(Odometry, f"/{self.target_name}/ground_truth", self._odom_cb, 10)
        self.create_subscription(Bool, "/search_capture/stop", self._stop_cb, 10)

        self.client = ActionClient(self, NavigateToPose, f"/{self.target_name}/navigate_to_pose")
        self.goal_handle = None
        self.busy = False
        self.stopped = False
        self.tick_count = 0

        self.timer = self.create_timer(self.period, self._tick)
        self.get_logger().info(
            f"Uniform graph target motion ready | target={self.target_name} | graph={self.graph_path} | "
            f"nodes={len(self.graph.vs)} | edges={len(self.graph.es)}"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _tick(self) -> None:
        if self.stopped or self.latest_odom is None or self.busy:
            return
        self.tick_count += 1
        if self.move_every_n_cycles > 1 and (self.tick_count % self.move_every_n_cycles) != 0:
            return
        if not self.client.wait_for_server(timeout_sec=0.1):
            return

        current_xy = self._xy_from_odom(self.latest_odom)
        current_node = self._nearest_node(current_xy)
        neighbors = self.graph.neighbors(current_node)
        candidates = [current_node] + neighbors
        
        if not candidates:
            self.get_logger().warn(f"No candidates for movement from node {current_node}")
            return
            
        next_node = random.choice(candidates)

        pose = self._node_to_pose(next_node)
        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.busy = True
        fut = self.client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception:  # noqa: BLE001
            self.busy = False
            return

        if not goal_handle.accepted:
            self.busy = False
            return

        self.goal_handle = goal_handle
        result_fut = goal_handle.get_result_async()
        result_fut.add_done_callback(self._on_result)

    def _on_result(self, _future) -> None:
        self.busy = False

    def _stop_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.stopped = True
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
        self.get_logger().info("Received stop event; target motion halted")

    @staticmethod
    def _xy_from_odom(msg: Odometry) -> Tuple[float, float]:
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

    def _node_to_pose(self, node_idx: int) -> PoseStamped:
        x, y = self.node_world_xy[node_idx]
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = f"{self.target_name}/map"
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
    def _read_map_yaml(path: str):
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
                dims += next_token().split()
            height = int(dims[1])
            return height

    @staticmethod
    def _read_gml_pixels(path: str):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        blocks = re.findall(r"node\s*\[(.*?)\]", text, flags=re.DOTALL)
        out = {}
        for b in blocks:
            id_match = re.search(r"\bid\s+(-?\d+)", b)
            pixels = re.findall(r"\bpixel\s+(-?\d+(?:\.\d+)?)", b)
            if id_match and len(pixels) >= 2:
                out[int(id_match.group(1))] = (float(pixels[0]), float(pixels[1]))
        return out


def main(args=None):
    rclpy.init(args=args)
    node = TargetGraphUniform()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
