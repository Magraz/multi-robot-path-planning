import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import igraph as ig
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


class SearchMetricsLogger(Node):
    """Log capture time and graph exploration metrics for searcher robots."""

    def __init__(self) -> None:
        super().__init__("search_metrics_logger")

        self.declare_parameter("graph_path", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("searcher_names", ["robot_0", "robot_1"])
        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("metrics_csv", "")

        self.graph_path = str(self.get_parameter("graph_path").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.searchers = list(self.get_parameter("searcher_names").value)
        self.target_name = str(self.get_parameter("target_name").value)
        metrics_csv_param = str(self.get_parameter("metrics_csv").value)

        if metrics_csv_param:
            self.metrics_csv = Path(metrics_csv_param)
        else:
            self.metrics_csv = Path.home() / ".ros" / "search_metrics.csv"

        self.graph = None
        self.node_world_xy: Dict[int, Tuple[float, float]] = {}
        self._load_graph_and_coordinates()

        self.start_wall_time = time.monotonic()
        self.capture_wall_time: Optional[float] = None
        self.capture_received = False
        self.logged = False

        self.agent_nodes: Dict[str, Set[int]] = {name: set() for name in self.searchers}
        self.all_unique_nodes: Set[int] = set()

        for name in self.searchers:
            self.create_subscription(Odometry, f"/{name}/ground_truth", self._odom_cb(name), 10)

        self.create_subscription(Bool, "/search_capture/stop", self._stop_cb, 10)
        self.create_subscription(PoseStamped, "/search_capture/location", self._capture_cb, 10)

        self.get_logger().info(
            f"Search metrics logger ready | searchers={self.searchers} | target={self.target_name}"
        )
        self.get_logger().info(f"Metrics CSV: {self.metrics_csv}")

    def _odom_cb(self, name: str):
        def cb(msg: Odometry) -> None:
            xy = self._xy_from_odom(msg)
            node = self._nearest_node(xy)
            self.agent_nodes[name].add(node)
            self.all_unique_nodes.add(node)

        return cb

    def _capture_cb(self, _msg: PoseStamped) -> None:
        if self.capture_received:
            return
        self.capture_received = True
        self.capture_wall_time = time.monotonic()

    def _stop_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self.logged:
            return
        self._log_metrics()

    def _log_metrics(self) -> None:
        self.logged = True
        end_time = self.capture_wall_time if self.capture_wall_time is not None else time.monotonic()
        elapsed = end_time - self.start_wall_time

        outcome = "CAPTURED" if self.capture_received else "STOPPED (non-capture or blocked)"
        self.get_logger().info("===== SEARCH METRICS =====")
        self.get_logger().info(f"Outcome: {outcome}")
        self.get_logger().info(f"Time to capture/stop: {elapsed:.2f} s")

        for name in self.searchers:
            self.get_logger().info(
                f"Nodes explored by {name}: {len(self.agent_nodes[name])}"
            )

        self.get_logger().info(
            f"Unique nodes explored (all searchers): {len(self.all_unique_nodes)}"
        )
        self.get_logger().info("==========================")

        self._append_csv(
            elapsed_sec=elapsed,
            outcome=outcome,
            unique_nodes_all=len(self.all_unique_nodes),
        )

    def _append_csv(self, elapsed_sec: float, outcome: str, unique_nodes_all: int) -> None:
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "outcome": outcome,
            "elapsed_sec": f"{elapsed_sec:.3f}",
            "unique_nodes_all": str(unique_nodes_all),
            "graph": Path(self.graph_path).name,
        }

        fieldnames = ["timestamp", "outcome", "elapsed_sec", "unique_nodes_all", "graph"]
        for name in self.searchers:
            key = f"nodes_{name}"
            fieldnames.append(key)
            row[key] = str(len(self.agent_nodes[name]))

        write_header = (not self.metrics_csv.exists()) or self.metrics_csv.stat().st_size == 0
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.get_logger().info(f"Metrics appended to {self.metrics_csv}")

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
    node = SearchMetricsLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
