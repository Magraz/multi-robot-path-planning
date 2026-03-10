import os
import re
from typing import Dict, Optional, Tuple

import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class RealtimeGraphVisualizer(Node):
    """Realtime map+graph+robot visualization with online target belief."""

    def __init__(self) -> None:
        super().__init__("realtime_graph_visualizer")

        self.declare_parameter("graph_path", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("searcher_names", ["robot_0", "robot_1"])
        self.declare_parameter("target_name", "target_0")
        self.declare_parameter("period", 0.5)
        self.declare_parameter("rotate_graph_deg", 0.0)
        self.declare_parameter("pixel_order", "rc")

        self.graph_path = str(self.get_parameter("graph_path").value)
        self.map_yaml = str(self.get_parameter("map_yaml").value)
        self.searchers = list(self.get_parameter("searcher_names").value)
        self.target_name = str(self.get_parameter("target_name").value)
        self.period = float(self.get_parameter("period").value)
        self.rotate_graph_deg = float(self.get_parameter("rotate_graph_deg").value)
        self.pixel_order = str(self.get_parameter("pixel_order").value)

        self.graph = None
        self.node_world_xy: Dict[int, Tuple[float, float]] = {}
        self.edges_xy = []
        self.map_extent = None
        self.map_img = None

        self.latest_odom: Dict[str, Optional[Odometry]] = {name: None for name in self.searchers}
        self.latest_odom[self.target_name] = None

        self.belief = None
        self.capture_belief = 0.0
        self.transition = None

        self._load_graph_and_map()
        self._init_belief()

        for name in self.latest_odom.keys():
            self.create_subscription(Odometry, f"/{name}/ground_truth", self._odom_cb(name), 10)

        self.fig, (self.ax_pos, self.ax_belief) = plt.subplots(1, 2, figsize=(14, 6))
        self._draw_static_backgrounds()
        plt.ion()
        plt.show(block=False)

        self.timer = self.create_timer(self.period, self._tick)
        self.get_logger().info("Realtime graph visualizer started")

    def _odom_cb(self, name: str):
        def cb(msg: Odometry):
            self.latest_odom[name] = msg

        return cb

    def _tick(self) -> None:
        self._update_belief()
        self._render()

    def _render(self) -> None:
        self.ax_pos.clear()
        self.ax_belief.clear()
        self._draw_static_backgrounds()

        xs = [self.node_world_xy[i][0] for i in range(self.graph.vcount())]
        ys = [self.node_world_xy[i][1] for i in range(self.graph.vcount())]

        # Position view
        self.ax_pos.scatter(xs, ys, s=20, c="#00bcd4", alpha=0.9, edgecolors="black", linewidths=0.3)
        for e0, e1 in self.edges_xy:
            self.ax_pos.plot([e0[0], e1[0]], [e0[1], e1[1]], color="gray", alpha=0.35, linewidth=0.7)

        # Belief view
        belief_plot = np.copy(self.belief)
        if np.max(belief_plot) > 0:
            belief_plot = belief_plot / np.max(belief_plot)
        self.ax_belief.scatter(xs, ys, s=70, c=belief_plot, cmap="hot", vmin=0.0, vmax=1.0, edgecolors="black", linewidths=0.2)
        for e0, e1 in self.edges_xy:
            self.ax_belief.plot([e0[0], e1[0]], [e0[1], e1[1]], color="gray", alpha=0.2, linewidth=0.6)

        # Robots and target markers
        colors = ["#1e88e5", "#43a047", "#8e24aa", "#fb8c00"]
        for i, name in enumerate(self.searchers):
            msg = self.latest_odom.get(name)
            if msg is None:
                continue
            x, y = self._xy_from_odom(msg)
            self.ax_pos.scatter([x], [y], marker="s", s=180, c=colors[i % len(colors)], edgecolors="white", linewidths=1.4)
            node_idx = self._nearest_node((x, y))
            nx, ny = self.node_world_xy[node_idx]
            self.ax_belief.scatter([nx], [ny], marker="s", s=200, c=colors[i % len(colors)], edgecolors="white", linewidths=1.4)

        tgt = self.latest_odom.get(self.target_name)
        if tgt is not None:
            tx, ty = self._xy_from_odom(tgt)
            self.ax_pos.scatter([tx], [ty], marker="s", s=220, c="red", edgecolors="white", linewidths=1.3)
            tnode = self._nearest_node((tx, ty))
            tnx, tny = self.node_world_xy[tnode]
            self.ax_belief.scatter([tnx], [tny], marker="s", s=240, c="red", edgecolors="white", linewidths=1.3)

        self.ax_pos.set_title("Realtime movement on map+graph")
        self.ax_belief.set_title(f"Belief over graph nodes | capture={self.capture_belief:.2f}")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_belief(self) -> None:
        # Prediction with uniform motion model.
        self.belief = self.belief @ self.transition

        # Update with searcher occupancy (captured nodes impossible).
        occupied = set()
        for name in self.searchers:
            msg = self.latest_odom.get(name)
            if msg is None:
                continue
            occupied.add(self._nearest_node(self._xy_from_odom(msg)))

        for idx in occupied:
            self.belief[idx] = 0.0

        s = float(np.sum(self.belief))
        if s > 1e-9:
            self.belief /= s
        self.capture_belief = max(0.0, 1.0 - float(np.sum(self.belief)))

    def _init_belief(self) -> None:
        n = self.graph.vcount()
        self.belief = np.ones(n, dtype=float) / float(n)

        self.transition = np.zeros((n, n), dtype=float)
        for i in range(n):
            nbrs = self.graph.neighbors(i)
            cand = [i] + nbrs
            p = 1.0 / float(len(cand))
            for j in cand:
                self.transition[i, j] = p

    def _load_graph_and_map(self) -> None:
        if not self.graph_path or not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"graph_path not found: {self.graph_path}")
        if not self.map_yaml or not os.path.exists(self.map_yaml):
            raise FileNotFoundError(f"map_yaml not found: {self.map_yaml}")

        self.graph = ig.Graph.Read_GML(self.graph_path)
        if self.graph.is_directed():
            self.graph = self.graph.as_undirected(combine_edges="first")

        res, origin_x, origin_y, image_path = self._read_map_yaml(self.map_yaml)
        img_h, img_w = self._read_pgm_shape(image_path)

        id_to_pixel_xy = self._read_gml_pixels(self.graph_path)
        for v in self.graph.vs:
            vid = int(float(v["id"])) if "id" in self.graph.vertex_attributes() else v.index
            p0, p1 = id_to_pixel_xy.get(vid, (0.0, 0.0))
            # draw_graph_on_map / overlay_graph_on_map default to pixel order row,col (rc)
            if self.pixel_order == "xy":
                col, row = p0, p1
            else:
                row, col = p0, p1
            wx = origin_x + col * res
            wy = origin_y + (img_h - 1 - row) * res
            self.node_world_xy[v.index] = (wx, wy)

        if abs(self.rotate_graph_deg) > 1e-9:
            self._rotate_nodes_about_map_center()

        for e in self.graph.es:
            s, t = e.tuple
            self.edges_xy.append((self.node_world_xy[s], self.node_world_xy[t]))

        self.map_img = mpimg.imread(image_path)
        x_max = origin_x + img_w * res
        y_max = origin_y + img_h * res
        self.map_extent = [origin_x, x_max, origin_y, y_max]

    def _rotate_nodes_about_map_center(self) -> None:
        if self.map_extent is None:
            # map_extent is set later in _load_graph_and_map, compute from current nodes if needed
            xs = [p[0] for p in self.node_world_xy.values()]
            ys = [p[1] for p in self.node_world_xy.values()]
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
        else:
            cx = 0.5 * (self.map_extent[0] + self.map_extent[1])
            cy = 0.5 * (self.map_extent[2] + self.map_extent[3])

        theta = np.deg2rad(self.rotate_graph_deg)
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        for idx, (x, y) in list(self.node_world_xy.items()):
            dx = x - cx
            dy = y - cy
            rx = c * dx - s * dy
            ry = s * dx + c * dy
            self.node_world_xy[idx] = (cx + rx, cy + ry)

    def _draw_static_backgrounds(self) -> None:
        for ax in (self.ax_pos, self.ax_belief):
            if self.map_img is not None and self.map_extent is not None:
                ax.imshow(self.map_img, cmap="gray", origin="lower", extent=self.map_extent, alpha=0.85)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2, linestyle="--")

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
    def _read_pgm_shape(path: str) -> Tuple[int, int]:
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
            width = int(dims[0])
            height = int(dims[1])
            return height, width

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
    node = RealtimeGraphVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
