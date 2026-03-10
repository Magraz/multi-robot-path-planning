"""Publish a GML graph as a MarkerArray for RViz visualization."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _load_gml(path: str):
    """Parse a GML file, return (nodes_dict, edges_list).

    nodes_dict: {id: {"world": [x,y,z], "pixel": [row,col], "label": str}}
    edges_list: [{"source": int, "target": int}]
    """
    lines = Path(path).read_text().splitlines()
    nodes: Dict[int, dict] = {}
    edges: List[dict] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s in ("node [", "edge ["):
            kind = s.split()[0]
            attrs: List[Tuple[str, object]] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "]":
                t = lines[i].strip()
                if t:
                    key, value_raw = t.split(None, 1)
                    attrs.append((key, _parse_value(value_raw)))
                i += 1
            by_key: Dict[str, list] = {}
            for k, v in attrs:
                by_key.setdefault(k, []).append(v)
            if kind == "node":
                node_id = int(by_key["id"][0])
                nodes[node_id] = {
                    "world": by_key.get("world", []),
                    "pixel": by_key.get("pixel", []),
                    "label": str(by_key.get("label", [str(node_id)])[0]),
                }
            else:
                edges.append({
                    "source": int(by_key["source"][0]),
                    "target": int(by_key["target"][0]),
                })
        i += 1
    return nodes, edges


def _parse_map_yaml(path: str) -> Tuple[float, float, float, Optional[int]]:
    """Return (resolution, origin_x, origin_y, image_height_px)."""
    import yaml
    from PIL import Image

    data = yaml.safe_load(Path(path).read_text())
    resolution = float(data["resolution"])
    origin = data["origin"]
    origin_x, origin_y = float(origin[0]), float(origin[1])

    img_path = Path(path).parent / data["image"]
    img = Image.open(str(img_path))
    img_height = img.height
    img.close()

    return resolution, origin_x, origin_y, img_height


def _pixel_to_world(row: float, col: float, img_height: int,
                    resolution: float, origin_x: float, origin_y: float):
    x = origin_x + col * resolution
    y = origin_y + (img_height - 1 - row) * resolution
    return x, y


class GraphVisualizer(Node):
    def __init__(self):
        super().__init__("graph_visualizer")

        self.declare_parameter("graph_file", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("frame_id", "robot_0/map")

        graph_file = self.get_parameter("graph_file").get_parameter_value().string_value
        map_yaml = self.get_parameter("map_yaml").get_parameter_value().string_value
        frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub = self.create_publisher(MarkerArray, "graph_markers", qos)

        if not graph_file or not os.path.isfile(graph_file):
            self.get_logger().warn(f"Graph file not found: '{graph_file}'")
            return

        nodes, edges = _load_gml(graph_file)
        self.get_logger().info(
            f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges from {graph_file}"
        )

        # Resolve world coordinates for each node
        positions: Dict[int, Tuple[float, float]] = {}
        needs_conversion = any(len(n["world"]) < 2 for n in nodes.values())

        if needs_conversion and map_yaml and os.path.isfile(map_yaml):
            res, ox, oy, img_h = _parse_map_yaml(map_yaml)
            for nid, n in nodes.items():
                if len(n["world"]) >= 2:
                    positions[nid] = (float(n["world"][0]), float(n["world"][1]))
                elif len(n["pixel"]) >= 2:
                    positions[nid] = _pixel_to_world(
                        float(n["pixel"][0]), float(n["pixel"][1]),
                        img_h, res, ox, oy,
                    )
        else:
            for nid, n in nodes.items():
                if len(n["world"]) >= 2:
                    positions[nid] = (float(n["world"][0]), float(n["world"][1]))

        if not positions:
            self.get_logger().error(
                "No world coordinates found. Provide map_yaml for pixel→world conversion."
            )
            return

        ma = self._build_markers(nodes, edges, positions, frame_id)
        self.pub.publish(ma)
        self.get_logger().info(f"Published {len(ma.markers)} markers on /graph_markers")

    def _build_markers(self, nodes, edges, positions, frame_id) -> MarkerArray:
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # --- Edge lines (single LINE_LIST marker) ---
        edge_marker = Marker()
        edge_marker.header.frame_id = frame_id
        edge_marker.header.stamp = stamp
        edge_marker.ns = "graph_edges"
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.08
        edge_marker.color = ColorRGBA(r=0.94, g=0.27, b=0.27, a=0.85)
        edge_marker.pose.orientation.w = 1.0

        for e in edges:
            if e["source"] in positions and e["target"] in positions:
                sx, sy = positions[e["source"]]
                tx, ty = positions[e["target"]]
                p1 = Point(x=sx, y=sy, z=0.05)
                p2 = Point(x=tx, y=ty, z=0.05)
                edge_marker.points.append(p1)
                edge_marker.points.append(p2)

        ma.markers.append(edge_marker)

        # --- Node spheres (single SPHERE_LIST marker) ---
        node_marker = Marker()
        node_marker.header.frame_id = frame_id
        node_marker.header.stamp = stamp
        node_marker.ns = "graph_nodes"
        node_marker.id = 0
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.25
        node_marker.scale.y = 0.25
        node_marker.scale.z = 0.25
        node_marker.color = ColorRGBA(r=0.23, g=0.51, b=0.96, a=0.9)
        node_marker.pose.orientation.w = 1.0

        for nid in sorted(positions):
            x, y = positions[nid]
            node_marker.points.append(Point(x=x, y=y, z=0.1))

        ma.markers.append(node_marker)

        # --- Node labels (individual TEXT markers) ---
        for idx, nid in enumerate(sorted(positions)):
            x, y = positions[nid]
            txt = Marker()
            txt.header.frame_id = frame_id
            txt.header.stamp = stamp
            txt.ns = "graph_labels"
            txt.id = idx
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = x
            txt.pose.position.y = y
            txt.pose.position.z = 0.35
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.2
            txt.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.9)
            txt.text = nodes[nid]["label"]
            ma.markers.append(txt)

        return ma


def main(args=None):
    rclpy.init(args=args)
    node = GraphVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
