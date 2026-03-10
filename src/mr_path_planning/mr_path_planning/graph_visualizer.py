"""Publish a GML graph as a MarkerArray for RViz visualization."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


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
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    nodes: Dict[int, dict] = {}
    edges: List[dict] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped in ("node [", "edge ["):
            kind = stripped.split()[0]
            attrs = []
            idx += 1
            while idx < len(lines) and lines[idx].strip() != "]":
                line = lines[idx].strip()
                if line:
                    key, value_raw = line.split(None, 1)
                    attrs.append((key, _parse_value(value_raw)))
                idx += 1
            by_key: Dict[str, list] = {}
            for key, value in attrs:
                by_key.setdefault(key, []).append(value)
            if kind == "node":
                node_id = int(by_key["id"][0])
                nodes[node_id] = {
                    "world": by_key.get("world", []),
                    "pixel": by_key.get("pixel", []),
                    "label": str(by_key.get("label", [str(node_id)])[0]),
                }
            else:
                edges.append(
                    {
                        "source": int(by_key["source"][0]),
                        "target": int(by_key["target"][0]),
                    }
                )
        idx += 1
    return nodes, edges


def _read_map_yaml(path: str) -> Tuple[float, float, float, str]:
    image = None
    resolution = None
    origin = None

    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("image:"):
                image = line.split(":", 1)[1].strip()
            elif line.startswith("resolution:"):
                resolution = float(line.split(":", 1)[1].strip())
            elif line.startswith("origin:"):
                values = line.split(":", 1)[1].strip().strip("[]")
                origin = [float(value.strip()) for value in values.split(",")]

    if image is None or resolution is None or origin is None:
        raise RuntimeError(f"Could not parse map yaml: {path}")

    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(path), image)
    return resolution, float(origin[0]), float(origin[1]), image


def _read_image_height(path: str) -> int:
    from PIL import Image

    with Image.open(path) as img:
        return int(img.height)


def _pixel_to_world(
    row: float,
    col: float,
    img_height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> Tuple[float, float]:
    world_x = origin_x + col * resolution
    world_y = origin_y + (img_height - 1 - row) * resolution
    return world_x, world_y


class GraphVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("graph_visualizer")

        self.declare_parameter("graph_path", "")
        self.declare_parameter("graph_file", "")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("frame_id", "robot_0/map")

        graph_path = str(self.get_parameter("graph_path").value)
        graph_file = str(self.get_parameter("graph_file").value)
        map_yaml = str(self.get_parameter("map_yaml").value)
        frame_id = str(self.get_parameter("frame_id").value)
        graph_source = graph_path or graph_file

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(MarkerArray, "graph_markers", qos)

        if not graph_source or not os.path.isfile(graph_source):
            self.get_logger().warn(f"Graph file not found: '{graph_source}'")
            return

        nodes, edges = _load_gml(graph_source)
        self.get_logger().info(
            f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges from {graph_source}"
        )

        positions: Dict[int, Tuple[float, float]] = {}
        needs_conversion = any(len(node["world"]) < 2 for node in nodes.values())

        if needs_conversion and map_yaml and os.path.isfile(map_yaml):
            resolution, origin_x, origin_y, image_path = _read_map_yaml(map_yaml)
            img_height = _read_image_height(image_path)
            for node_id, node in nodes.items():
                if len(node["world"]) >= 2:
                    positions[node_id] = (float(node["world"][0]), float(node["world"][1]))
                elif len(node["pixel"]) >= 2:
                    positions[node_id] = _pixel_to_world(
                        float(node["pixel"][0]),
                        float(node["pixel"][1]),
                        img_height,
                        resolution,
                        origin_x,
                        origin_y,
                    )
        else:
            for node_id, node in nodes.items():
                if len(node["world"]) >= 2:
                    positions[node_id] = (float(node["world"][0]), float(node["world"][1]))

        if not positions:
            self.get_logger().error(
                "No world coordinates found. Provide map_yaml for pixel-to-world conversion."
            )
            return

        marker_array = self._build_markers(nodes, edges, positions, frame_id)
        self.publisher.publish(marker_array)
        self.get_logger().info(
            f"Published {len(marker_array.markers)} markers on /graph_markers"
        )

    def _build_markers(self, nodes, edges, positions, frame_id: str) -> MarkerArray:
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

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

        for edge in edges:
            if edge["source"] in positions and edge["target"] in positions:
                source_x, source_y = positions[edge["source"]]
                target_x, target_y = positions[edge["target"]]
                edge_marker.points.append(Point(x=source_x, y=source_y, z=0.05))
                edge_marker.points.append(Point(x=target_x, y=target_y, z=0.05))

        marker_array.markers.append(edge_marker)

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

        for node_id in sorted(positions):
            point_x, point_y = positions[node_id]
            node_marker.points.append(Point(x=point_x, y=point_y, z=0.1))

        marker_array.markers.append(node_marker)

        for marker_id, node_id in enumerate(sorted(positions)):
            point_x, point_y = positions[node_id]
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = "graph_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = point_x
            text_marker.pose.position.y = point_y
            text_marker.pose.position.z = 0.35
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.2
            text_marker.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.9)
            text_marker.text = nodes[node_id]["label"]
            marker_array.markers.append(text_marker)

        return marker_array


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraphVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()