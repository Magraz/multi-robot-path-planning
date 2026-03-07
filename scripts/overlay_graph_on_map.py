#!/usr/bin/env python3
"""Overlay a waypoint graph (GML) on top of a map image."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Node:
    node_id: int
    label: str
    world: Tuple[float, float] | None
    pixel: Tuple[float, float] | None
    node_type: str


@dataclass
class Edge:
    source: int
    target: int
    edge_type: str


EDGE_COLORS = {
    "skeleton": (230, 57, 70, 220),
    "contour": (244, 162, 97, 220),
    "delaunay": (42, 157, 143, 220),
    "merge": (69, 123, 157, 220),
}

NODE_COLORS = {
    "boundary": (245, 158, 11, 240),
    "default": (16, 185, 129, 240),
}


def parse_value(raw: str) -> str | float | int:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_gml(path: Path) -> Tuple[Dict[int, Node], List[Edge]]:
    lines = path.read_text().splitlines()
    nodes: Dict[int, Node] = {}
    edges: List[Edge] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s in ("node [", "edge ["):
            kind = s.split()[0]
            attrs: List[Tuple[str, str | float | int]] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "]":
                t = lines[i].strip()
                if t:
                    key, value_raw = t.split(None, 1)
                    attrs.append((key, parse_value(value_raw)))
                i += 1

            if kind == "node":
                by_key: Dict[str, List[str | float | int]] = {}
                for k, v in attrs:
                    by_key.setdefault(k, []).append(v)
                node_id = int(by_key["id"][0])
                label = str(by_key.get("label", [str(node_id)])[0])
                world_vals = by_key.get("world", [])
                pixel_vals = by_key.get("pixel", [])
                world = (
                    (float(world_vals[0]), float(world_vals[1]))
                    if len(world_vals) >= 2
                    else None
                )
                pixel = (
                    (float(pixel_vals[0]), float(pixel_vals[1]))
                    if len(pixel_vals) >= 2
                    else None
                )
                node_type = str(by_key.get("node_type", [""])[0])
                nodes[node_id] = Node(
                    node_id=node_id,
                    label=label,
                    world=world,
                    pixel=pixel,
                    node_type=node_type,
                )
            else:
                by_key = {k: v for k, v in attrs}
                edges.append(
                    Edge(
                        source=int(by_key["source"]),
                        target=int(by_key["target"]),
                        edge_type=str(by_key.get("edge_type", "")),
                    )
                )
        i += 1
    return nodes, edges


def parse_map_yaml(path: Path) -> Tuple[float, float, float, Path]:
    resolution = None
    origin = None
    image_rel = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("resolution:"):
            resolution = float(line.split(":", 1)[1].strip())
        elif line.startswith("origin:"):
            origin = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("image:"):
            image_rel = line.split(":", 1)[1].strip()

    if resolution is None or origin is None:
        raise ValueError(f"Could not parse resolution/origin from {path}")
    if len(origin) < 2:
        raise ValueError(f"Invalid origin in {path}")

    image_path = (path.parent / image_rel).resolve() if image_rel else Path()
    return resolution, float(origin[0]), float(origin[1]), image_path


def world_to_pixel(
    x_world: float,
    y_world: float,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> Tuple[float, float]:
    x = (x_world - origin_x) / resolution
    y = (height - 1) - ((y_world - origin_y) / resolution)
    return x, y


def node_to_xy(
    node: Node,
    coord_source: str,
    pixel_order: str,
    width: int,
    height: int,
    resolution: float | None,
    origin_x: float | None,
    origin_y: float | None,
) -> Tuple[float, float]:
    if coord_source == "world":
        if node.world is None:
            raise ValueError(f"Node {node.node_id} has no world coords")
        if resolution is None or origin_x is None or origin_y is None:
            raise ValueError("World coordinate mode requires map yaml")
        return world_to_pixel(
            node.world[0], node.world[1], width, height, resolution, origin_x, origin_y
        )

    if node.pixel is None:
        raise ValueError(f"Node {node.node_id} has no pixel coords")
    p0, p1 = node.pixel
    if pixel_order == "xy":
        return p0, p1
    return p1, p0


def draw_overlay(
    map_image: Path,
    nodes: Dict[int, Node],
    edges: List[Edge],
    output: Path,
    coord_source: str,
    pixel_order: str,
    node_radius: int,
    edge_width: int,
    draw_labels: bool,
    resolution: float | None,
    origin_x: float | None,
    origin_y: float | None,
) -> None:
    img = Image.open(map_image).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    coords: Dict[int, Tuple[float, float]] = {}
    for node_id, node in nodes.items():
        coords[node_id] = node_to_xy(
            node,
            coord_source=coord_source,
            pixel_order=pixel_order,
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
        )

    for edge in edges:
        if edge.source not in coords or edge.target not in coords:
            continue
        color = EDGE_COLORS.get(edge.edge_type, (99, 102, 241, 190))
        draw.line([coords[edge.source], coords[edge.target]], fill=color, width=edge_width)

    for node in nodes.values():
        x, y = coords[node.node_id]
        color = NODE_COLORS["boundary"] if node.node_type == "boundary" else NODE_COLORS["default"]
        draw.ellipse(
            [x - node_radius, y - node_radius, x + node_radius, y + node_radius],
            fill=color,
            outline=(0, 0, 0, 240),
            width=1,
        )
        if draw_labels:
            draw.text((x + node_radius + 1, y - node_radius - 1), node.label, fill=(0, 0, 0, 255), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay a GML waypoint graph on a map image and export PNG."
    )
    parser.add_argument("--graph", type=Path, required=True, help="Path to graph.gml")
    parser.add_argument(
        "--map-image",
        type=Path,
        default=None,
        help="Path to map image (.png/.pgm/.jpg)",
    )
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=None,
        help="Optional map yaml; required for --coord-source world",
    )
    parser.add_argument(
        "--coord-source",
        choices=["auto", "world", "pixel"],
        default="auto",
        help="Use node world or pixel fields (default: auto)",
    )
    parser.add_argument(
        "--pixel-order",
        choices=["rc", "xy"],
        default="rc",
        help="How to interpret pixel fields from GML in pixel mode (default: rc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("waypoint_graph.png"),
        help="Output overlay PNG (default: waypoint_graph.png)",
    )
    parser.add_argument("--node-radius", type=int, default=3, help="Node circle radius")
    parser.add_argument("--edge-width", type=int, default=2, help="Edge line width")
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw node labels",
    )
    args = parser.parse_args()

    nodes, edges = parse_gml(args.graph)

    resolution = None
    origin_x = None
    origin_y = None

    coord_source = args.coord_source
    if coord_source == "auto":
        coord_source = "world" if args.map_yaml else "pixel"

    if coord_source == "world":
        if args.map_yaml is None:
            raise ValueError("--map-yaml is required when --coord-source world")
        resolution, origin_x, origin_y, yaml_image = parse_map_yaml(args.map_yaml)
        if args.map_image is None and yaml_image.exists():
            args.map_image = yaml_image

    if args.map_image is None:
        raise ValueError("--map-image is required unless --map-yaml provides image path")
    if not args.map_image.exists():
        raise FileNotFoundError(f"Map image not found: {args.map_image}")

    draw_overlay(
        map_image=args.map_image,
        nodes=nodes,
        edges=edges,
        output=args.output,
        coord_source=coord_source,
        pixel_order=args.pixel_order,
        node_radius=args.node_radius,
        edge_width=args.edge_width,
        draw_labels=args.draw_labels,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    print(
        f"Saved overlay to {args.output} | "
        f"nodes={len(nodes)} edges={len(edges)} coord_source={coord_source}"
    )


if __name__ == "__main__":
    main()
