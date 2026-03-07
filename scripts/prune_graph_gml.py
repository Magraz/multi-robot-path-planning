#!/usr/bin/env python3
"""Prune near-duplicate nodes and short edges from a simple GML graph."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class Node:
    node_id: int
    attrs: List[Tuple[str, object]]
    x: float
    y: float
    node_type: str


@dataclass
class Edge:
    source: int
    target: int
    attrs: List[Tuple[str, object]]
    edge_type: str


class UnionFind:
    def __init__(self, items: Sequence[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_value(raw: str) -> object:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def format_value(value: object) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, float):
        return repr(value)
    return str(value)


def parse_gml(path: Path) -> Tuple[List[Node], List[Edge]]:
    lines = path.read_text().splitlines()
    nodes: List[Node] = []
    edges: List[Edge] = []
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
                    attrs.append((key, parse_value(value_raw)))
                i += 1
            if kind == "node":
                node_id = int(first_value(attrs, "id"))
                world_vals = values(attrs, "world")
                if len(world_vals) < 2:
                    raise ValueError(f"Node {node_id} has no 2D world coordinates")
                x = float(world_vals[0])
                y = float(world_vals[1])
                node_type = str(first_value(attrs, "node_type", ""))
                nodes.append(Node(node_id=node_id, attrs=attrs, x=x, y=y, node_type=node_type))
            else:
                source = int(first_value(attrs, "source"))
                target = int(first_value(attrs, "target"))
                edge_type = str(first_value(attrs, "edge_type", ""))
                edges.append(Edge(source=source, target=target, attrs=attrs, edge_type=edge_type))
        i += 1
    return nodes, edges


def values(attrs: List[Tuple[str, object]], key: str) -> List[object]:
    return [value for attr_key, value in attrs if attr_key == key]


def first_value(attrs: List[Tuple[str, object]], key: str, default: object | None = None) -> object:
    for attr_key, value in attrs:
        if attr_key == key:
            return value
    if default is not None:
        return default
    raise KeyError(key)


def node_distance(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def choose_representatives(
    nodes: List[Node], edges: List[Edge], node_min_dist: float
) -> Tuple[Dict[int, int], Dict[int, Node]]:
    node_ids = [n.node_id for n in nodes]
    uf = UnionFind(node_ids)
    for idx, node in enumerate(nodes):
        for other in nodes[idx + 1 :]:
            if node_distance(node, other) < node_min_dist:
                uf.union(node.node_id, other.node_id)

    degree = {n.node_id: 0 for n in nodes}
    for edge in edges:
        if edge.source in degree:
            degree[edge.source] += 1
        if edge.target in degree:
            degree[edge.target] += 1

    node_by_id = {n.node_id: n for n in nodes}
    clusters: Dict[int, List[int]] = {}
    for node_id in node_ids:
        clusters.setdefault(uf.find(node_id), []).append(node_id)

    mapping: Dict[int, int] = {}
    keep_nodes: Dict[int, Node] = {}
    for members in clusters.values():
        members_sorted = sorted(
            members,
            key=lambda nid: (
                0 if node_by_id[nid].node_type != "boundary" else 1,
                -degree[nid],
                nid,
            ),
        )
        rep = members_sorted[0]
        for nid in members:
            mapping[nid] = rep
        keep_nodes[rep] = node_by_id[rep]
    return mapping, keep_nodes


def prune_edges(
    edges: List[Edge],
    node_mapping: Dict[int, int],
    keep_nodes: Dict[int, Node],
    edge_min_dist: float,
) -> List[Edge]:
    edge_type_priority = {"skeleton": 4, "contour": 3, "delaunay": 2, "merge": 1}

    def pair_dist(a: int, b: int) -> float:
        na = keep_nodes[a]
        nb = keep_nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)

    best_for_pair: Dict[Tuple[int, int], Tuple[int, str]] = {}

    for edge in edges:
        if edge.source not in node_mapping or edge.target not in node_mapping:
            continue
        source = node_mapping[edge.source]
        target = node_mapping[edge.target]
        if source == target:
            continue
        a, b = sorted((source, target))
        dist = pair_dist(a, b)
        if dist < edge_min_dist:
            continue

        edge_type = edge.edge_type
        pri = edge_type_priority.get(edge_type, 0)
        prev = best_for_pair.get((a, b))
        if prev is None or pri > prev[0]:
            best_for_pair[(a, b)] = (pri, edge_type)

    pruned_edges: List[Edge] = []
    for source, target in sorted(best_for_pair.keys()):
        dist = pair_dist(source, target)
        edge_type = best_for_pair[(source, target)][1]
        attrs: List[Tuple[str, object]] = [
            ("source", source),
            ("target", target),
            ("weight", dist),
        ]
        if edge_type:
            attrs.append(("edge_type", edge_type))
        pruned_edges.append(Edge(source=source, target=target, attrs=attrs, edge_type=edge_type))
    return pruned_edges


def write_gml(path: Path, nodes: List[Node], edges: List[Edge]) -> None:
    out: List[str] = ["graph ["]
    for node in sorted(nodes, key=lambda n: n.node_id):
        out.append("  node [")
        for key, value in node.attrs:
            out.append(f"    {key} {format_value(value)}")
        out.append("  ]")
    for edge in edges:
        out.append("  edge [")
        for key, value in edge.attrs:
            out.append(f"    {key} {format_value(value)}")
        out.append("  ]")
    out.append("]")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune nodes and edges that are too close in a GML graph."
    )
    parser.add_argument("input", type=Path, help="Input graph.gml path")
    parser.add_argument("output", type=Path, help="Output pruned graph.gml path")
    parser.add_argument(
        "--node-min-dist",
        type=float,
        default=0.3,
        help="Merge nodes with world distance below this threshold (default: 0.3)",
    )
    parser.add_argument(
        "--edge-min-dist",
        type=float,
        default=0.3,
        help="Drop edges whose endpoint world distance is below this threshold (default: 0.3)",
    )
    args = parser.parse_args()

    nodes, edges = parse_gml(args.input)
    node_mapping, keep_nodes = choose_representatives(nodes, edges, args.node_min_dist)
    pruned_edges = prune_edges(edges, node_mapping, keep_nodes, args.edge_min_dist)
    write_gml(args.output, list(keep_nodes.values()), pruned_edges)

    print(
        f"Pruned graph written to {args.output} | "
        f"nodes: {len(nodes)} -> {len(keep_nodes)} | "
        f"edges: {len(edges)} -> {len(pruned_edges)} | "
        f"node_min_dist={args.node_min_dist} edge_min_dist={args.edge_min_dist}"
    )


if __name__ == "__main__":
    main()
