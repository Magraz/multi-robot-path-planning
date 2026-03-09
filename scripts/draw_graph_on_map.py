#!/usr/bin/env python3
"""Interactive web-based tool: draw a waypoint graph on a map image and export to GML.

Runs a local HTTP server — works over SSH without X11 forwarding.

Usage:
    python scripts/draw_graph_on_map.py --map-image path/to/map.png --output graph.gml

    Load an existing GML to continue editing:
    python scripts/draw_graph_on_map.py --map-image path/to/map.png --load existing.gml --output graph.gml

    With world coordinate conversion:
    python scripts/draw_graph_on_map.py --map-image path/to/map.png --map-yaml map.yaml --output graph.gml

    Custom port:
    python scripts/draw_graph_on_map.py --map-image path/to/map.png --port 9000 --output graph.gml

Controls (in browser):
    Left-click          Place a new node (in Node mode) / select nodes for edge (in Edge mode)
    Right-click node    Delete node and its edges
    n                   Switch to Node mode
    e                   Switch to Edge mode
    Ctrl+z              Undo last action
    Ctrl+s              Save graph to GML
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import sys
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GraphNode:
    __slots__ = ("node_id", "pixel_row", "pixel_col", "world_x", "world_y",
                 "world_z", "label", "node_type")

    def __init__(self, node_id: int, pixel_row: float, pixel_col: float,
                 world_x: Optional[float] = None, world_y: Optional[float] = None,
                 world_z: float = 0.0, label: Optional[str] = None,
                 node_type: str = ""):
        self.node_id = node_id
        self.pixel_row = pixel_row
        self.pixel_col = pixel_col
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.label = label
        self.node_type = node_type

    def to_dict(self) -> dict:
        return {"id": self.node_id, "row": self.pixel_row, "col": self.pixel_col,
                "label": self.label or str(self.node_id)}


class GraphEdge:
    __slots__ = ("source", "target", "edge_type")

    def __init__(self, source: int, target: int, edge_type: str = "manual"):
        self.source = source
        self.target = target
        self.edge_type = edge_type

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target}


class GraphState:
    def __init__(self) -> None:
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.next_id: int = 0

    def add_node(self, row: float, col: float) -> GraphNode:
        node = GraphNode(node_id=self.next_id, pixel_row=row, pixel_col=col,
                         label=str(self.next_id))
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node

    def add_edge(self, src: int, tgt: int) -> Optional[GraphEdge]:
        if src == tgt:
            return None
        for e in self.edges:
            if {e.source, e.target} == {src, tgt}:
                return None
        edge = GraphEdge(source=src, target=tgt)
        self.edges.append(edge)
        return edge

    def remove_node(self, node_id: int) -> Tuple[Optional[GraphNode], List[GraphEdge]]:
        node = self.nodes.pop(node_id, None)
        removed = [e for e in self.edges if e.source == node_id or e.target == node_id]
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
        return node, removed

    def remove_edge(self, src: int, tgt: int) -> Optional[GraphEdge]:
        pair = {src, tgt}
        for i, e in enumerate(self.edges):
            if {e.source, e.target} == pair:
                return self.edges.pop(i)
        return None

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in sorted(self.nodes.values(), key=lambda n: n.node_id)],
            "edges": [e.to_dict() for e in self.edges],
            "next_id": self.next_id,
        }


# ---------------------------------------------------------------------------
# GML I/O
# ---------------------------------------------------------------------------

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


def load_gml(path: Path) -> GraphState:
    lines = path.read_text().splitlines()
    state = GraphState()
    i = 0
    max_id = -1
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
            by_key: Dict[str, List[object]] = {}
            for k, v in attrs:
                by_key.setdefault(k, []).append(v)
            if kind == "node":
                node_id = int(by_key["id"][0])
                label = str(by_key.get("label", [str(node_id)])[0])
                pixel_vals = by_key.get("pixel", [])
                world_vals = by_key.get("world", [])
                node = GraphNode(
                    node_id=node_id,
                    pixel_row=float(pixel_vals[0]) if len(pixel_vals) >= 1 else 0.0,
                    pixel_col=float(pixel_vals[1]) if len(pixel_vals) >= 2 else 0.0,
                    world_x=float(world_vals[0]) if len(world_vals) >= 1 else None,
                    world_y=float(world_vals[1]) if len(world_vals) >= 2 else None,
                    world_z=float(world_vals[2]) if len(world_vals) >= 3 else 0.0,
                    label=label,
                    node_type=str(by_key.get("node_type", [""])[0]),
                )
                state.nodes[node_id] = node
                if node_id > max_id:
                    max_id = node_id
            else:
                state.edges.append(GraphEdge(
                    source=int(by_key["source"][0]),
                    target=int(by_key["target"][0]),
                    edge_type=str(by_key.get("edge_type", ["manual"])[0]),
                ))
        i += 1
    state.next_id = max_id + 1 if max_id >= 0 else 0
    return state


def pixel_to_world(row: float, col: float, img_height: int,
                   resolution: float, origin_x: float, origin_y: float) -> Tuple[float, float]:
    x_world = origin_x + col * resolution
    y_world = origin_y + (img_height - 1 - row) * resolution
    return x_world, y_world


def save_overlay_png(png_path: Path, map_image_path: Path, state: GraphState) -> None:
    """Render the graph overlaid on the map and save as PNG."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(map_image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    node_radius = 4
    edge_color = (239, 68, 68, 220)
    node_color = (59, 130, 246, 240)
    label_color = (251, 191, 36, 255)

    # Draw edges
    for edge in state.edges:
        if edge.source in state.nodes and edge.target in state.nodes:
            ns = state.nodes[edge.source]
            nt = state.nodes[edge.target]
            draw.line(
                [(ns.pixel_col, ns.pixel_row), (nt.pixel_col, nt.pixel_row)],
                fill=edge_color, width=2,
            )

    # Draw nodes
    for node in state.nodes.values():
        x, y = node.pixel_col, node.pixel_row
        draw.ellipse(
            [x - node_radius, y - node_radius, x + node_radius, y + node_radius],
            fill=node_color, outline=(255, 255, 255, 240), width=1,
        )
        if font:
            draw.text(
                (x + node_radius + 2, y - node_radius - 2),
                str(node.node_id), fill=label_color, font=font,
            )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)


def save_gml(path: Path, state: GraphState,
             img_height: Optional[int] = None,
             resolution: Optional[float] = None,
             origin_x: Optional[float] = None,
             origin_y: Optional[float] = None) -> None:
    has_world = resolution is not None and origin_x is not None and origin_y is not None
    out: List[str] = ["graph ["]
    for node in sorted(state.nodes.values(), key=lambda n: n.node_id):
        out.append("  node [")
        out.append(f"    id {node.node_id}")
        out.append(f'    label "{node.label or node.node_id}"')
        if has_world and img_height is not None:
            wx, wy = pixel_to_world(node.pixel_row, node.pixel_col,
                                    img_height, resolution, origin_x, origin_y)
            out.append(f"    world {format_value(wx)}")
            out.append(f"    world {format_value(wy)}")
            out.append(f"    world {format_value(node.world_z)}")
        elif node.world_x is not None and node.world_y is not None:
            out.append(f"    world {format_value(node.world_x)}")
            out.append(f"    world {format_value(node.world_y)}")
            out.append(f"    world {format_value(node.world_z)}")
        out.append(f"    pixel {int(round(node.pixel_row))}")
        out.append(f"    pixel {int(round(node.pixel_col))}")
        if node.node_type:
            out.append(f'    node_type "{node.node_type}"')
        out.append("  ]")
    for edge in state.edges:
        out.append("  edge [")
        out.append(f"    source {edge.source}")
        out.append(f"    target {edge.target}")
        if edge.source in state.nodes and edge.target in state.nodes:
            ns, nt = state.nodes[edge.source], state.nodes[edge.target]
            dist = math.hypot(ns.pixel_row - nt.pixel_row, ns.pixel_col - nt.pixel_col)
            out.append(f"    weight {format_value(dist)}")
        if edge.edge_type:
            out.append(f'    edge_type "{edge.edge_type}"')
        out.append("  ]")
    out.append("]")
    path.write_text("\n".join(out) + "\n")


def parse_map_yaml(path: Path) -> Tuple[float, float, float]:
    resolution = origin = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("resolution:"):
            resolution = float(line.split(":", 1)[1].strip())
        elif line.startswith("origin:"):
            origin = ast.literal_eval(line.split(":", 1)[1].strip())
    if resolution is None or origin is None or len(origin) < 2:
        raise ValueError(f"Could not parse resolution/origin from {path}")
    return resolution, float(origin[0]), float(origin[1])


# ---------------------------------------------------------------------------
# HTML frontend (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Graph Editor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }
  #toolbar {
    position: fixed; top: 0; left: 0; right: 0; height: 44px; z-index: 10;
    background: #16213e; display: flex; align-items: center; padding: 0 16px; gap: 12px;
    border-bottom: 1px solid #0f3460; font-size: 14px;
  }
  #toolbar button {
    padding: 5px 14px; border: 1px solid #0f3460; border-radius: 4px;
    background: #1a1a2e; color: #eee; cursor: pointer; font-size: 13px;
  }
  #toolbar button:hover { background: #0f3460; }
  #toolbar button.active { background: #e94560; border-color: #e94560; }
  #status { margin-left: auto; font-size: 12px; color: #aaa; }
  #canvas-wrap {
    position: fixed; top: 44px; left: 0; right: 0; bottom: 0; overflow: auto; cursor: crosshair;
  }
  canvas { display: block; }
</style>
</head>
<body>
<div id="toolbar">
  <button id="btn-node" class="active" onclick="setMode('node')">Node (n)</button>
  <button id="btn-edge" onclick="setMode('edge')">Edge (e)</button>
  <span style="color:#555">|</span>
  <button onclick="undo()">Undo (Ctrl+z)</button>
  <button onclick="save()">Save (Ctrl+s)</button>
  <span id="status">Loading...</span>
</div>
<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

let img = new Image();
let nodes = [];          // {id, row, col, label}
let edges = [];          // {source, target}
let nextId = 0;
let mode = 'node';       // 'node' or 'edge'
let edgeStart = null;    // node id or null
let undoStack = [];      // [{type, ...}]
let dirty = false;
let scale = 1;

const NODE_RADIUS = 6;
const SNAP_RADIUS = 14;
const NODE_COLOR = '#3b82f6';
const NODE_SELECTED = '#22c55e';
const EDGE_COLOR = 'rgba(239, 68, 68, 0.85)';
const LABEL_COLOR = '#fbbf24';

// --- Init ---
fetch('/api/state').then(r => r.json()).then(data => {
  nodes = data.nodes;
  edges = data.edges;
  nextId = data.next_id;
  img.src = '/map-image';
});
img.onload = () => {
  canvas.width = img.width;
  canvas.height = img.height;
  draw();
  updateStatus();
};

// --- Drawing ---
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  // Edges
  ctx.strokeStyle = EDGE_COLOR;
  ctx.lineWidth = 2;
  for (const e of edges) {
    const s = nodes.find(n => n.id === e.source);
    const t = nodes.find(n => n.id === e.target);
    if (s && t) {
      ctx.beginPath();
      ctx.moveTo(s.col, s.row);
      ctx.lineTo(t.col, t.row);
      ctx.stroke();
    }
  }

  // Nodes
  for (const n of nodes) {
    ctx.beginPath();
    ctx.arc(n.col, n.row, NODE_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = (edgeStart === n.id) ? NODE_SELECTED : NODE_COLOR;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '11px monospace';
    ctx.fillText(n.id, n.col + NODE_RADIUS + 2, n.row - NODE_RADIUS);
  }
}

function updateStatus() {
  const d = dirty ? ' *' : '';
  status.textContent = `Nodes: ${nodes.length}  Edges: ${edges.length}  Mode: ${mode}${d}`;
}

function setMode(m) {
  mode = m;
  edgeStart = null;
  document.getElementById('btn-node').classList.toggle('active', m === 'node');
  document.getElementById('btn-edge').classList.toggle('active', m === 'edge');
  draw();
  updateStatus();
}

function findNearest(x, y) {
  let best = null, bestDist = SNAP_RADIUS;
  for (const n of nodes) {
    const d = Math.hypot(n.col - x, n.row - y);
    if (d < bestDist) { bestDist = d; best = n; }
  }
  return best;
}

// --- Mouse ---
canvas.addEventListener('click', (ev) => {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
  const y = (ev.clientY - rect.top) * (canvas.height / rect.height);

  if (mode === 'node') {
    const node = { id: nextId++, row: y, col: x, label: '' + (nextId - 1) };
    nodes.push(node);
    undoStack.push({ type: 'add_node', node });
    dirty = true;
  } else if (mode === 'edge') {
    const clicked = findNearest(x, y);
    if (!clicked) return;
    if (edgeStart === null) {
      edgeStart = clicked.id;
    } else {
      if (edgeStart !== clicked.id) {
        const exists = edges.some(e =>
          (e.source === edgeStart && e.target === clicked.id) ||
          (e.source === clicked.id && e.target === edgeStart));
        if (!exists) {
          const edge = { source: edgeStart, target: clicked.id };
          edges.push(edge);
          undoStack.push({ type: 'add_edge', edge });
          dirty = true;
        }
      }
      edgeStart = null;
    }
  }
  draw();
  updateStatus();
});

canvas.addEventListener('contextmenu', (ev) => {
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
  const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
  const clicked = findNearest(x, y);
  if (!clicked) return;

  // Delete node
  const removedEdges = edges.filter(e => e.source === clicked.id || e.target === clicked.id);
  edges = edges.filter(e => e.source !== clicked.id && e.target !== clicked.id);
  nodes = nodes.filter(n => n.id !== clicked.id);
  undoStack.push({ type: 'delete_node', node: clicked, removedEdges });
  if (edgeStart === clicked.id) edgeStart = null;
  dirty = true;
  draw();
  updateStatus();
});

// --- Keyboard ---
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'n') setMode('node');
  else if (ev.key === 'e') setMode('edge');
  else if (ev.key === 'z' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); undo(); }
  else if (ev.key === 's' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); save(); }
});

function undo() {
  if (!undoStack.length) return;
  const action = undoStack.pop();
  if (action.type === 'add_node') {
    edges = edges.filter(e => e.source !== action.node.id && e.target !== action.node.id);
    nodes = nodes.filter(n => n.id !== action.node.id);
  } else if (action.type === 'add_edge') {
    edges = edges.filter(e =>
      !(e.source === action.edge.source && e.target === action.edge.target));
  } else if (action.type === 'delete_node') {
    nodes.push(action.node);
    for (const e of action.removedEdges) edges.push(e);
  }
  dirty = true;
  draw();
  updateStatus();
}

function save() {
  const data = { nodes, edges, next_id: nextId };
  fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }).then(r => r.json()).then(resp => {
    dirty = false;
    status.textContent = resp.message;
    setTimeout(updateStatus, 3000);
  }).catch(err => {
    status.textContent = 'Save failed: ' + err;
  });
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class GraphHandler(BaseHTTPRequestHandler):
    state: GraphState
    map_image_bytes: bytes
    map_content_type: str
    map_image_path: Path
    output_path: Path
    img_height: int
    resolution: Optional[float]
    origin_x: Optional[float]
    origin_y: Optional[float]

    def log_message(self, fmt, *args):
        # Quiet logging
        pass

    def do_GET(self):
        if self.path == "/":
            self._respond(200, "text/html", HTML_PAGE.encode())
        elif self.path == "/map-image":
            self._respond(200, self.map_content_type, self.map_image_bytes)
        elif self.path == "/api/state":
            data = self.state.to_dict()
            self._respond(200, "application/json", json.dumps(data).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            self._apply_state(body)
            save_gml(self.output_path, self.state,
                     img_height=self.img_height,
                     resolution=self.resolution,
                     origin_x=self.origin_x,
                     origin_y=self.origin_y)
            png_path = self.output_path.with_suffix(".png")
            save_overlay_png(png_path, self.map_image_path, self.state)
            n, e = len(self.state.nodes), len(self.state.edges)
            msg = f"Saved to {self.output_path} + {png_path.name} ({n} nodes, {e} edges)"
            print(f"  [save] {msg}")
            self._respond(200, "application/json",
                          json.dumps({"ok": True, "message": msg}).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _apply_state(self, data: dict):
        self.state.nodes.clear()
        self.state.edges.clear()
        for nd in data.get("nodes", []):
            node = GraphNode(
                node_id=int(nd["id"]),
                pixel_row=float(nd["row"]),
                pixel_col=float(nd["col"]),
                label=str(nd.get("label", nd["id"])),
            )
            self.state.nodes[node.node_id] = node
        for ed in data.get("edges", []):
            self.state.edges.append(GraphEdge(
                source=int(ed["source"]), target=int(ed["target"])))
        self.state.next_id = int(data.get("next_id", 0))

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web-based graph editor: draw waypoints on a map and save as GML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--map-image", type=Path, required=True,
                        help="Background map image (PNG/PGM/JPG)")
    parser.add_argument("--output", type=Path, default=Path("graph.gml"),
                        help="Output GML path (default: graph.gml)")
    parser.add_argument("--load", type=Path, default=None,
                        help="Load an existing GML to continue editing")
    parser.add_argument("--map-yaml", type=Path, default=None,
                        help="Map YAML for world coordinate conversion")
    parser.add_argument("--port", type=int, default=8765,
                        help="HTTP server port (default: 8765)")
    args = parser.parse_args()

    if not args.map_image.exists():
        print(f"Error: map image not found: {args.map_image}", file=sys.stderr)
        sys.exit(1)

    # Read image bytes
    map_image_bytes = args.map_image.read_bytes()
    suffix = args.map_image.suffix.lower()
    content_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "pgm": "image/x-portable-graymap"}.get(suffix.lstrip("."), "image/png")

    # Get image dimensions
    from PIL import Image
    with Image.open(args.map_image) as im:
        img_height = im.height

    resolution = origin_x = origin_y = None
    if args.map_yaml:
        resolution, origin_x, origin_y = parse_map_yaml(args.map_yaml)
        print(f"Map YAML loaded: resolution={resolution}, origin=({origin_x}, {origin_y})")

    if args.load and args.load.exists():
        state = load_gml(args.load)
        print(f"Loaded {len(state.nodes)} nodes and {len(state.edges)} edges from {args.load}")
    else:
        state = GraphState()

    # Configure handler class attributes
    handler = partial(GraphHandler)
    GraphHandler.state = state
    GraphHandler.map_image_bytes = map_image_bytes
    GraphHandler.map_content_type = content_type
    GraphHandler.map_image_path = args.map_image.resolve()
    GraphHandler.output_path = args.output.resolve()
    GraphHandler.img_height = img_height
    GraphHandler.resolution = resolution
    GraphHandler.origin_x = origin_x
    GraphHandler.origin_y = origin_y

    server = HTTPServer(("0.0.0.0", args.port), GraphHandler)
    url = f"http://localhost:{args.port}"
    print(f"\nGraph editor running at: {url}")
    print(f"  (On remote machine, forward port: ssh -L {args.port}:localhost:{args.port} ...)")
    print(f"  Output: {args.output.resolve()}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
