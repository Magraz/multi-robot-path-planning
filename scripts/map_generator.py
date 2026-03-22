#!/usr/bin/env python3
r"""
Procedural office map generator for multi-robot search scenarios.
Generates PNG, PGM, and YAML files compatible with nav2 map_server.
ros2 run mr_path_planning map_generator \                          
  --width 800 --height 500 \
  --resolution 0.05 \
  --seed 123 \
  --name world_1 \
  --output-dir src/mr_path_planning/world/bitmaps
"""

import argparse
import random
import numpy as np
from PIL import Image
from pathlib import Path


# Nav2 map conventions: 254 = free (white), 0 = occupied (black), 205 = unknown (gray)
FREE = 254
OCCUPIED = 0
WALL_THICKNESS = 2  # pixels


def create_empty_map(width_px, height_px):
    """Create a map filled with free space, surrounded by outer walls."""
    grid = np.full((height_px, width_px), FREE, dtype=np.uint8)
    # Outer walls
    grid[:WALL_THICKNESS, :] = OCCUPIED
    grid[-WALL_THICKNESS:, :] = OCCUPIED
    grid[:, :WALL_THICKNESS] = OCCUPIED
    grid[:, -WALL_THICKNESS:] = OCCUPIED
    return grid


def draw_rect_wall(grid, x, y, w, h):
    """Draw a filled rectangle of walls."""
    grid[y : y + h, x : x + w] = OCCUPIED


def draw_hline(grid, x, y, length):
    """Draw a horizontal wall."""
    draw_rect_wall(grid, x, y, length, WALL_THICKNESS)


def draw_vline(grid, x, y, length):
    """Draw a vertical wall."""
    draw_rect_wall(grid, x, y, WALL_THICKNESS, length)


def add_door(grid, x, y, horizontal, door_width=50):
    """Punch a door opening in a wall segment."""
    if horizontal:
        grid[y : y + WALL_THICKNESS, x : x + door_width] = FREE
    else:
        grid[y : y + door_width, x : x + WALL_THICKNESS] = FREE


def generate_corridor_layout(grid, rng):
    """Generate a variable number of corridors that divide the map."""
    h, w = grid.shape
    margin = 40
    min_spacing = 120
    corridors = []

    # Scale max corridors by map size
    max_h = max(1, (h - 2 * margin) // 200)
    max_v = max(1, (w - 2 * margin) // 200)
    n_h = rng.randint(1, max(1, min(max_h, 3)))
    n_v = rng.randint(1, max(1, min(max_v, 3)))

    # Place horizontal corridors with spacing constraints
    h_positions = _place_corridors(h, margin, n_h, min_spacing, rng)
    for cy in h_positions:
        cw = rng.randint(40, 70)
        if cy + cw + WALL_THICKNESS >= h - margin:
            cw = h - margin - cy - WALL_THICKNESS
        if cw < 30:
            continue
        draw_hline(grid, 0, cy, w)
        draw_hline(grid, 0, cy + cw, w)
        corridors.append(("h", cy, cw))

    # Place vertical corridors with spacing constraints
    v_positions = _place_corridors(w, margin, n_v, min_spacing, rng)
    for cx in v_positions:
        cw = rng.randint(40, 70)
        if cx + cw + WALL_THICKNESS >= w - margin:
            cw = w - margin - cx - WALL_THICKNESS
        if cw < 30:
            continue
        draw_vline(grid, cx, 0, h)
        draw_vline(grid, cx + cw, 0, h)
        corridors.append(("v", cx, cw))

    # Clear all corridor intersections
    for c1 in corridors:
        for c2 in corridors:
            if c1[0] == "h" and c2[0] == "v":
                hy, hw = c1[1], c1[2]
                vx, vw = c2[1], c2[2]
                grid[hy : hy + hw + WALL_THICKNESS, vx : vx + vw + WALL_THICKNESS] = (
                    FREE
                )

    return corridors


def _place_corridors(span, margin, count, min_spacing, rng):
    """Place `count` corridor positions within `span`, keeping minimum spacing."""
    positions = []
    for _ in range(count * 10):
        pos = rng.randint(margin + 40, span - margin - 80)
        if all(abs(pos - p) >= min_spacing for p in positions):
            positions.append(pos)
        if len(positions) == count:
            break
    positions.sort()
    return positions


def get_regions(grid, corridors):
    """Identify rectangular regions between corridors and outer walls."""
    h, w = grid.shape
    margin = WALL_THICKNESS

    h_bounds = [margin]
    v_bounds = [margin]

    for orient, pos, cw in corridors:
        if orient == "h":
            h_bounds.extend([pos, pos + cw + WALL_THICKNESS])
        else:
            v_bounds.extend([pos, pos + cw + WALL_THICKNESS])

    h_bounds.append(h - margin)
    v_bounds.append(w - margin)
    h_bounds = sorted(set(h_bounds))
    v_bounds = sorted(set(v_bounds))

    regions = []
    for i in range(len(h_bounds) - 1):
        for j in range(len(v_bounds) - 1):
            y1, y2 = h_bounds[i], h_bounds[i + 1]
            x1, x2 = v_bounds[j], v_bounds[j + 1]
            rw, rh = x2 - x1, y2 - y1
            if rw > 60 and rh > 60:
                regions.append((x1, y1, rw, rh))
    return regions


def subdivide_region_into_rooms(grid, region, rng):
    """Divide a region into a grid of rooms with doors."""
    x, y, w, h = region
    min_room = 80
    door_width = 40

    # Determine grid of rooms
    n_cols = max(1, w // rng.randint(min_room, min_room + 60))
    n_rows = max(1, h // rng.randint(min_room, min_room + 60))

    col_w = w // max(n_cols, 1)
    row_h = h // max(n_rows, 1)

    # Draw vertical room walls
    for i in range(1, n_cols):
        wall_x = x + i * col_w
        draw_vline(grid, wall_x, y, h)
        # One door per row in this wall
        for j in range(n_rows):
            ry = y + j * row_h
            rh = row_h if j < n_rows - 1 else (y + h - ry)
            door_y = ry + (rh - door_width) // 2
            if door_y > ry + 4 and door_y + door_width < ry + rh - 4:
                add_door(grid, wall_x, door_y, horizontal=False, door_width=door_width)

    # Draw horizontal room walls
    for j in range(1, n_rows):
        wall_y = y + j * row_h
        draw_hline(grid, x, wall_y, w)
        # One door per column in this wall
        for i in range(n_cols):
            rx = x + i * col_w
            rw = col_w if i < n_cols - 1 else (x + w - rx)
            door_x = rx + (rw - door_width) // 2
            if door_x > rx + 4 and door_x + door_width < rx + rw - 4:
                add_door(grid, door_x, wall_y, horizontal=True, door_width=door_width)


def add_corridor_doors(grid, corridors, regions, rng):
    """Punch evenly-spaced doors in corridor walls aligned with rooms."""
    door_width = 50

    for orient, pos, cw in corridors:
        if orient == "h":
            for wall_y in [pos, pos + cw]:
                # Find regions that border this wall
                for rx, ry, rw, rh in regions:
                    if ry == wall_y + WALL_THICKNESS or ry + rh == wall_y:
                        # Punch a centered door for this region
                        door_x = rx + (rw - door_width) // 2
                        if door_x > rx + 4 and door_x + door_width < rx + rw - 4:
                            add_door(
                                grid,
                                door_x,
                                wall_y,
                                horizontal=True,
                                door_width=door_width,
                            )
        else:
            for wall_x in [pos, pos + cw]:
                for rx, ry, rw, rh in regions:
                    if rx == wall_x + WALL_THICKNESS or rx + rw == wall_x:
                        door_y = ry + (rh - door_width) // 2
                        if door_y > ry + 4 and door_y + door_width < ry + rh - 4:
                            add_door(
                                grid,
                                wall_x,
                                door_y,
                                horizontal=False,
                                door_width=door_width,
                            )


def generate_office_map(width_px=800, height_px=600, resolution=0.05, seed=None):
    """
    Generate a procedural office map.

    Args:
        width_px: Map width in pixels.
        height_px: Map height in pixels.
        resolution: Meters per pixel (0.05 = 5cm per pixel).
        seed: Random seed for reproducibility.

    Returns:
        grid: 2D numpy array of the map.
        resolution: The resolution used.
    """
    rng = random.Random(seed)

    grid = create_empty_map(width_px, height_px)
    corridors = generate_corridor_layout(grid, rng)
    regions = get_regions(grid, corridors)

    for region in regions:
        subdivide_region_into_rooms(grid, region, rng)

    add_corridor_doors(grid, corridors, regions, rng)

    return grid, resolution


def find_spawn_positions(grid, resolution, count=3):
    """Find free-space positions for robot spawning, spread across the map."""
    h, w = grid.shape
    free_ys, free_xs = np.where(grid == FREE)

    # Divide map into vertical strips and pick one free point per strip
    strip_width = w // count
    positions = []
    for i in range(count):
        x_lo = strip_width * i + 20
        x_hi = strip_width * (i + 1) - 20
        mask = (free_xs >= x_lo) & (free_xs < x_hi)
        strip_xs = free_xs[mask]
        strip_ys = free_ys[mask]
        if len(strip_xs) == 0:
            continue
        # Pick the point closest to the vertical center of the strip
        mid_y = h // 2
        idx = np.argmin(np.abs(strip_ys - mid_y))
        px_x, px_y = int(strip_xs[idx]), int(strip_ys[idx])
        # Convert pixel to world meters (origin at center, y flipped)
        wx = (px_x - w / 2.0) * resolution
        wy = -(px_y - h / 2.0) * resolution
        positions.append((wx, wy))

    return positions


def save_map(grid, resolution, output_dir, name="office_map", world_dir=None):
    """
    Save map as PNG, PGM, YAML, and Stage .world files.

    Args:
        grid: 2D numpy array (uint8) of the map.
        resolution: Meters per pixel.
        output_dir: Directory to save bitmap files (png, pgm, yaml).
        name: Base name for the output files.
        world_dir: Directory for .world file. Defaults to parent of output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if world_dir is None:
        world_dir = output_dir.parent
    else:
        world_dir = Path(world_dir)
    world_dir.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(grid, mode="L")

    # Save PNG
    png_path = output_dir / f"{name}.png"
    img.save(str(png_path))

    # Save PGM (P5 binary format)
    pgm_path = output_dir / f"{name}.pgm"
    img.save(str(pgm_path))

    # Save YAML for nav2 map_server
    h, w = grid.shape
    origin_x = -(w * resolution) / 2.0
    origin_y = -(h * resolution) / 2.0

    yaml_path = output_dir / f"{name}.yaml"
    yaml_content = (
        f"image: {name}.pgm\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin_x}, {origin_y}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    yaml_path.write_text(yaml_content)

    # Save Stage .world file
    width_m = w * resolution
    height_m = h * resolution
    bitmap_rel = f"bitmaps/{name}.png"
    gui_scale = max(10, int(800 / width_m))

    spawns = find_spawn_positions(grid, resolution, count=3)

    world_path = world_dir / f"{name}.world"
    world_content = f"""\
# {name}.world - auto-generated office environment

include "include/robots.inc"

resolution {resolution}

interval_sim 100

define floorplan model
(
  color "gray30"
  boundary 1
  gui_nose 0
  gui_grid 0
  gui_outline 0
  gripper_return 0
  fiducial_return 0
  laser_return 1
)

window
(
  size [ {w} {h} ]
  scale {gui_scale}
  center [ 0 0 ]
  rotate [ 0 0 ]
  show_data 1
)

floorplan
(
  name "{name}"
  size [{width_m:.3f} {height_m:.3f} 0.800]
  pose [0 0 0 0]
  bitmap "{bitmap_rel}"
  gui_move 0
)

turtlebot_with_laser
(
  name "robot_0"
  color "red"
  pose [ {spawns[0][0]:.2f} {spawns[0][1]:.2f} 0 45 ]
)

turtlebot_with_laser
(
  name "robot_1"
  color "purple"
  pose [ {spawns[1][0]:.2f} {spawns[1][1]:.2f} 0 45 ]
)

turtlebot_with_laser
(
  name "target_0"
  color "blue"
  pose [ {spawns[2][0]:.2f} {spawns[2][1]:.2f} 0 45 ]
)
"""
    world_path.write_text(world_content)

    print(f"Saved: {png_path}")
    print(f"Saved: {pgm_path}")
    print(f"Saved: {yaml_path}")
    print(f"Saved: {world_path}")
    print(f"Map size: {w}x{h} px = {width_m:.1f}x{height_m:.1f} m")

    return png_path, pgm_path, yaml_path, world_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate office-style maps for nav2 multi-robot search."
    )
    parser.add_argument(
        "--width", type=int, default=800, help="Map width in pixels (default: 800)"
    )
    parser.add_argument(
        "--height", type=int, default=600, help="Map height in pixels (default: 600)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Meters per pixel (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="office_map",
        help="Base name for output files (default: office_map)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--world-dir",
        type=str,
        default=None,
        help="Output directory for .world file (default: parent of output-dir)",
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of maps to generate (default: 1)"
    )
    args = parser.parse_args()

    for i in range(args.count):
        seed = args.seed + i if args.seed is not None else None
        name = f"{args.name}_{i}" if args.count > 1 else args.name
        grid, resolution = generate_office_map(
            width_px=args.width,
            height_px=args.height,
            resolution=args.resolution,
            seed=seed,
        )
        save_map(grid, resolution, args.output_dir, name, world_dir=args.world_dir)


if __name__ == "__main__":
    main()
