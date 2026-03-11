#!/usr/bin/env python3
"""Generate search metrics plots from CSV logs and GML graph files.

Outputs:
  1) One combined boxplot figure (grouped by metric), with all graphs:
     - total explored percentage by graph
     - elapsed_sec
  2) One combined all-maps scatter plot:
     - x: elapsed_sec
     - y: total explored percentage
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NODE_LINE_RE = re.compile(r"^\s*node\s*\[")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=[
            Path("results/search_metrics_world_1.csv"),
            Path("results/search_metrics_world_2.csv"),
            Path("results/search_metrics_world_3.csv"),
        ],
        help="CSV metrics files.",
    )
    parser.add_argument(
        "--gml-dir",
        type=Path,
        default=Path("src/mr_path_planning/world/bitmaps"),
        help="Directory containing graph .gml files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for generated plots.",
    )
    return parser.parse_args()


def count_gml_nodes(gml_path: Path) -> int:
    count = 0
    with gml_path.open("r", encoding="utf-8") as f:
        for line in f:
            if NODE_LINE_RE.match(line):
                count += 1
    if count <= 0:
        raise ValueError(f"No nodes found in GML file: {gml_path}")
    return count


def resolve_graph_path(graph_name: str, gml_dir: Path) -> Path:
    direct = gml_dir / graph_name
    if direct.exists():
        return direct

    # Fallback for sparse naming if only dense graph exists.
    if graph_name.endswith("_sparse.gml"):
        fallback = gml_dir / graph_name.replace("_sparse.gml", ".gml")
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Could not resolve graph file for '{graph_name}' in {gml_dir}"
    )


def load_records(
    csv_paths: list[Path], gml_dir: Path
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    node_counts_by_graph: dict[str, int] = {}

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                graph = row["graph"].strip()
                if not graph or graph == "graph":
                    continue

                if graph not in node_counts_by_graph:
                    gml_path = resolve_graph_path(graph, gml_dir)
                    node_counts_by_graph[graph] = count_gml_nodes(gml_path)

                elapsed = float(row["elapsed_sec"])
                nodes_robot_0 = float(row["nodes_robot_0"])
                nodes_robot_1 = float(row["nodes_robot_1"])
                total_nodes_explored = nodes_robot_0 + nodes_robot_1
                total_nodes_in_graph = node_counts_by_graph[graph]
                explored_pct = (total_nodes_explored / total_nodes_in_graph) * 100.0

                rows.append(
                    {
                        "graph": graph,
                        "elapsed_sec": elapsed,
                        "total_nodes_explored": total_nodes_explored,
                        "explored_pct": explored_pct,
                    }
                )

    if not rows:
        raise RuntimeError("No metric rows loaded from CSV inputs.")
    return rows, node_counts_by_graph


def make_grouped_boxplots(
    rows: list[dict], node_counts_by_graph: dict[str, int], out_dir: Path
) -> Path:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["graph"]].append(row)

    graphs = sorted(grouped)
    labels = [f"{Path(g).stem}\n(n={node_counts_by_graph[g]})" for g in graphs]
    explored_data = [[r["explored_pct"] for r in grouped[g]] for g in graphs]
    elapsed_data = [[r["elapsed_sec"] for r in grouped[g]] for g in graphs]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

    axes[0].boxplot(
        explored_data,
        labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#8ecae6"},
        medianprops={"color": "#1d3557", "linewidth": 1.5},
    )
    axes[0].set_title("Total nodes explored (%)")
    axes[0].set_ylabel("Percent of graph nodes")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(
        elapsed_data,
        labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#b7e4c7"},
        medianprops={"color": "#2d6a4f", "linewidth": 1.5},
    )
    axes[1].set_title("Elapsed time (elapsed_sec)")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(12)
            tick.set_ha("right")

    fig.suptitle("Search metrics grouped by data type across all graphs", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / "boxplots_grouped_by_metric.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def make_all_maps_scatter(
    rows: list[dict], node_counts_by_graph: dict[str, int], out_dir: Path
) -> Path:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["graph"]].append(row)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    cmap = plt.get_cmap("tab10")

    for idx, graph in enumerate(sorted(grouped)):
        graph_rows = grouped[graph]
        ax.scatter(
            [r["elapsed_sec"] for r in graph_rows],
            [r["explored_pct"] for r in graph_rows],
            s=52,
            alpha=0.85,
            color=cmap(idx),
            label=f"{graph} (nodes={node_counts_by_graph[graph]})",
        )

    ax.set_title("All maps: elapsed_sec vs total nodes explored (%)")
    ax.set_xlabel("elapsed_sec (s)")
    ax.set_ylabel("Total explored nodes (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = out_dir / "all_maps_elapsed_vs_explored_pct.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, node_counts_by_graph = load_records(args.csv, args.gml_dir)
    grouped_boxplot_path = make_grouped_boxplots(rows, node_counts_by_graph, out_dir)
    all_maps_path = make_all_maps_scatter(rows, node_counts_by_graph, out_dir)

    print("Node counts by graph:")
    for graph in sorted(node_counts_by_graph):
        print(f"  {graph}: {node_counts_by_graph[graph]} nodes")

    print("\nGenerated plots:")
    print(f"  {grouped_boxplot_path}")
    print(f"  {all_maps_path}")


if __name__ == "__main__":
    main()
