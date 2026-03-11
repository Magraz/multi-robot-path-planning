#!/usr/bin/env python3
"""Generate search metrics plots from CSV logs and GML graph files.

Outputs:
  1) One combined boxplot figure (grouped by metric), with all graphs:
     - total explored percentage by graph, split by method color
     - elapsed_sec, split by method color
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
from matplotlib.patches import Patch


NODE_LINE_RE = re.compile(r"^\s*node\s*\[")
METHOD_COLORS = {
    "baseline": "#e76f51",
    "mespp": "#457b9d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="*",
        default=None,
        help="CSV metrics files. If omitted, auto-discovers search_metrics_*.csv with method tags.",
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Use only the lowest K runs by elapsed_sec per (graph, method).",
    )
    return parser.parse_args()


def discover_default_csvs() -> list[Path]:
    candidates = sorted(Path("results").glob("search_metrics_*.csv"))
    return [p for p in candidates if infer_method_from_filename(p) is not None]


def infer_method_from_filename(csv_path: Path) -> str | None:
    stem = csv_path.stem.lower()
    if "mespp" in stem:
        return "mespp"
    if "baseline" in stem:
        return "baseline"
    return None


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
    skipped_csvs: list[Path] = []

    for csv_path in csv_paths:
        method = infer_method_from_filename(csv_path)
        if method is None:
            skipped_csvs.append(csv_path)
            continue

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
                        "method": method,
                        "elapsed_sec": elapsed,
                        "total_nodes_explored": total_nodes_explored,
                        "explored_pct": explored_pct,
                    }
                )

    if skipped_csvs:
        print("Skipped CSV files without method tag (expected 'baseline' or 'mespp'):")
        for p in skipped_csvs:
            print(f"  {p}")

    if not rows:
        raise RuntimeError("No metric rows loaded from CSV inputs.")
    return rows, node_counts_by_graph


def select_fastest_rows(rows: list[dict], top_k: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["graph"], row["method"])].append(row)

    selected: list[dict] = []
    for key in sorted(grouped):
        fastest = sorted(grouped[key], key=lambda r: r["elapsed_sec"])[:top_k]
        selected.extend(fastest)
    return selected


def make_grouped_boxplots(
    rows: list[dict], node_counts_by_graph: dict[str, int], out_dir: Path
) -> Path:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    method_set: set[str] = set()
    graph_set: set[str] = set()
    for row in rows:
        grouped[(row["graph"], row["method"])].append(row)
        graph_set.add(row["graph"])
        method_set.add(row["method"])

    graphs = sorted(graph_set)
    methods = [m for m in ("baseline", "mespp") if m in method_set]
    for m in sorted(method_set):
        if m not in methods:
            methods.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3))

    def draw_metric(ax: plt.Axes, metric_key: str, ylabel: str, title: str) -> None:
        data: list[list[float]] = []
        positions: list[float] = []
        colors: list[str] = []
        tick_positions: list[float] = []
        tick_labels: list[str] = []

        group_width = len(methods) + 1
        for graph_idx, graph in enumerate(graphs):
            base = graph_idx * group_width + 1
            graph_positions: list[float] = []
            for method_idx, method in enumerate(methods):
                group_rows = grouped.get((graph, method), [])
                if not group_rows:
                    continue
                values = [r[metric_key] for r in group_rows]
                pos = base + method_idx
                data.append(values)
                positions.append(pos)
                colors.append(METHOD_COLORS.get(method, "#8d99ae"))
                graph_positions.append(pos)

            if graph_positions:
                tick_positions.append(sum(graph_positions) / len(graph_positions))
                tick_labels.append(
                    f"{Path(graph).stem}\n(n={node_counts_by_graph[graph]})"
                )

        artists = ax.boxplot(
            data,
            positions=positions,
            widths=0.72,
            patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.5},
        )
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=10, ha="right")
        ax.grid(axis="y", alpha=0.25)

    draw_metric(
        axes[0],
        metric_key="explored_pct",
        ylabel="Percent of graph nodes",
        title="Total nodes explored (%)",
    )
    draw_metric(
        axes[1],
        metric_key="elapsed_sec",
        ylabel="Seconds",
        title="Elapsed time",
    )

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS.get(m, "#8d99ae"),
            edgecolor="black",
            label=m.upper(),
        )
        for m in methods
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.suptitle(
        "MESPP vs. Baseline grouped by percentage of nodes explored, and elapsed time.",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = out_dir / "boxplots_grouped_by_metric.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def make_all_maps_scatter(
    rows: list[dict], node_counts_by_graph: dict[str, int], out_dir: Path
) -> Path:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["graph"], row["method"])].append(row)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    graph_markers = ["o", "s", "^", "D", "P", "X"]
    graph_to_marker: dict[str, str] = {}
    for idx, graph in enumerate(sorted({r["graph"] for r in rows})):
        graph_to_marker[graph] = graph_markers[idx % len(graph_markers)]

    for graph, method in sorted(grouped):
        graph_rows = grouped[(graph, method)]
        ax.scatter(
            [r["elapsed_sec"] for r in graph_rows],
            [r["explored_pct"] for r in graph_rows],
            s=52,
            alpha=0.85,
            color=METHOD_COLORS.get(method, "#8d99ae"),
            marker=graph_to_marker[graph],
            label=f"{Path(graph).stem} | {method.upper()}",
        )

    ax.set_title("All maps: elapsed_sec vs total nodes explored (%) | method-colored")
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

    csv_paths = args.csv if args.csv else discover_default_csvs()
    if not csv_paths:
        raise RuntimeError(
            "No CSV files found. Provide --csv files with 'baseline' or 'mespp' in filename."
        )

    rows, node_counts_by_graph = load_records(csv_paths, args.gml_dir)
    rows = select_fastest_rows(rows, args.top_k)
    grouped_boxplot_path = make_grouped_boxplots(rows, node_counts_by_graph, out_dir)
    all_maps_path = make_all_maps_scatter(rows, node_counts_by_graph, out_dir)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["graph"], row["method"])] += 1

    print("Node counts by graph:")
    for graph in sorted(node_counts_by_graph):
        print(f"  {graph}: {node_counts_by_graph[graph]} nodes")

    print(f"\nRows selected per (graph, method) using top_k={args.top_k}:")
    for graph, method in sorted(counts):
        print(f"  {graph} | {method}: {counts[(graph, method)]}")

    print("\nGenerated plots:")
    print(f"  {grouped_boxplot_path}")
    print(f"  {all_maps_path}")


if __name__ == "__main__":
    main()
