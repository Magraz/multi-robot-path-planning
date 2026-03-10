#!/usr/bin/env python3

import argparse
import gurobipy as gp
from gurobipy import GRB
import igraph as ig
import numpy as np
import random
import re
import time
from pathlib import Path

from helper import Helper
from searchers import Searchers
from target import Target


class Mespp(Helper):
    """MILP solver for multi-searcher path planning."""

    def __init__(
        self,
        graph_path: str | None = None,
        horizon: int = 20,
        num_searchers: int = 2,
        target_start_vertex: int | None = None,
        searcher_starts: list[int] | None = None,
        motion: str = "uniform",
    ):
        # Default scenario (edit these values for new experiments).
        self.HORIZON = horizon

        # Build either a custom graph (GML) or default 2D grid graph.
        if graph_path:
            graph_file = Path(graph_path).expanduser().resolve()
            if not graph_file.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_file}")
            self.g = ig.Graph.Read_GML(str(graph_file))
            if self.g.is_directed():
                self.g = self.g.as_undirected(combine_edges="first")

            # Preserve map-like node geometry for plotting when available.
            pixel_layout = self._read_gml_pixel_layout(str(graph_file), self.g)
            if pixel_layout is not None:
                self.g["plot_layout"] = pixel_layout
        else:
            grid_side = 10
            self.g = ig.Graph.Lattice(dim=[grid_side, grid_side], circular=False)

        self.N = self.g.vcount()
        self.belief_g = self.g.copy()

        # Target starts at a fixed vertex; motion can be "uniform" or "stationary".
        if target_start_vertex is None:
            target_start_vertex = self.N // 2

        if not (0 <= target_start_vertex < self.N):
            raise ValueError(
                f"target_start_vertex={target_start_vertex} is out of range [0, {self.N - 1}]"
            )

        self.target = Target(
            self.g,
            N=self.N,
            initial_position=target_start_vertex,
            motion=motion,
        )

        # Searcher team configuration.
        if searcher_starts is None:
            # Pick the farthest vertices from target by shortest-path distance.
            dists = self.g.shortest_paths(target_start_vertex)[0]
            ranked = sorted(range(self.N), key=lambda v: (-dists[v], v))
            starts = [v for v in ranked if v != target_start_vertex][:num_searchers]
            if len(starts) < num_searchers:
                raise RuntimeError("Could not select enough valid searcher start vertices")
            searcher_starts = starts

        if len(searcher_starts) != num_searchers:
            raise ValueError(
                f"searcher_starts length ({len(searcher_starts)}) must equal num_searchers ({num_searchers})"
            )
        for s in searcher_starts:
            if not (0 <= s < self.N):
                raise ValueError(f"searcher start {s} is out of range [0, {self.N - 1}]")

        self.searchers = Searchers(
            self.g,
            N=self.N,
            M=num_searchers,
            initial_positions=np.array(searcher_starts),
            target_initial_position=target_start_vertex,
        )

        self.m = gp.Model("planner")

        # Decision variable containers
        self.presence = {}
        self.transition = {}
        self.legal_V = {}

        self.beliefs = None
        self.prop_beliefs = None
        self.capture = None

    @staticmethod
    def _read_gml_pixel_layout(graph_path: str, graph: ig.Graph):
        """Parse node pixel coordinates from GML and map them to igraph vertex order.

        Nvidia Swagger GML exports coordinates as repeated keys:
          pixel <x>
          pixel <y>
        This keeps only the last value when loaded via igraph, so we recover
        both values here from raw text.
        """
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return None

        node_blocks = re.findall(r"node\s*\[(.*?)\]", text, flags=re.DOTALL)
        if not node_blocks:
            return None

        id_to_xy = {}
        for block in node_blocks:
            id_match = re.search(r"\bid\s+(-?\d+)", block)
            pixel_vals = re.findall(r"\bpixel\s+(-?\d+(?:\.\d+)?)", block)
            if id_match and len(pixel_vals) >= 2:
                vid = int(id_match.group(1))
                x = float(pixel_vals[0])
                y = float(pixel_vals[1])
                # Negate y so image-style coordinates (down is +y) render naturally.
                id_to_xy[vid] = (x, -y)

        if not id_to_xy:
            return None

        # Build layout in igraph vertex index order; rely on vertex "id" attr from GML.
        layout = []
        for v in graph.vs:
            raw_id = v["id"] if "id" in graph.vertex_attributes() else v.index
            vid = int(float(raw_id))
            layout.append(id_to_xy.get(vid, (float(v.index), 0.0)))

        return layout

    def run(self):
        # Main execution pipeline: build model -> solve -> render frames.
        print("\n---------------MESPP started----------------\n")

        self.build_milp_variables()
        self.m.update()
        self.build_milp_constraints()
        self.configure_objective()
        self.plan()

        for t in range(self.HORIZON + 1):
            self.plot(t)
            self.target.updateTargetPosition()

        print("\n---------------Exiting----------------\n")

    def build_milp_variables(self):
        # Create all decision variables used in the MILP.
        print("\n>> Adding MILP Variables...\n")
        start_time = time.time()

        self.addTransitionVariables()
        self.addBeliefVariables()

        end_time = time.time()
        print(f"\n<< Finished adding MILP Variables in {end_time - start_time:.2f}s\n")

    def build_milp_constraints(self):
        # Add flow, belief propagation, and capture constraints.
        print("\n>> Adding MILP Constraints...\n")
        start_time = time.time()

        self.addTransitionConstraints()
        self.addBeliefConstraints()

        end_time = time.time()
        print(f"\n<< Finished adding MILP Constraints in {end_time - start_time:.2f}s\n")

    def configure_objective(self):
        # Maximize discounted capture probability over time.
        discount = 0.8
        discount_weights = np.array([discount**t for t in range(self.HORIZON + 1)])
        self.m.setObjective(discount_weights @ self.beliefs[0, :], GRB.MAXIMIZE)

    def plan(self):
        # Solve the optimization model.
        print("\n>> Planning routine started...\n")
        start_time = time.time()

        self.m.optimize()
        if self.m.status == GRB.OPTIMAL:
            print("Success!")
            print(f"Objective value: {self.m.objVal:.2f}")
        else:
            print("Failed to optimize!")

        end_time = time.time()
        print(f"\n<< Finished planning routine in {end_time - start_time:.2f}s\n")

    # Backward-compatible API names
    def start(self):
        self.run()

    def addMILPVariables(self):
        self.build_milp_variables()

    def addMILPConstraints(self):
        self.build_milp_constraints()

    def setMILPObjective(self):
        self.configure_objective()


if __name__ == '__main__':
    # For reproducibility
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=None, help="Path to graph file (GML)")
    parser.add_argument("--horizon", type=int, default=20, help="Planning horizon")
    parser.add_argument("--searchers", type=int, default=2, help="Number of searchers")
    parser.add_argument("--target-start", type=int, default=None, help="Target start vertex id")
    parser.add_argument(
        "--searcher-starts",
        type=int,
        nargs="+",
        default=None,
        help="Searcher start vertex ids (count must match --searchers)",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="uniform",
        choices=["uniform", "stationary"],
        help="Target motion model",
    )
    args = parser.parse_args()

    print("\nWelcome to the Multi-robot Efficient Search Path Planning solver!\n")
    solver = Mespp(
        graph_path=args.graph,
        horizon=args.horizon,
        num_searchers=args.searchers,
        target_start_vertex=args.target_start,
        searcher_starts=args.searcher_starts,
        motion=args.motion,
    )
    solver.start()
    print("\nThank you!\n")
