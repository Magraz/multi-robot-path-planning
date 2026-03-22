#!/usr/bin/env python3

import os
import sys

import cairo
import igraph as ig
import numpy as np
from gurobipy import GRB


class Helper:
  """Shared MILP construction and plotting utilities."""

  @staticmethod
  def _scalar(value):
    """Extract scalar numeric value from Gurobi arrays/scalars."""
    if isinstance(value, np.ndarray):
      return value.item() if value.size == 1 else value[0]
    return value

  def define_transition_variables(self):
    """Define presence (`x`) and transition (`y`) decision variables."""
    for t in range(self.HORIZON + 1):
      # legal_V[t][s] = vertices searcher s can reach at time t.
      self.legal_V[t] = self.g.neighborhood(self.searchers.initial_positions, order=t)
      self.presence[t] = {}
      self.transition[t] = {}

      for s in range(self.searchers.M):
        self.presence[t][s] = {}
        self.transition[t][s] = {}

        for v in self.legal_V[t][s]:
          self.presence[t][s][v] = self.m.addMVar((1,), vtype=GRB.BINARY, name=f"x_{v}^{s},{t}")

        for u in self.legal_V[t][s]:
          self.transition[t][s][u] = {}
          if t < self.HORIZON:
            # Normal transition edges for times 0..HORIZON-1.
            for v in self.g.neighborhood(u, order=1):
              self.transition[t][s][u][v] = self.m.addMVar(
                (1,), vtype=GRB.BINARY, name=f"y_{u},{v}^{s},{t}"
              )
          elif t == self.HORIZON:
            # Terminal transition to goal sink representation used by model.
            self.transition[t][s][u] = self.m.addMVar(
              (1,), vtype=GRB.BINARY, name=f"y_{u},vg^{s},tau"
            )
          else:
            raise RuntimeError("Invalid time index while creating transition variables")

  def define_belief_variables(self):
    """Define belief (`beta`), propagated belief (`alpha`) and capture (`psi`) variables."""
    self.beliefs = self.m.addMVar(
      (self.N + 1, self.HORIZON + 1),
      lb=0,
      ub=1,
      vtype=GRB.CONTINUOUS,
      name="beta",
    )
    self.prop_beliefs = self.m.addMVar(
      (self.N, self.HORIZON + 1),
      lb=0,
      ub=1,
      vtype=GRB.CONTINUOUS,
      name="alpha",
    )
    self.capture = self.m.addMVar((self.N, self.HORIZON + 1), vtype=GRB.BINARY, name="psi")

  def define_transition_constraints(self):
    """Add transition flow constraints for searchers."""
    for s in range(self.searchers.M):
      # Initial condition at t=0.
      start_vertex = self.legal_V[0][s][0]
      self.m.addConstr(self.presence[0][s][start_vertex] == 1)
      self.m.addConstr(
        sum(self.transition[0][s][start_vertex][j] for j in self.g.neighborhood(start_vertex, order=1)) == 1
      )
      self.m.addConstr(
        # Terminal flow condition.
        sum(self.transition[self.HORIZON][s][j] for j in self.legal_V[self.HORIZON][s]) == 1
      )

    for t in range(1, self.HORIZON + 1):
      for s in range(self.searchers.M):
        self.m.addConstrs(
          (
            sum(
              self.transition[t - 1][s][j][v]
              for j in list(
                set(self.g.neighborhood(v, order=1))
                & set(self.legal_V[t - 1][s])
              )
            )
            == self.presence[t][s][v]
            for v in self.legal_V[t][s]
          )
        )

        if t < self.HORIZON:
          self.m.addConstrs(
            (
              sum(self.transition[t][s][v][i] for i in self.g.neighborhood(v, order=1))
              == self.presence[t][s][v]
              for v in self.legal_V[t][s]
            )
          )
        elif t == self.HORIZON:
          self.m.addConstrs(
            (self.transition[t][s][v] == self.presence[t][s][v] for v in self.legal_V[t][s])
          )
        else:
          raise RuntimeError("Invalid time index while creating transition constraints")

  def define_belief_constraints(self):
    """Add belief propagation and capture coupling constraints."""
    # Initial belief: all mass at target start (no capture at t=0).
    self.m.addConstr(self.beliefs[:, 0] == self.searchers.initial_belief)

    for t in range(1, self.HORIZON + 1):
      # Propagate occupancy belief through the target motion model.
      self.m.addConstr(self.beliefs[1:, t - 1] @ self.target.motion_model == self.prop_beliefs[:, t])
      # Standard linearization constraints for belief update with capture.
      self.m.addConstr(self.beliefs[1:, t] <= (np.ones(self.N) - self.capture[:, t]))
      self.m.addConstr(self.beliefs[1:, t] <= self.prop_beliefs[:, t])
      self.m.addConstr(self.beliefs[1:, t] >= (self.prop_beliefs[:, t] - self.capture[:, t]))

    for t in range(self.HORIZON + 1):
      for v in range(self.N):
        valid_searchers = [s for s in range(self.searchers.M) if v in self.legal_V[t][s]]
        if valid_searchers:
          self.m.addConstr(
            sum(self.presence[t][s][v] for s in valid_searchers)
            <= self.searchers.M * self.capture[v, t]
          )
          self.m.addConstr(
            self.capture[v, t] <= sum(self.presence[t][s][v] for s in valid_searchers)
          )
        else:
          self.m.addConstr(self.capture[v, t] == 0)

    self.m.addConstrs(
      (self.beliefs[0, t] == (1 - np.ones(self.N) @ self.beliefs[1:, t]) for t in range(self.HORIZON + 1))
    )

  def has_captured(self):
    """Return whether target is currently colocated with any searcher."""
    return self.target.position in self.searchers.positions

  def render_frame(self, t):
    """Render and save visualization for time step `t`."""
    layout = self.g["plot_layout"] if "plot_layout" in self.g.attributes() else self.g.layout("grid")

    # Shared style for both subplots.
    visual_style = {
      "vertex_size": 20,
      "layout": layout,
      "vertex_label": range(self.g.vcount()),
    }

    self.g.vs["color"] = "white"
    self.g.vs[self.target.position]["color"] = "red"
    for idx in range(self.N):
      # Highlight potential capture vertices.
      capture_val = self._scalar(self.capture[idx, t].X)
      if capture_val:
        self.g.vs[idx]["color"] = "yellow"
        if idx == self.target.position:
          self.g.vs[idx]["color"] = "#24fc03"

    self.belief_g.vs["color"] = "white"

    capture_belief = abs(round(self._scalar(self.beliefs[0, t].X), 2))

    belief_array = np.array([round(self._scalar(b.X), 2) for b in self.beliefs[1:, t]])
    max_belief = np.max(belief_array)
    for idx in range(self.N):
      # Stronger red means larger target belief at that vertex.
      if belief_array[idx] > 0:
        alpha = int(255 * belief_array[idx] / max_belief)
        self.belief_g.vs[idx]["color"] = f"#ff0000{alpha:0>2x}"

    width = 700
    graph_width = 560
    left_margin = 70
    top_margin = 100
    output_path = os.path.join(sys.path[0], f"../results/path_t={t}.png")
    plot = ig.Plot(output_path, bbox=(2 * width, width), background="white")

    plot.add(
      self.g,
      bbox=(left_margin, top_margin, left_margin + graph_width, top_margin + graph_width),
      **visual_style,
    )
    plot.add(
      self.belief_g,
      bbox=(width + left_margin, top_margin, width + left_margin + graph_width, top_margin + graph_width),
      **visual_style,
    )

    plot.redraw()
    ctx = cairo.Context(plot.surface)
    ctx.set_font_size(20)
    ctx.set_source_rgb(0, 0, 0)

    actual_capture = False
    for s in range(self.searchers.M):
      pres = self.presence[t][s].get(self.target.position)
      if pres is not None and self._scalar(pres.X) > 0.5:
        # If solver places a searcher on target, show full capture in title.
        actual_capture = True
        break

    capture_belief_display = 1.0 if actual_capture else capture_belief
    ctx.move_to(30, 40)
    ctx.show_text(f"Positions and Beliefs at t={t}")
    ctx.move_to(30, 70)
    ctx.show_text(f"Capture belief: {capture_belief_display}")

    ctx.set_font_size(18)
    ctx.move_to(left_margin, 90)
    ctx.show_text("Robots and Target Movement")
    ctx.move_to(width + left_margin, 90)
    ctx.show_text("Central Planner Belief Distribution for Target")

    plot.save()

  # Backward-compatible API (existing callers still work)
  def addTransitionVariables(self):
    self.define_transition_variables()

  def addBeliefVariables(self):
    self.define_belief_variables()

  def addTransitionConstraints(self):
    self.define_transition_constraints()

  def addBeliefConstraints(self):
    self.define_belief_constraints()

  def plot(self, t):
    self.render_frame(t)