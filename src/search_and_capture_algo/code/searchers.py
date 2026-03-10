#!/usr/bin/env python3

import numpy as np


class Searchers:
    """Searcher team state container used by the planner."""

    def __init__(
        self,
        g,
        N=100,
        M=2,
        initial_positions=np.array([90, 58]),
        target_initial_position=45,
    ):
        self.N = N
        self.M = M
        # Initial and current positions of all searchers.
        self.initial_positions = initial_positions
        self.positions = self.initial_positions.copy()

        # Belief vector has N+1 entries: index 0 is capture state,
        # indices 1..N map to graph vertices 0..N-1.
        self.initial_belief = np.zeros(N + 1)
        self.initial_belief[target_initial_position + 1] = 1

    def updatePositions(self):
        # Kept as placeholder (positions are optimized in the MILP model).
        pass