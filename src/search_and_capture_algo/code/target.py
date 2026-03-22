#!/usr/bin/env python3

import random
import numpy as np


class Target:
    """Target process with either stationary or uniform random walk dynamics."""

    def __init__(self, g, N=100, initial_position=45, motion="uniform"):
        self.N = N
        self.initial_position = initial_position
        self.position = self.initial_position
        self.g = g
        # motion_model[i, j] = P(target moves from i to j)
        self.motion_model = np.zeros((N, N))

        if motion == "stationary":
            # Target stays at its current vertex.
            self.motion_model = np.eye(N)
        elif motion == "uniform":
            # Target moves uniformly over its 1-hop neighborhood.
            neighborhood_list = g.neighborhood(np.arange(N), order=1)
            for v in range(N):
                neighbors = neighborhood_list[v]
                self.motion_model[v, neighbors] = 1.0 / len(neighbors)
        else:
            print("Exception! Motion model not recognized, assuming stationary")
            self.motion_model = np.eye(N)

    def updateTargetPosition(self):
        """Sample next target position uniformly from current neighborhood."""
        # Used for simulation/plotting after planning.
        neighborhood = self.g.neighborhood(self.position, order=1)
        self.position = random.choice(neighborhood)
