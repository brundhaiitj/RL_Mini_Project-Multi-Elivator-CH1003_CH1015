"""Random baseline (owner: B).

Assigns each hall call to a random working elevator.
Gives the lower bound for the comparison.
"""
import numpy as np


class RandomAgent:
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def select(self, sim, key):
        cands = [e.idx for e in sim.elevators if e.available and e.load < e.capacity]
        if not cands:
            cands = [e.idx for e in sim.elevators if e.available]
        if not cands:
            return None
        return int(self.rng.choice(cands))