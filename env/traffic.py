"""Passenger arrival generator (owner: B).

Each second the number of new passengers is drawn from a Poisson
distribution. Each passenger gets an (origin, destination) pair
depending on the traffic pattern.
"""
import numpy as np

PATTERNS = ("up_peak", "down_peak", "mixed", "fluctuating")


class TrafficGenerator:
    def __init__(self, cfg, pattern="mixed", seed=None):
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}")
        self.n_floors = cfg["building"]["n_floors"]
        self.lobby = cfg["building"]["lobby_floor"]
        self.episode_length = cfg["simulation"]["episode_length"]
        self.base_rate = cfg["traffic"]["arrival_rate"]
        self.patterns = cfg["traffic"]["patterns"]
        self.pattern = pattern
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _active_pattern(self, t):
        if self.pattern != "fluctuating":
            return self.pattern
        phase = t / self.episode_length
        if phase < 1 / 3:
            return "up_peak"
        if phase < 2 / 3:
            return "mixed"
        return "down_peak"

    def _sample_od(self, mix):
        u = self.rng.random()
        others = [f for f in range(self.n_floors) if f != self.lobby]
        if u < mix["up"]:
            return self.lobby, int(self.rng.choice(others))
        if u < mix["up"] + mix["down"]:
            return int(self.rng.choice(others)), self.lobby
        origin = int(self.rng.integers(self.n_floors))
        dest = int(self.rng.integers(self.n_floors - 1))
        if dest >= origin:
            dest += 1
        return origin, dest

    def arrivals(self, t):
        """Return a list of (origin, destination) for passengers arriving at second t."""
        mix = self.patterns[self._active_pattern(t)]
        rate = self.base_rate * mix["rate_multiplier"]
        n = self.rng.poisson(rate)
        return [self._sample_od(mix) for _ in range(n)]