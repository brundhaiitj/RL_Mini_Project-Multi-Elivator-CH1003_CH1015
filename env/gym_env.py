"""Gymnasium wrapper around the elevator simulator (owner: A).

One step = one dispatch decision. The agent sees the building state
for the newest unassigned hall call and picks which elevator serves it.
The simulator then runs second by second until the next call needs a
decision, and the reward covers everything that happened in between.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.elevator_env import ElevatorSim, FailureModel
from env.traffic import TrafficGenerator


class ElevatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, pattern="mixed", failures=True):
        super().__init__()
        self.cfg = cfg
        self.pattern = pattern
        self.use_failures = failures
        self.n_floors = cfg["building"]["n_floors"]
        self.n_elev = cfg["building"]["n_elevators"]
        self.capacity = cfg["building"]["capacity"]
        self.T = cfg["simulation"]["episode_length"]
        r = cfg["reward"]
        self.alpha, self.beta, self.gamma = r["alpha"], r["beta"], r["gamma"]
        self.thr = r["unfair_threshold"]
        self.e_floor = r["energy_per_floor"]
        self.e_start = r["energy_per_start_stop"]
        self.obs_dim = 6 * self.n_elev + 3
        self.observation_space = spaces.Box(-1.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(self.n_elev)
        self.sim = None
        self.key = None

    # ---------- helpers ----------
    def action_mask(self):
        return np.array([e.available for e in self.sim.elevators], dtype=np.int8)

    def _obs(self):
        f = self.n_floors - 1
        call_floor, call_dir = self.key if self.key is not None else (0, 0)
        x = []
        for e in self.sim.elevators:
            x += [e.floor / f,
                  float(e.direction),
                  e.load / self.capacity,
                  float(e.available),
                  min(len(e.stops) / 10, 1.0),
                  abs(call_floor - e.floor) / f]
        x += [call_floor / f, float(call_dir), min(self.sim.n_waiting() / 50, 1.0)]
        return np.array(x, dtype=np.float32)

    def _energy(self):
        return sum(e.floors_moved * self.e_floor + e.starts * self.e_start
                   for e in self.sim.elevators)

    def _unfair_now(self):
        t = self.sim.t
        return sum(1 for q in self.sim.waiting.values() for p in q
                   if t - p.t_arrive > self.thr)

    def _advance(self):
        """Run the simulator until a call needs a decision or the episode ends.
        Returns the unfair passenger-seconds accumulated meanwhile."""
        unfair = 0
        while not self.sim.unassigned and self.sim.t < self.T:
            self.sim.tick()
            unfair += self._unfair_now()
        self.key = self.sim.unassigned[0] if self.sim.unassigned else None
        return unfair

    def _info(self):
        waits = [p.t_board - p.t_arrive for p in self.sim.finished]
        return {"action_mask": self.action_mask(),
                "avg_wait": float(np.mean(waits)) if waits else 0.0,
                "max_wait": float(max(waits)) if waits else 0.0,
                "energy": float(self._energy()),
                "delivered": len(self.sim.finished)}

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = 0 if seed is None else int(seed)
        traffic = TrafficGenerator(self.cfg, self.pattern, seed=s)
        self.sim = ElevatorSim(self.cfg, traffic)
        if self.use_failures:
            self.sim.failures = FailureModel(self.cfg, seed=s + 100000)
        self._advance()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        mask = self.action_mask()
        if not mask[action]:                      # failed elevator chosen
            valid = np.flatnonzero(mask)
            action = int(valid[0]) if len(valid) else None
        if action is not None:
            self.sim.assign(self.key, action)
        else:                                     # every elevator is down
            self.sim.unassigned.append(self.sim.unassigned.pop(0))

        w0, e0 = self.sim.wait_accum, self._energy()
        unfair = self._advance()
        wait = self.sim.wait_accum - w0
        energy = self._energy() - e0
        reward = -(self.alpha * wait + self.beta * unfair + self.gamma * energy)

        terminated = self.sim.t >= self.T
        info = self._info()
        info["reward_terms"] = {"wait": wait, "unfair": unfair, "energy": energy}
        return self._obs(), float(reward), terminated, False, info