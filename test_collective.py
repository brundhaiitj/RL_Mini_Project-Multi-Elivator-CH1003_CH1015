import yaml
import numpy as np
from env.elevator_env import ElevatorSim
from env.traffic import TrafficGenerator
from baselines.random_agent import RandomAgent
from baselines.nearest_car import NearestCar
from baselines.collective import CollectiveControl

cfg = yaml.safe_load(open("configs/default.yaml"))

# Part 1: direction logic
sim = ElevatorSim(cfg)
e0, e1 = sim.elevators[0], sim.elevators[1]
e0.floor, e0.direction, e0.stops = 3, 1, {7}    # moving up, going to floor 7
e1.floor = 0                                     # idle at the lobby
sim.elevators[2].available = False
sim.elevators[3].available = False
cc, nc = CollectiveControl(), NearestCar()
print("call at 5 going UP:   collective ->", cc.select(sim, (5, 1)), " nearest ->", nc.select(sim, (5, 1)))
print("call at 5 going DOWN: collective ->", cc.select(sim, (5, -1)), " nearest ->", nc.select(sim, (5, -1)))
assert cc.select(sim, (5, 1)) == 0      # on its way up: pick up en route
assert cc.select(sim, (5, -1)) == 1     # would have to go to 7 and come back
assert nc.select(sim, (5, -1)) == 0     # nearest-car ignores direction

# Part 2: full hour of traffic, 3 seeds, 2 patterns
def run(policy, pattern, seed):
    sim = ElevatorSim(cfg, TrafficGenerator(cfg, pattern, seed=seed))
    policy.reset(seed)
    for _ in range(cfg["simulation"]["episode_length"]):
        sim.tick()
        for key in list(sim.unassigned):
            idx = policy.select(sim, key)
            if idx is not None:
                sim.assign(key, idx)
    waits = [p.t_board - p.t_arrive for p in sim.finished]
    energy = sum(e.floors_moved + 0.5 * e.starts for e in sim.elevators)
    return np.mean(waits), max(waits), energy

for pattern in ["mixed", "up_peak"]:
    for policy in [RandomAgent(0), NearestCar(), CollectiveControl()]:
        r = np.array([run(policy, pattern, s) for s in range(3)]).mean(axis=0)
        print(f"{pattern:8s} {policy.name:12s} avg_wait={r[0]:5.1f}s max_wait={r[1]:5.1f}s energy={r[2]:5.0f}")

print("collective tests passed")