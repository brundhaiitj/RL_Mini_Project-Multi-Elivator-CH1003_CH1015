import yaml
import numpy as np
from env.elevator_env import ElevatorSim
from env.traffic import TrafficGenerator
from baselines.nearest_car import NearestCar
from baselines.random_agent import RandomAgent

cfg = yaml.safe_load(open("configs/default.yaml"))


def run(policy, pattern="mixed", seed=0):
    traffic = TrafficGenerator(cfg, pattern, seed=seed)
    sim = ElevatorSim(cfg, traffic)
    policy.reset(seed)
    for _ in range(cfg["simulation"]["episode_length"]):
        sim.tick()
        for key in list(sim.unassigned):
            idx = policy.select(sim, key)
            if idx is not None:
                sim.assign(key, idx)
    assert len(sim.finished) > 0, "no passenger was delivered"
    waits = [p.t_board - p.t_arrive for p in sim.finished]
    energy = sum(e.floors_moved + 0.5 * e.starts for e in sim.elevators)
    return len(sim.finished), float(np.mean(waits)), max(waits), energy, sim.n_waiting()


for policy in [RandomAgent(0), NearestCar()]:
    delivered, avg_wait, max_wait, energy, left = run(policy)
    print(f"{policy.name:12s} delivered={delivered} avg_wait={avg_wait:.1f}s "
          f"max_wait={max_wait}s energy={energy:.0f} still_waiting={left}")

print("baseline tests finished")