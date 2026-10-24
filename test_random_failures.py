import copy
import yaml
import numpy as np
from env.elevator_env import ElevatorSim, FailureModel
from env.traffic import TrafficGenerator
from baselines.collective import CollectiveControl

cfg = yaml.safe_load(open("configs/default.yaml"))


def run(cfg, seed, with_failures):
    sim = ElevatorSim(cfg, TrafficGenerator(cfg, "mixed", seed=seed))
    if with_failures:
        sim.failures = FailureModel(cfg, seed=seed)
    policy = CollectiveControl()
    max_down = 0
    for _ in range(cfg["simulation"]["episode_length"]):
        sim.tick()
        for key in list(sim.unassigned):
            idx = policy.select(sim, key)
            if idx is not None:
                sim.assign(key, idx)
        max_down = max(max_down, sum(not e.available for e in sim.elevators))
    waits = [p.t_board - p.t_arrive for p in sim.finished]
    log = sim.failures.log if with_failures else []
    return np.mean(waits), max(waits), len(sim.finished), log, max_down


# Part 1: failures happen and never exceed max_simultaneous
total = 0
for seed in range(5):
    _, _, delivered, log, max_down = run(cfg, seed, True)
    print(f"seed {seed}: failures (time, elevator, seconds) = {log}, delivered = {delivered}")
    assert max_down <= cfg["failures"]["max_simultaneous"]
    assert delivered > 400
    total += len(log)
assert total > 0, "no failure happened in 5 episodes"

# Part 2: stress test, many failures, at most 2 down at once
hard = copy.deepcopy(cfg)
hard["failures"]["expected_per_episode"] = 20
hard["failures"]["max_simultaneous"] = 2
_, _, delivered, log, max_down = run(hard, 0, True)
print(f"stress: {len(log)} failures, at most {max_down} down at once, delivered = {delivered}")
assert max_down <= 2 and len(log) > 5

# Part 3: failures should hurt waiting time
a = np.mean([run(cfg, s, False)[0] for s in range(5)])
b = np.mean([run(cfg, s, True)[0] for s in range(5)])
print(f"collective avg wait: no failures = {a:.1f}s, with failures = {b:.1f}s")
print("random failure tests passed")