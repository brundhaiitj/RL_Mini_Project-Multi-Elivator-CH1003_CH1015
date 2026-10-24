import yaml
import numpy as np
from gymnasium.utils.env_checker import check_env
from env.gym_env import ElevatorEnv
from baselines.random_agent import RandomAgent
from baselines.nearest_car import NearestCar
from baselines.collective import CollectiveControl

cfg = yaml.safe_load(open("configs/default.yaml"))
env = ElevatorEnv(cfg, pattern="mixed", failures=True)

# Part 1: shapes and the official Gymnasium checker
obs, info = env.reset(seed=0)
print("observation shape:", obs.shape, " actions:", env.action_space.n)
print("first call:", env.key, " mask:", info["action_mask"])
assert obs.shape == (27,) and env.observation_space.contains(obs)
check_env(env.unwrapped, skip_render_check=True)
print("gymnasium check_env passed")

# Part 2: run one full episode per policy through the env
def run(policy, seed):
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    total, steps, done = 0.0, 0, False
    while not done:
        a = policy.select(env.sim, env.key)
        obs, r, done, _, info = env.step(a if a is not None else 0)
        assert env.observation_space.contains(obs)
        total += r
        steps += 1
    return total, steps, info

for policy in [RandomAgent(0), NearestCar(), CollectiveControl()]:
    res = [run(policy, s) for s in range(3)]
    ret = np.mean([r[0] for r in res])
    steps = np.mean([r[1] for r in res])
    wait = np.mean([r[2]["avg_wait"] for r in res])
    energy = np.mean([r[2]["energy"] for r in res])
    print(f"{policy.name:12s} decisions={steps:5.0f} return={ret:8.1f} "
          f"avg_wait={wait:5.1f}s energy={energy:5.0f}")

print("gym env tests passed")