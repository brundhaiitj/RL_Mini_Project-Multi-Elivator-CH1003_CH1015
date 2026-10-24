import os
import yaml
import numpy as np
from agent.dqn import DQNAgent

cfg = yaml.safe_load(open("configs/default.yaml"))
agent = DQNAgent(27, 4, cfg["dqn"], seed=0)

# Part 1: masking - a failed elevator is never chosen
obs = np.zeros(27, np.float32)
mask = np.array([1, 0, 1, 1])
picks = {agent.act(obs, mask, eps=e) for e in (0.0, 1.0) for _ in range(200)}
print("elevators chosen with elevator 1 failed:", sorted(picks))
assert 1 not in picks

# Part 2: the network can learn - elevator 2 always gives reward 0, others -1
rng = np.random.default_rng(0)
for _ in range(2000):
    s = rng.random(27).astype(np.float32)
    a = int(rng.integers(4))
    r = 0.0 if a == 2 else -1.0
    agent.buffer.add(s, a, r, s, 1.0, np.ones(4))
losses = [agent.update() for _ in range(500)]
print(f"loss: first = {losses[0]:.3f}, last = {losses[-1]:.4f}")
choice = agent.act(rng.random(27).astype(np.float32), np.ones(4))
print("greedy choice after learning:", choice)
assert choice == 2 and losses[-1] < losses[0]

# Part 3: save and load give identical decisions
os.makedirs("models", exist_ok=True)
agent.save("models/test_dqn.pt")
agent2 = DQNAgent(27, 4, cfg["dqn"], seed=1)
agent2.load("models/test_dqn.pt")
x = rng.random(27).astype(np.float32)
assert agent.act(x, np.ones(4)) == agent2.act(x, np.ones(4))
os.remove("models/test_dqn.pt")
print("dqn tests passed")