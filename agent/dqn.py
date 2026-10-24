"""DQN agent with action masking (owner: A).

Q-network: small MLP, state (27) -> one Q-value per elevator (4).
Extras used here:
- experience replay buffer (breaks correlation between samples)
- target network (stable training targets)
- Double DQN target (reduces over-estimation)
- action mask: failed elevators get Q = -inf, so they are never chosen
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, n_actions, seed=None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.next_masks = np.zeros((capacity, n_actions), np.float32)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        i = self.idx
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i], self.next_masks[i] = next_obs, done, next_mask
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        j = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.obs[j]), torch.from_numpy(self.actions[j]),
                torch.from_numpy(self.rewards[j]), torch.from_numpy(self.next_obs[j]),
                torch.from_numpy(self.dones[j]), torch.from_numpy(self.next_masks[j]))


class DQNAgent:
    name = "dqn"

    def __init__(self, obs_dim, n_actions, cfg_dqn, seed=0):
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.discount = cfg_dqn["discount"]
        self.batch_size = cfg_dqn["batch_size"]
        self.target_update = cfg_dqn["target_update_steps"]
        self.q = QNetwork(obs_dim, n_actions, cfg_dqn["hidden"])
        self.q_target = QNetwork(obs_dim, n_actions, cfg_dqn["hidden"])
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg_dqn["lr"])
        self.buffer = ReplayBuffer(cfg_dqn["buffer_size"], obs_dim, n_actions, seed)
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def reset(self, seed=None):
        """Called by evaluate.py at the start of each episode (nothing to reset)."""

    def act(self, obs, mask, eps=0.0):
        """Epsilon-greedy over the elevators allowed by the mask."""
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if self.rng.random() < eps:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            qv = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
        qv[np.asarray(mask) == 0] = -np.inf
        return int(np.argmax(qv))

    def update(self):
        """One gradient step on a random batch. Returns the loss."""
        if self.buffer.size < self.batch_size:
            return None
        obs, act, rew, nxt, done, nmask = self.buffer.sample(self.batch_size)
        q_sa = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online net picks the next action, target net scores it
            q_next_online = self.q(nxt).masked_fill(nmask == 0, -1e9)
            a_next = q_next_online.argmax(1, keepdim=True)
            q_next = self.q_target(nxt).gather(1, a_next).squeeze(1)
            target = rew + self.discount * (1.0 - done) * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location="cpu"))
        self.q_target.load_state_dict(self.q.state_dict())