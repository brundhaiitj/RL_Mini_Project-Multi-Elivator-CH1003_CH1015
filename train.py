"""Train the DQN dispatcher (owner: A).

Usage:
    python train.py                         # settings from configs/default.yaml
    python train.py --episodes 20           # quick test
"""
import argparse
import csv
import os
import time

import numpy as np
import yaml

from agent.dqn import DQNAgent
from env.gym_env import ElevatorEnv


def train(cfg, pattern, episodes, out_model, out_log, seed):
    d = cfg["dqn"]
    env = ElevatorEnv(cfg, pattern=pattern, failures=True)
    agent = DQNAgent(env.obs_dim, env.n_elev, d, seed=seed)
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_log) or ".", exist_ok=True)
    log = open(out_log, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["episode", "return", "avg_wait", "max_wait", "energy", "epsilon", "loss"])

    step, best, start = 0, -np.inf, time.time()
    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + seed * 100000 + ep)   # training seeds >= 1000
        total, losses, done = 0.0, [], False
        while not done:
            frac = min(step / d["eps_decay_steps"], 1.0)
            eps = d["eps_start"] + frac * (d["eps_end"] - d["eps_start"])
            mask = info["action_mask"]
            action = agent.act(obs, mask, eps)
            next_obs, reward, done, _, info = env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, float(done), info["action_mask"])
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs, total, step = next_obs, total + reward, step + 1
        mean_loss = float(np.mean(losses)) if losses else 0.0
        writer.writerow([ep, round(total, 3), round(info["avg_wait"], 3), info["max_wait"],
                         info["energy"], round(eps, 3), round(mean_loss, 5)])
        log.flush()
        if total > best:
            best = total
            agent.save(out_model.replace(".pt", "_best.pt"))
        if ep % 10 == 0 or ep == episodes - 1:
            print(f"ep {ep:4d}  return {total:7.1f}  avg_wait {info['avg_wait']:5.1f}s  "
                  f"eps {eps:.2f}  loss {mean_loss:.4f}  {time.time() - start:5.0f}s")
    log.close()
    agent.save(out_model)
    print(f"final model saved to {out_model} (best training return {best:.1f})")
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pattern", default="mixed")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/dqn_mixed.pt")
    ap.add_argument("--log", default="results/train_log_mixed.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    episodes = args.episodes or cfg["dqn"]["train_episodes"]
    train(cfg, args.pattern, episodes, args.out, args.log, args.seed)