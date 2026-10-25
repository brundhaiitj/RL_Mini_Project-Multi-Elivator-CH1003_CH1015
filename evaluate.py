"""Evaluate dispatch methods on fixed seeds (owner: B).

Every method runs through the same Gymnasium environment, on the same
traffic patterns and seeds, with and without failures, so the numbers
are directly comparable. Metrics follow METRICS.md.

Usage:
    python evaluate.py                    # baselines + trained DQN, 20 seeds
    python evaluate.py --seeds 3          # quick check
    python evaluate.py --no-dqn --out results/baselines.csv   # baselines only
"""
import argparse
import csv
import os
import time

import numpy as np
import yaml

from env.gym_env import ElevatorEnv
from baselines.random_agent import RandomAgent
from baselines.nearest_car import NearestCar
from baselines.collective import CollectiveControl
from agent.dqn import DQNAgent

PATTERNS = ["up_peak", "down_peak", "mixed", "fluctuating"]
METRICS = ["return", "avg_wait", "p95_wait", "max_wait", "pct_over_60",
           "avg_trip", "energy", "delivered"]


def make_baselines():
    return {"random": RandomAgent(), "nearest_car": NearestCar(),
            "collective": CollectiveControl()}


def load_dqn(cfg, path):
    n = cfg["building"]["n_elevators"]
    agent = DQNAgent(6 * n + 3, n, cfg["dqn"])
    agent.load(path)
    return agent


def choose(policy, env, obs, info):
    """Baselines look at the simulator; the DQN agent uses obs + action mask."""
    if hasattr(policy, "act"):
        return policy.act(obs, info["action_mask"])
    a = policy.select(env.sim, env.key)
    return 0 if a is None else a


def episode_metrics(env, total_return):
    sim, t = env.sim, env.sim.t
    thr = env.cfg["reward"]["unfair_threshold"]
    boarded = [p.t_board - p.t_arrive for p in sim.finished]
    riding = [p.t_board - p.t_arrive for e in sim.elevators for p in e.passengers]
    waiting = [t - p.t_arrive for q in sim.waiting.values() for p in q]
    waits = boarded + riding
    everyone = waits + waiting                    # unfinished counted with wait so far
    trips = [p.t_done - p.t_arrive for p in sim.finished]
    return {
        "return": total_return,
        "avg_wait": float(np.mean(waits)) if waits else 0.0,
        "p95_wait": float(np.percentile(waits, 95)) if waits else 0.0,
        "max_wait": float(max(everyone)) if everyone else 0.0,
        "pct_over_60": 100.0 * np.mean([w > thr for w in everyone]) if everyone else 0.0,
        "avg_trip": float(np.mean(trips)) if trips else 0.0,
        "energy": float(sum(e.floors_moved + 0.5 * e.starts for e in sim.elevators)),
        "delivered": len(sim.finished),
    }


def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    total, done = 0.0, False
    while not done:
        obs, r, done, _, info = env.step(choose(policy, env, obs, info))
        total += r
    return episode_metrics(env, total)


def evaluate(cfg, methods, seeds, patterns=PATTERNS):
    rows = []
    for pattern in patterns:
        for failures in (False, True):
            env = ElevatorEnv(cfg, pattern=pattern, failures=failures)
            for name, policy in methods.items():
                runs = [run_episode(env, policy, s) for s in seeds]
                row = {"method": name, "pattern": pattern, "failures": failures}
                for m in METRICS:
                    vals = [r[m] for r in runs]
                    row[m] = float(np.mean(vals))
                    row[m + "_std"] = float(np.std(vals))
                rows.append(row)
                print(f"{pattern:11s} failures={str(failures):5s} {name:12s} "
                      f"avg_wait={row['avg_wait']:5.1f}s p95={row['p95_wait']:5.1f}s "
                      f"max={row['max_wait']:5.1f}s energy={row['energy']:5.0f} "
                      f"return={row['return']:7.1f}")
    return rows


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, default=20, help="evaluation seeds 0..N-1")
    ap.add_argument("--dqn", default="models/dqn_mixed.pt", help="trained DQN model")
    ap.add_argument("--no-dqn", action="store_true", help="evaluate baselines only")
    ap.add_argument("--out", default="results/all_methods.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    methods = make_baselines()
    if not args.no_dqn:
        methods["dqn"] = load_dqn(cfg, args.dqn)
    start = time.time()
    rows = evaluate(cfg, methods, range(args.seeds))
    save_csv(rows, args.out)
    print(f"saved {len(rows)} rows to {args.out} in {time.time() - start:.0f}s")