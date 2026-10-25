"""Forced-failure experiment (owner: A).

One elevator (elevator 0) breaks down for 20 minutes in the middle of the
hour (t = 1200 s to 2400 s). Every method is run on the same 20 seeds with
and without this breakdown, and we measure how much the waiting time of
passengers arriving during the outage grows.

Writes results/forced_failure.csv and results/fig_forced_failure.png

Usage:
    python experiment_failure.py
    python experiment_failure.py --seeds 3     # quick check
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from env.gym_env import ElevatorEnv
from evaluate import make_baselines, load_dqn, choose

START, DURATION, ELEVATOR = 1200, 1200, 0
COLORS = {"dqn": "#2a78d6", "collective": "#eb6834",
          "nearest_car": "#1baf7a", "random": "#eda100"}
LABELS = {"dqn": "DQN", "collective": "Collective control",
          "nearest_car": "Nearest car", "random": "Random"}


class ForcedFailure:
    """Breaks one elevator at a fixed time and records the queue every second."""

    def __init__(self, broken=True):
        self.broken = broken
        self.queue = []

    def step(self, sim):
        if self.broken and sim.t == START:
            sim.fail_elevator(ELEVATOR, DURATION)
        self.queue.append(sim.n_waiting())


def run(cfg, pattern, policy, seed, broken):
    env = ElevatorEnv(cfg, pattern=pattern, failures=False)
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    tracker = ForcedFailure(broken)
    env.sim.failures = tracker                  # called by the simulator every second
    done = False
    while not done:
        obs, _, done, _, info = env.step(choose(policy, env, obs, info))
    sim = env.sim
    boarded = sim.finished + [p for e in sim.elevators for p in e.passengers]
    waits = [p.t_board - p.t_arrive for p in boarded]
    window = [p.t_board - p.t_arrive for p in boarded
              if START <= p.t_arrive < START + DURATION]
    queue = np.zeros(cfg["simulation"]["episode_length"])
    queue[-len(tracker.queue):] = tracker.queue
    return np.mean(waits), np.mean(window), np.percentile(window, 95), queue


def main(seeds, patterns, out_csv, out_fig):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    methods = make_baselines()
    methods["dqn"] = load_dqn(cfg, "models/dqn_mixed.pt")
    rows, queues = [], {}
    for pattern in patterns:
        for name, policy in methods.items():
            res = {}
            for broken in (False, True):
                runs = [run(cfg, pattern, policy, s, broken) for s in seeds]
                res[broken] = [np.mean([r[i] for r in runs]) for i in range(3)]
                queues[(pattern, name, broken)] = np.mean([r[3] for r in runs], axis=0)
            base, hit = res[False], res[True]
            row = {"pattern": pattern, "method": name,
                   "window_wait_normal": base[1], "window_wait_failure": hit[1],
                   "increase_s": hit[1] - base[1],
                   "increase_pct": 100 * (hit[1] - base[1]) / base[1],
                   "p95_window_normal": base[2], "p95_window_failure": hit[2],
                   "hour_wait_normal": base[0], "hour_wait_failure": hit[0]}
            rows.append(row)
            print(f"{pattern:8s} {name:12s} wait during outage: {base[1]:5.1f}s -> {hit[1]:5.1f}s "
                  f"(+{row['increase_s']:4.1f}s, +{row['increase_pct']:3.0f}%)  "
                  f"p95: {base[2]:5.1f}s -> {hit[2]:5.1f}s")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Figure: passengers waiting over time, with the outage shaded
    fig, axes = plt.subplots(1, len(patterns), figsize=(7.0, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    minutes = np.arange(cfg["simulation"]["episode_length"]) / 60
    for ax, pattern in zip(axes, patterns):
        ax.axvspan(START / 60, (START + DURATION) / 60, color="#e4e3df", zorder=0)
        for name in methods:
            q = np.convolve(queues[(pattern, name, True)], np.ones(60) / 60, mode="same")
            ax.plot(minutes, q, color=COLORS[name], linewidth=1.6, label=LABELS[name])
        ax.set_title(f"{pattern.replace('_', '-')} traffic, elevator 0 down 20-40 min", fontsize=9)
        ax.set_xlabel("Minute")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Passengers waiting")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    print(f"saved {out_csv} and {out_fig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    main(range(args.seeds), ["mixed", "up_peak"],
         "results/forced_failure.csv", "results/fig_forced_failure.png")