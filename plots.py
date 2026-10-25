"""Make the report figures from the saved results (owner: B).

Reads  results/train_log_mixed.csv and results/all_methods.csv
Writes results/fig_learning_curve.png
       results/fig_comparison.png
       results/fig_tradeoff.png

Usage:
    python plots.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed colour per method (colour-blind-checked palette); DQN is slot 1.
COLORS = {"dqn": "#2a78d6", "collective": "#eb6834",
          "nearest_car": "#1baf7a", "random": "#eda100"}
LABELS = {"dqn": "DQN", "collective": "Collective control",
          "nearest_car": "Nearest car", "random": "Random"}
METHODS = ["random", "nearest_car", "collective", "dqn"]
PATTERNS = ["up_peak", "down_peak", "mixed", "fluctuating"]
PATTERN_LABELS = ["Up", "Down", "Mixed", "Fluct."]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rolling(x, w=20):
    x = np.asarray(x, float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def results_lookup(rows):
    return {(r["method"], r["pattern"], r["failures"] == "True"): r for r in rows}


def learning_curve(log, res, out):
    ep = [int(r["episode"]) for r in log]
    wait = [float(r["avg_wait"]) for r in log]
    ret = [float(r["return"]) for r in log]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    a1.plot(ep, ret, color=COLORS["dqn"], alpha=0.25, linewidth=0.8)
    a1.plot(ep, rolling(ret), color=COLORS["dqn"], linewidth=2, label="DQN (20-episode mean)")
    a1.set(title="(a) Episode return", xlabel="Training episode", ylabel="Return (higher is better)")
    a1.legend(loc="lower right", fontsize=8)

    a2.plot(ep, wait, color=COLORS["dqn"], alpha=0.25, linewidth=0.8)
    a2.plot(ep, rolling(wait), color=COLORS["dqn"], linewidth=2, label="DQN")
    for m, style in (("nearest_car", (0, (4, 3))), ("collective", (0, (1, 2)))):
        y = float(res[(m, "mixed", True)]["avg_wait"])
        a2.axhline(y, color=COLORS[m], linewidth=1.5, linestyle=style, label=LABELS[m])
    a2.set(title="(b) Average waiting time", xlabel="Training episode", ylabel="Seconds")
    a2.set_ylim(bottom=0)
    a2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def comparison(res, out, failures=True):
    metrics = [("avg_wait", "(a) Average wait (s)"),
               ("p95_wait", "(b) 95th-percentile wait (s)"),
               ("energy", "(c) Energy")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    x = np.arange(len(PATTERNS))
    w = 0.2
    for ax, (m, title) in zip(axes, metrics):
        for i, meth in enumerate(METHODS):
            vals = [float(res[(meth, p, failures)][m]) for p in PATTERNS]
            errs = [float(res[(meth, p, failures)][m + "_std"]) for p in PATTERNS]
            ax.bar(x + (i - 1.5) * w, vals, width=w * 0.9, color=COLORS[meth],
                   yerr=errs, error_kw={"elinewidth": 0.6, "ecolor": MUTED, "capsize": 0},
                   label=LABELS[meth])
        ax.set_xticks(x, PATTERN_LABELS, fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)


def tradeoff(res, out, pattern="mixed"):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for meth in METHODS:
        for failures, marker in ((False, "o"), (True, "s")):
            r = res[(meth, pattern, failures)]
            ax.scatter(float(r["energy"]), float(r["avg_wait"]), s=36, marker=marker,
                       color=COLORS[meth], edgecolor="white", linewidth=1.5, zorder=3)
        r = res[(meth, pattern, True)]
        left = meth == "random"                  # rightmost point: label on its left
        ax.annotate(LABELS[meth], (float(r["energy"]), float(r["avg_wait"])),
                    xytext=(-8 if left else 8, 6), textcoords="offset points",
                    ha="right" if left else "left", fontsize=8, color=INK)
    ax.scatter([], [], marker="o", color=MUTED, label="no failures")
    ax.scatter([], [], marker="s", color=MUTED, label="with failures")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set(title="Wait vs energy (mixed traffic)", xlabel="Energy (lower is better)",
           ylabel="Average wait (s)")
    top = max(float(r["avg_wait"]) for r in res.values() if r["pattern"] == pattern)
    ax.set_ylim(0, top * 1.2)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.12 * (x1 - x0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    log = read_csv("results/train_log_mixed.csv")
    res = results_lookup(read_csv("results/all_methods.csv"))
    learning_curve(log, res, "results/fig_learning_curve.png")
    comparison(res, "results/fig_comparison.png")
    tradeoff(res, "results/fig_tradeoff.png")
    print("saved results/fig_learning_curve.png, fig_comparison.png, fig_tradeoff.png")