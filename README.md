# Multi-Elevator Dispatching Under Disruptions using RL

A Deep Q-Network (DQN) dispatcher for a building with 4 elevators, trained in our own simulator with changing traffic and random elevator breakdowns. It is compared with three rule-based dispatchers: Random, Nearest car and Collective control.

**Team:** Brundha Devi A (B23CH1015), Anshika Rolya (B23CH1003)

## Main result

Mixed traffic with random breakdowns, average of 20 unseen test hours:

| Method | Avg wait (s) | 95th-percentile wait (s) | Energy | Reward |
|---|---|---|---|---|
| Random | 12.8 | 35.6 | 3167 | -101.9 |
| Nearest car | 10.0 | 40.0 | 2560 | -81.5 |
| Collective control | 5.2 | 15.2 | 3078 | -59.0 |
| **DQN (ours)** | 6.0 | 20.7 | **2678** | -59.8 |

The DQN beats Random and Nearest car, and matches Collective control on reward while using about 13% less energy. During a forced 20-minute outage of one elevator, all methods get slower and Collective control degrades the least. See `results/` and the report for details.

## Requirements

- Python 3.10 or newer (tested with Python 3.13 and 3.14)
- Packages in `requirements.txt`: gymnasium, numpy, torch, matplotlib, pyyaml
- A normal CPU is enough; no GPU needed

## Installation

```bash
git clone https://github.com/brundhaiitj/RL_Mini_Project-Multi-Elivator-CH1003_CH1015.git
cd RL_Mini_Project-Multi-Elivator-CH1003_CH1015
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional, smaller CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## How to run

All commands are run from the repository folder with the virtual environment active.

| Step | Command | What it does | Time |
|---|---|---|---|
| 1. Test | `python3 test_gym_env.py` | Checks the environment (see "Tests" for all tests) | seconds |
| 2. Train | `python3 train.py` | Trains the DQN for 500 episodes, saves `models/dqn_mixed.pt` and `results/train_log_mixed.csv` | about 7-10 min |
| 3. Evaluate | `python3 evaluate.py` | Runs all 4 methods on 20 test seeds, 4 traffic patterns, with and without breakdowns; saves `results/all_methods.csv` | about 1 min |
| 4. Plots | `python3 plots.py` | Makes `fig_learning_curve.png`, `fig_comparison.png`, `fig_tradeoff.png` in `results/` | seconds |
| 5. Outage test | `python3 experiment_failure.py` | Switches off elevator 0 from minute 20 to 40; saves `results/forced_failure.csv` and `fig_forced_failure.png` | about 4 min |

Quick versions: `python3 train.py --episodes 20`, `python3 evaluate.py --seeds 3`, `python3 experiment_failure.py --seeds 3`.

A trained model is already included (`models/dqn_mixed.pt`), so steps 3 to 5 work without training first.

## Reproducing our results

1. Install as above.
2. Run steps 2 to 5 in order. Training uses seeds from 1000 upwards; evaluation always uses seeds 0 to 19, the same for every method.
3. Compare with the files in `results/`. Evaluating the included model reproduces `results/all_methods.csv` exactly. Retraining gives slightly different numbers each time, because neural-network training varies a little between runs and computers.

## Tests

```bash
python3 test_elevator.py          # elevator movement and doors
python3 test_traffic.py           # passenger arrivals for each traffic pattern
python3 test_sim.py               # boarding and delivery times
python3 test_failure.py           # breakdown: passengers requeued, calls reassigned, repair
python3 test_random_failures.py   # random breakdowns respect the limits
python3 test_baselines.py         # random and nearest-car run for one hour
python3 test_collective.py        # collective control uses direction correctly
python3 test_gym_env.py           # Gymnasium checker and all baselines through the env
python3 test_mask.py              # full elevators are masked (no stuck elevator)
python3 test_dqn.py               # masking, learning, save/load of the DQN
```

## RL formulation (short)

- **State (31 numbers):** for each of the 4 elevators: floor, direction, load, working/broken, queued stops, distance to the call, estimated arrival time at the call; plus call floor, call direction and number of people waiting.
- **Action (4):** which elevator takes the new call. Broken elevators, and full ones when another has room, are masked.
- **Reward:** `r = -(0.01 * waiting + 0.05 * waiting beyond 60 s + 0.01 * energy)`, counted between two decisions.
- **Algorithm:** DQN with experience replay, target network, Double-DQN target and epsilon-greedy exploration.

Full details: `SPEC.md` (interface) and `METRICS.md` (reward and metrics).

## Project structure

```
configs/default.yaml     all settings: building, traffic, breakdowns, reward weights, DQN
env/traffic.py           passenger generator (Poisson arrivals, 4 traffic patterns)
env/elevator_env.py      simulator: elevators, passengers, breakdowns
env/gym_env.py           Gymnasium environment: state, action mask, reward
baselines/               random_agent.py, nearest_car.py, collective.py
agent/dqn.py             Q-network, replay buffer, DQN agent
train.py                 trains the DQN
evaluate.py              compares all methods on fixed seeds
plots.py                 makes the report figures
experiment_failure.py    20-minute outage experiment
test_*.py                tests
models/                  trained DQN (dqn_mixed.pt) and best-episode backup
results/                 result tables (.csv), training log and figures (.png)
report/                  project report
```

## Changing settings

All numbers are in `configs/default.yaml`, for example the number of floors and elevators, the passenger arrival rate, how often elevators break down, the reward weights and the DQN settings. After a change, retrain (`train.py`) and re-run steps 3 to 5.