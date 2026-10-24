# Reward and metrics definition (owner: B)

All numbers come from configs/default.yaml.

## Reward (computed once per dispatch decision)

r = -(alpha * wait + beta * unfair + gamma * energy)

Each term covers the time since the previous decision:

- wait: passenger-seconds waited by all waiting passengers
  (sum over seconds of the number of passengers waiting).
- unfair: passenger-seconds waited beyond unfair_threshold (60 s).
  For each waiting passenger, every second beyond 60 s of waiting adds 1.
- energy: floors_moved * energy_per_floor
  + starts * energy_per_start_stop
  (floors travelled by all elevators plus start penalties).

Default weights: alpha = 0.01, beta = 0.05, gamma = 0.1.

## Evaluation metrics (measured per episode, averaged over seeds)

1. avg_wait: mean of (t_board - t_arrive) over all boarded passengers.
2. p95_wait: 95th percentile of the same wait times (fairness).
3. max_wait: largest wait of any passenger.
4. pct_over_60: share of passengers with wait > 60 s.
5. avg_trip: mean of (t_done - t_arrive) over delivered passengers.
6. energy: total floors moved + 0.5 * total starts, over all elevators.
7. delivered: number of passengers delivered by the end of the episode.

Passengers still waiting or riding at the end of an episode are counted
in max_wait and pct_over_60 using their wait time so far.

## Evaluation protocol

- Traffic patterns: up_peak, down_peak, mixed, fluctuating.
- Each pattern is run without failures and with failures.
- Methods: random, nearest-car, collective control, DQN.
- Evaluation seeds: 0 to 19 (same seeds for every method).
- Training seeds are different from evaluation seeds (use 1000 and above).