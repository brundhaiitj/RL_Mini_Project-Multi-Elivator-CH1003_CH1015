# Interface spec (frozen Sat 24 Oct)

## Setup
- 10 floors (0-9), 4 elevators, capacity 8, time step 1 s, episode 3600 s
- All values live in configs/default.yaml; code reads from there

## Decision timing
The agent acts once per new hall call (event-driven).
After the assignment, each elevator serves its own stop queue
(continue in direction, stop at queued floors, then reverse).

## State (31-dim float vector, all normalised)
Per elevator i (7 features, in elevator order 0..3):
1. floor / (n_floors - 1)
2. direction: -1 down, 0 idle, +1 up
3. load / capacity
4. available: 1 working, 0 failed
5. queued stops / 10 (clipped to 1)
6. |call_floor - floor| / (n_floors - 1)
7. estimated arrival time at the call / 60 s (clipped to 1); direction-aware:
   an elevator moving toward the call in the call's direction picks it up on the way,
   otherwise it finishes its current run first; each queued stop adds one door time.
   (Added Sun 25 Oct: improved DQN waiting time on every traffic pattern.)
Global (3 features):
29. call_floor / (n_floors - 1)
30. call_direction: +1 up, -1 down
31. total waiting passengers / 50 (clipped to 1)

## Action
Discrete(n_elevators). Action i = assign the new call to elevator i.
Failed elevators are masked, and full elevators are masked while another has room: info["action_mask"] is a 0/1 array.

## Reward
r = -(alpha * wait + beta * unfair + gamma * energy)
- wait: summed waiting seconds of all waiting passengers since last decision
- unfair: penalty for waits above 60 s (max wait term)
- energy: floors travelled + small penalty per start/stop
alpha, beta, gamma are set in config.

## Environment API (Gymnasium)
- env.reset(seed=...) -> obs, info
- env.step(action) -> obs, reward, terminated, truncated, info
- info keys: action_mask, avg_wait, max_wait, energy

## Disruptions
At random times an elevator goes out of service for a random duration.
Its passengers exit at the next floor and its queued calls are reassigned.

## File ownership
- A: env/elevator_env.py, agent/, train.py
- B: configs/, env/traffic.py, baselines/, evaluate.py
