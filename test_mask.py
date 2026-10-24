import yaml
from env.gym_env import ElevatorEnv
from env.elevator_env import Passenger

cfg = yaml.safe_load(open("configs/default.yaml"))
env = ElevatorEnv(cfg, pattern="down_peak", failures=False)
env.reset(seed=0)
floor = env.key[0]
cap = cfg["building"]["capacity"]

# Case 1: elevator 0 is full -> masked, the others stay allowed
env.sim.elevators[0].passengers = [Passenger(0, 5, 0) for _ in range(cap)]
print("elevator 0 full:", env.action_mask())
assert list(env.action_mask()) == [0, 1, 1, 1]

# Case 2: every elevator full, elevator 1 standing at the call floor -> only 1 masked
for e in env.sim.elevators:
    e.passengers = [Passenger(0, 5, 0) for _ in range(cap)]
    e.floor = (floor + 1) % cfg["building"]["n_floors"]
env.sim.elevators[1].floor = floor
print("all full, elevator 1 at call floor:", env.action_mask())
assert list(env.action_mask()) == [1, 0, 1, 1]

# Case 3: every elevator full and at the call floor -> nobody allowed; time still moves
for e in env.sim.elevators:
    e.floor = floor
t0 = env.sim.t
env.step(0)
assert env.sim.t > t0
print("no elevator can take the call: simulator still advanced", env.sim.t - t0, "s")
print("mask tests passed")