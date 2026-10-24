import yaml
from env.elevator_env import ElevatorSim

cfg = yaml.safe_load(open("configs/default.yaml"))
sim = ElevatorSim(cfg)

p = sim.add_passenger(0, 8)      # rides elevator 0 upward
q = sim.add_passenger(6, 1)      # call at floor 6, also given to elevator 0
sim.assign((0, 1), 0)
sim.assign((6, -1), 0)

for _ in range(11):
    sim.tick()
assert sim.elevators[0].floor == 3 and p in sim.elevators[0].passengers
print("before failure: elevator 0 at floor 3 carrying the passenger")

sim.fail_elevator(0, 100)
e0 = sim.elevators[0]
assert not e0.available and e0.load == 0 and e0.stops == set()
assert set(sim.unassigned) == {(3, 1), (6, -1)}
assert p.origin == 3 and p.t_board is None
print("after failure: passenger waits at floor 3, calls unassigned:", sim.unassigned)

for key in list(sim.unassigned):     # dispatcher reassigns to elevator 1
    sim.assign(key, 1)
for _ in range(200):
    sim.tick()
assert p.t_done is not None and q.t_done is not None
assert sim.elevators[0].available
print("both delivered by elevator 1; elevator 0 repaired")
print("failure tests passed")