import yaml
from env.elevator_env import ElevatorSim

cfg = yaml.safe_load(open("configs/default.yaml"))
sim = ElevatorSim(cfg)

# Test 1: passenger 0 -> 5, elevator 0 already at floor 0
p = sim.add_passenger(0, 5)
assert sim.unassigned == [(0, 1)]
sim.assign((0, 1), 0)
for _ in range(20):
    sim.tick()
print("test 1:", p.t_arrive, p.t_board, p.t_done)
assert (p.t_arrive, p.t_board, p.t_done) == (0, 1, 15)
assert sim.elevators[0].floor == 5 and len(sim.finished) == 1

# Test 2: passenger 7 -> 2, elevator 1 must travel up first
sim.reset()
p = sim.add_passenger(7, 2)
assert sim.unassigned == [(7, -1)]
sim.assign((7, -1), 1)
for _ in range(40):
    sim.tick()
print("test 2:", p.t_arrive, p.t_board, p.t_done)
assert (p.t_arrive, p.t_board, p.t_done) == (0, 14, 28)

# Test 3: two passengers share one hall call
sim.reset()
sim.add_passenger(0, 5)
sim.add_passenger(0, 3)
assert sim.unassigned == [(0, 1)]
sim.assign((0, 1), 0)
for _ in range(40):
    sim.tick()
print("test 3 delivered:", len(sim.finished))
assert len(sim.finished) == 2

print("simulator tests passed")