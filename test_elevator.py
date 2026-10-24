import yaml
from env.elevator_env import Elevator

cfg = yaml.safe_load(open("configs/default.yaml"))

e = Elevator(0, cfg)
e.add_stop(3)
for t in range(1, 12):
    ev = e.tick()
    print("t =", t, "floor =", e.floor, "event =", ev)
assert e.floor == 3
assert e.floors_moved == 3
assert e.starts == 1
assert e.stops == set()
assert e.direction == 0

e2 = Elevator(1, cfg)
e2.add_stop(5)
e2.available = False
for _ in range(20):
    e2.tick()
assert e2.floor == 0

print("elevator movement tests passed")