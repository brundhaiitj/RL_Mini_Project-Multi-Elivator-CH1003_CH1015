"""Elevator simulator (owner: A).

Step 3: single elevator movement. The Gymnasium environment class
will be added below this class in later steps.
"""


class Elevator:
    def __init__(self, idx, cfg):
        self.idx = idx
        self.n_floors = cfg["building"]["n_floors"]
        self.capacity = cfg["building"]["capacity"]
        self.spf = cfg["simulation"]["seconds_per_floor"]
        self.door_time = cfg["simulation"]["door_time"]
        self.reset()

    def reset(self):
        self.floor = 0
        self.direction = 0        # -1 down, 0 idle, +1 up
        self.stops = set()        # floors this elevator must stop at
        self.passengers = []      # passengers inside (used in step 4)
        self.available = True
        self.door_timer = 0       # seconds of door time left
        self.move_timer = 0       # seconds left to reach the next floor
        self.was_stopped = True
        self.floors_moved = 0     # energy counters
        self.starts = 0

    @property
    def load(self):
        return len(self.passengers)

    def add_stop(self, floor):
        self.stops.add(int(floor))

    def _next_direction(self):
        if not self.stops:
            return 0
        above = any(s > self.floor for s in self.stops)
        below = any(s < self.floor for s in self.stops)
        if self.direction == 1:
            return 1 if above else (-1 if below else 0)
        if self.direction == -1:
            return -1 if below else (1 if above else 0)
        nearest = min(self.stops, key=lambda s: abs(s - self.floor))
        return 1 if nearest > self.floor else -1

    def _try_open(self):
        if self.floor in self.stops:
            self.stops.discard(self.floor)
            self.door_timer = self.door_time
            self.was_stopped = True
            return "open"
        return None

    def tick(self):
        """Advance one second. Returns "open" when doors just opened."""
        if not self.available:
            return None
        if self.door_timer > 0:
            self.door_timer -= 1
            return None
        if self.move_timer == 0:
            event = self._try_open()
            if event:
                return event
            self.direction = self._next_direction()
            if self.direction == 0:
                return None
            self.move_timer = self.spf
            if self.was_stopped:
                self.starts += 1
                self.was_stopped = False
        self.move_timer -= 1
        if self.move_timer == 0:
            self.floor += self.direction
            self.floors_moved += 1
            return self._try_open()
        return None




class Passenger:
    __slots__ = ("origin", "dest", "t_arrive", "t_board", "t_done")

    def __init__(self, origin, dest, t_arrive):
        self.origin = origin
        self.dest = dest
        self.t_arrive = t_arrive
        self.t_board = None
        self.t_done = None

    @property
    def direction(self):
        return 1 if self.dest > self.origin else -1


class ElevatorSim:
    """The whole building: elevators, waiting passengers, hall calls."""

    def __init__(self, cfg, traffic=None):
        self.cfg = cfg
        self.n_floors = cfg["building"]["n_floors"]
        self.n_elevators = cfg["building"]["n_elevators"]
        self.traffic = traffic            # any object with arrivals(t)
        self.elevators = [Elevator(i, cfg) for i in range(self.n_elevators)]
        self.reset()

    def reset(self):
        for e in self.elevators:
            e.reset()
        self.t = 0
        self.waiting = {}      # (floor, direction) -> list of Passenger
        self.calls = {}        # (floor, direction) -> assigned elevator index
        self.unassigned = []   # hall calls waiting for a dispatch decision
        self.finished = []     # delivered passengers
        self.wait_accum = 0    # passenger-seconds waited (reset by the env)

    def n_waiting(self):
        return sum(len(q) for q in self.waiting.values())

    def add_passenger(self, origin, dest):
        p = Passenger(origin, dest, self.t)
        key = (origin, p.direction)
        self.waiting.setdefault(key, []).append(p)
        if key not in self.calls and key not in self.unassigned:
            self.unassigned.append(key)
        return p

    def assign(self, key, elev_idx):
        """Dispatch decision: elevator elev_idx answers hall call key."""
        self.unassigned.remove(key)
        self.calls[key] = elev_idx
        self.elevators[elev_idx].add_stop(key[0])

    def _open_doors(self, e):
        stay = []
        for p in e.passengers:           # unload
            if p.dest == e.floor:
                p.t_done = self.t
                self.finished.append(p)
            else:
                stay.append(p)
        e.passengers = stay
        for d in (1, -1):                # board
            key = (e.floor, d)
            if self.calls.get(key) != e.idx:
                continue
            queue = self.waiting.get(key, [])
            while queue and e.load < e.capacity:
                p = queue.pop(0)
                p.t_board = self.t
                e.passengers.append(p)
                e.add_stop(p.dest)
            del self.calls[key]
            if queue:                    # elevator full: call needs a new decision
                self.unassigned.append(key)
            else:
                self.waiting.pop(key, None)

    def tick(self):
        self.t += 1
        self.wait_accum += self.n_waiting()
        if self.traffic is not None:
            for o, d in self.traffic.arrivals(self.t):
                self.add_passenger(o, d)
        for e in self.elevators:
            if e.tick() == "open":
                self._open_doors(e)