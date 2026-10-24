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