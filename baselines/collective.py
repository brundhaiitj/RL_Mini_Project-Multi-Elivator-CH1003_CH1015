"""Collective-control baseline (owner: B).

Assigns each hall call to the elevator with the lowest estimated
time of arrival (ETA), taking its travel direction into account:
- an elevator moving toward the call in the same direction picks it up on the way;
- otherwise it must finish its current run first, then come back.
Each queued stop adds one door time to the estimate.
"""


class CollectiveControl:
    name = "collective"

    def reset(self, seed=None):
        pass

    def eta(self, e, floor, direction):
        if e.direction == 0:
            dist = abs(floor - e.floor)
        else:
            d = e.direction
            diff = (floor - e.floor) * d
            ahead = diff > 0 or (diff == 0 and e.move_timer == 0)
            if ahead and direction == d:
                dist = abs(floor - e.floor)
            else:
                stops = list(e.stops) + [e.floor]
                far = max(stops) if d == 1 else min(stops)
                dist = abs(far - e.floor) + abs(far - floor)
        return dist * e.spf + len(e.stops) * e.door_time + e.door_timer

    def select(self, sim, key):
        floor, direction = key
        cands = [e for e in sim.elevators if e.available and e.load < e.capacity]
        if not cands:
            cands = [e for e in sim.elevators if e.available]
        if not cands:
            return None
        best = min(cands, key=lambda e: (self.eta(e, floor, direction), e.idx))
        return best.idx