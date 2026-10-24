"""Nearest-car baseline (owner: B).

Assigns each hall call to the closest elevator that is working.
Elevators that are full are avoided when another one is available.
"""


class NearestCar:
    name = "nearest_car"

    def reset(self, seed=None):
        pass

    def select(self, sim, key):
        floor = key[0]
        cands = [e for e in sim.elevators if e.available and e.load < e.capacity]
        if not cands:
            cands = [e for e in sim.elevators if e.available]
        if not cands:
            return None          # every elevator is down: leave the call waiting
        best = min(cands, key=lambda e: (abs(e.floor - floor), e.idx))
        return best.idx