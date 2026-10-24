import yaml
from env.traffic import TrafficGenerator

cfg = yaml.safe_load(open("configs/default.yaml"))
for p in ["up_peak", "down_peak", "mixed", "fluctuating"]:
    g = TrafficGenerator(cfg, p, seed=0)
    total = lobby = 0
    for t in range(3600):
        for o, d in g.arrivals(t):
            assert o != d
            total += 1
            lobby += (o == 0)
    print(p, "passengers:", total, "share starting at lobby:", round(lobby / total, 2))