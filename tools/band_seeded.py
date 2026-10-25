"""Seed periodic band search with tilted-stick solutions, then basin-hop to beat them.
usage: seed outdir"""
import sys, os, pickle, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ribbon import basin_hop, periodic_feasible, periodic_slp
seed, outdir = int(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(seed)
out = os.path.join(outdir, 'band_seeded.pkl')
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
cases = []
for h in [3, 4, 5, 6, 7]:
    m = h + 1
    for q in range(2, 15):          # period length (integer units)
        for ps in range(1, q):      # sticks per period
            c = ps / q
            if c >= 1: continue
            t = math.acos(c)
            f = m * c + math.sin(t) - h
            if not (0.6 < f < 0.99): continue
            g = m * ps / q - h
            if g <= 0.2: continue
            cases.append((f, h, q, ps, t))
cases.sort()
print(len(cases), flush=True)
rnd = 0
while True:
    rnd += 1
    for (f0, h, q, ps, t) in cases:
        m = h + 1
        key = (q, 0, tuple([0] * q), tuple([h] * q), m * ps)
        c, s = math.cos(t), math.sin(t)
        if key not in res:
            X = []
            for i in range(ps):
                bx = i / c + 0.5 * (c - s) + 0.3
                for j in range(m):
                    X.append((bx - j * s, 0.5 * (s + c) + j * c, t))
            X = np.array(X)
            fs = f0 + 1e-7
            if not periodic_feasible(X, (q, 0), [0] * q, [h] * q, q, 0, fs):
                fs = f0 + 1e-4
            X, fs = periodic_slp(X, (q, 0), [0] * q, [h] * q, q, 0, fs)
            res[key] = (fs, X, f0)
        fb, Xb, f0 = res[key]
        fn, Xn = basin_hop(Xb, (q, 0), [0] * q, [h] * q, q, 0, fb, rng, n_hops=10 ** 6, time_limit=40 * rnd)
        if fn < fb:
            res[key] = (fn, Xn, f0)
        pickle.dump(res, open(out, 'wb'))
        print(rnd, 'h', h, 'q', q, 'sticks', ps, 'g', round(m * ps / q - h, 3), 'stick f', round(f0, 6), 'best f', round(res[key][0], 6), 'BETTER' if res[key][0] < f0 - 1e-6 else '', flush=True)
