"""Periodic horizontal band cross-sections (flat floor = grid core, flat ceiling = wall at h+f).
Gain per unit length g = k/p - h.  usage: job njobs seed outdir"""
import sys, os, pickle
import numpy as np
from fractions import Fraction
sys.path.insert(0, os.path.dirname(__file__))
from ribbon import search, basin_hop
job, njobs, seed, outdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
out = os.path.join(outdir, f'band_{job}.pkl')
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
cases = []
for h in [2, 3, 4, 5, 6]:
    for p in range(2, 13):
        for e in range(1, p):
            g = Fraction(e, p)
            if g.denominator != p or not (0.3 <= g <= 0.75):
                continue
            k = h * p + e
            if k > 70:
                continue
            cases.append((p, 0, tuple([0] * p), tuple([h] * p), k))
cases = cases[job::njobs]
print(len(cases), flush=True)
rnd = 0
while True:
    rnd += 1
    for key in cases:
        p, q, hA, hB, k = key
        cur = res.get(key, (None, None))
        if cur[0] is None:
            if key in res:
                continue
            f, X = search(list(hA), list(hB), p, q, k, 0.995, rng, restarts=30, time_limit=60)
            if f is None:
                res[key] = (None, None); pickle.dump(res, open(out, 'wb')); continue
            cur = (f, X)
        f, X = basin_hop(cur[1], (p, q), list(hA), list(hB), p, q, cur[0], rng, n_hops=10 ** 6, time_limit=60 * rnd)
        if f < cur[0]:
            cur = (f, X)
        res[key] = cur
        pickle.dump(res, open(out, 'wb'))
        h = hB[0]
        print(rnd, 'h', h, 'p', p, 'k', k, 'g', round(k / p - h, 3), 'f', round(cur[0], 6), flush=True)
