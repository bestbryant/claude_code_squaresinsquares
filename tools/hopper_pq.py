"""Basin hopping over periodic ribbons with slope q/p != 1. usage: job njobs seed outdir"""
import sys, os, pickle, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ribbon import search, basin_hop
job, njobs, seed, outdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
out = os.path.join(outdir, f'hoppq_{job}.pkl')
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
cases = []
for p, q in [(2, 1), (3, 1), (3, 2), (4, 1), (4, 3), (5, 2), (5, 3)]:
    # staircase of slope q/p: hA(i) = floor(i*q/p)
    hA = [(i * q) // p for i in range(p)]
    for Ls in itertools.product([1, 2, 3], repeat=p):
        if len(set(Ls)) > 2 or Ls[0] != max(Ls):
            continue
        hB = [hA[i] + Ls[i] for i in range(p)]
        Lc = sum(Ls)
        for extra in range(1, p + q + 1):
            g = extra / p
            if 0.2 <= g <= 1.4:
                cases.append((p, q, tuple(hA), tuple(hB), Lc + extra))
cases = cases[job::njobs]
print(len(cases), flush=True)
rnd = 0
while True:
    rnd += 1
    for (p, q, hA, hB, k) in cases:
        key = (p, q, hA, hB, k)
        cur = res.get(key, (None, None))
        if cur[0] is None:
            if key in res:
                continue
            f, X = search(list(hA), list(hB), p, q, k, 0.995, rng, restarts=30, time_limit=45)
            if f is None:
                res[key] = (None, None); pickle.dump(res, open(out, 'wb')); continue
            cur = (f, X)
        f, X = basin_hop(cur[1], (p, q), list(hA), list(hB), p, q, cur[0], rng, n_hops=10 ** 6, time_limit=45 * rnd)
        if f < cur[0]:
            cur = (f, X)
        res[key] = cur
        pickle.dump(res, open(out, 'wb'))
        print(rnd, key, 'gain/col', round((k - sum(b - a for a, b in zip(hA, hB))) / p, 3), 'f', round(cur[0], 6), flush=True)
