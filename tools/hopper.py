"""Long-running basin-hopping campaign over periodic diagonal ribbons.
usage: python hopper.py job njobs seed outdir
"""
import sys, os, pickle, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ribbon import search, basin_hop

job, njobs, seed, outdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
out = os.path.join(outdir, f'hop_{job}.pkl')
res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
cases = []
for m in range(2, 7):
    for Ls in itertools.product([1, 2, 3], repeat=m):
        if Ls[0] != max(Ls) or len(set(Ls)) > 2:
            continue  # canonical rotation-ish, limit variety
        Lc = sum(Ls)
        for extra in range(1, m + 1):
            g = extra / m
            if 0.3 <= g <= 1.0:
                cases.append((m, Ls, Lc + extra))
cases = cases[job::njobs]
print(len(cases), flush=True)
rnd = 0
while True:
    rnd += 1
    tl = 60 * rnd
    for (m, Ls, k) in cases:
        key = (m, m, Ls, k)
        hA = list(range(m)); hB = [i + Ls[i] for i in range(m)]
        cur = res.get(key, (None, None))
        if cur[0] is None:
            if rnd > 1 and key in res:
                continue  # infeasible before
            f, X = search(hA, hB, m, m, k, 0.995, rng, restarts=40, time_limit=60)
            if f is None:
                res[key] = (None, None); pickle.dump(res, open(out, 'wb')); continue
            cur = (f, X)
        f, X = basin_hop(cur[1], (m, m), hA, hB, m, m, cur[0], rng, n_hops=10 ** 6, time_limit=tl)
        if f < cur[0]:
            cur = (f, X)
        res[key] = cur
        pickle.dump(res, open(out, 'wb'))
        print(rnd, key, 'gain', round((k - sum(Ls)) / m, 3), 'f', round(cur[0], 6), flush=True)
