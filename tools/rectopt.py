"""Direct optimisation of rectangle gadgets: w*h+dm unit squares in (w+f) x (h+f), minimise f.

Basin hopping with the overlap-penalty relaxation: from the best feasible packing, shrink f by
delta, kick a random subset and relax; accept when overlap-free.  Starts are a w x h grid with dm
extra squares dropped at random points, plus (optionally) a seed packing.
"""
import math, sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from relax import relax


def ok(E):
    return E < 1e-24


def start(w, h, dm, f, rng):
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, indexing='ij')
    P = np.c_[xs.ravel() * (w + f) / w, ys.ravel() * (h + f) / h, np.zeros(w * h)]
    E = np.c_[rng.uniform(0.5, w + f - 0.5, dm), rng.uniform(0.5, h + f - 0.5, dm), rng.uniform(0, math.pi / 2, dm)]
    P = np.vstack([P, E])
    P[:, :2] += rng.normal(0, 0.05, (len(P), 2))
    P[:, 2] += rng.normal(0, 0.05, len(P))
    return P


def feasible_at(P, w, h, f, rng, tries=6, kick=0.15):
    S = (w + f, h + f)
    Q, E = relax(P, S, iters=1500, rounds=4, mu=1e-11)
    if ok(E):
        return Q
    for _ in range(tries):
        R = P.copy()
        n = len(R)
        k = rng.integers(1, n + 1)
        sel = rng.choice(n, k, replace=False)
        R[sel, :2] += rng.normal(0, kick, (k, 2))
        R[sel, 2] += rng.normal(0, kick, k)
        Q, E = relax(R, S, iters=1500, rounds=4, mu=1e-11)
        if ok(E):
            return Q
    return None


def optimise(w, h, dm, time_limit=60, seed=0, seedP=None, f_start=0.95):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_f, best_P = None, None
    if seedP is not None:
        best_P, best_f = seedP[0].copy(), seedP[1]
    while time.time() - t0 < time_limit:
        # new start
        if best_P is None or rng.random() < 0.3:
            f = f_start if best_f is None else best_f + 0.05
            P = feasible_at(start(w, h, dm, f, rng), w, h, f, rng, tries=10, kick=0.3)
            if P is None:
                continue
            cf, cP = f, P
        else:
            cf, cP = best_f, best_P
        delta = 0.02
        fails = 0
        while time.time() - t0 < time_limit and delta > 2e-5 and fails < 30:
            ft = cf - delta
            Q = feasible_at(cP, w, h, ft, rng, tries=3)
            if Q is not None:
                cf, cP = ft, Q
                delta *= 1.4
                fails = 0
            else:
                delta *= 0.6
                fails += 1
        if best_f is None or cf < best_f - 1e-9:
            best_f, best_P = cf, cP
    return best_f, best_P


def job(args):
    w, h, dm, tl, seed = args
    f, P = optimise(w, h, dm, tl, seed)
    return (w, h, dm), f, P


if __name__ == '__main__':
    from multiprocessing import Pool
    out = sys.argv[1]
    tl = float(sys.argv[2])
    specs = [tuple(map(int, s.split(','))) for s in sys.argv[3:]]
    res = {}
    if os.path.exists(out):
        res = pickle.load(open(out, 'rb'))
    with Pool(4) as pool:
        for key, f, P in pool.imap_unordered(job, [(w, h, dm, tl, i) for i, (w, h, dm) in enumerate(specs)]):
            print(key, f, flush=True)
            if f is not None and (key not in res or f < res[key][0]):
                res[key] = (f, P)
                pickle.dump(res, open(out, 'wb'))
