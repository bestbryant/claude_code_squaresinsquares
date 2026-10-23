"""For a band type (W, m, t) find, for each square count c, the minimal end depth D_c.

usage: python endsearch.py W c_lo c_hi seed out.npz
Squeeze: start feasible at a generous D, then shrink D with warm-started relaxation and
basin hopping; record the smallest feasible D for every count.
"""
import sys, math, time
import numpy as np
from polypack import relax_poly, sample_in_poly
from bandend import stick_params, trapezoid, sticks

TOL = 1e-22


def feasible_at(P, W, t, D, rng, hops=40, mu=1e-9):
    A, b, X0 = trapezoid(W, t, D)
    Q, E = relax_poly(P, A, b, mu)
    if E < TOL:
        return Q, E
    best = (Q, E)
    for h in range(hops):
        R = best[0].copy()
        k = rng.integers(1, 4)
        idx = rng.choice(len(R), k, replace=False)
        if rng.random() < 0.4:
            R[idx, :2] = sample_in_poly(A, b, (0, X0, 0, W), rng, k)
        else:
            R[idx, :2] += rng.normal(0, 0.25, (k, 2))
        if rng.random() < 0.5:
            R[idx, 2] = rng.choice([0.0, t, -t], k) + rng.normal(0, 0.03, k)
        R, E2 = relax_poly(R, A, b, mu)
        if E2 < best[1]:
            best = (R, E2)
        if E2 < TOL:
            break
    return best


def min_depth(W, count, rng, D_start=None, init=None, t_budget=600):
    m, t, p = stick_params(W)
    t0 = time.time()
    if D_start is None:
        D_start = count / m * p + 1.0
    D = D_start
    A, b, X0 = trapezoid(W, t, D)
    if init is None or len(init) != count:
        P = None
        for attempt in range(20):
            pts = sample_in_poly(A, b, (0, X0, 0, W), rng, count)
            P0 = np.c_[pts, rng.choice([0.0, t], count)]
            P, E = feasible_at(P0, W, t, D, rng, hops=10)
            if E < TOL:
                break
        else:
            return None, None
    else:
        P = init
    best_D, best_P = D, P.copy()
    step = 0.2
    while step > 2e-4 and time.time() - t0 < t_budget:
        Dn = best_D - step
        if Dn < 0:
            step /= 2
            continue
        Q, E = feasible_at(best_P, W, t, Dn, rng)
        if E < TOL:
            best_D, best_P = Dn, Q
        else:
            step /= 2
    return best_D, best_P


if __name__ == '__main__':
    W = float(sys.argv[1]); lo = int(sys.argv[2]); hi = int(sys.argv[3])
    seed = int(sys.argv[4]); out = sys.argv[5]
    rng = np.random.default_rng(seed)
    res = {}
    for c in range(lo, hi + 1):
        D, P = min_depth(W, c, rng)
        print(f'W={W} count={c} minD={D}', flush=True)
        if D is not None:
            res[c] = (D, P)
            np.savez(out, **{f'c{k}_D': v[0] for k, v in res.items()}, **{f'c{k}_P': v[1] for k, v in res.items()})
