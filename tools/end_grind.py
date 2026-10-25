"""Insert extra squares near band ends of a packing by local penalty relaxation.
usage: python end_grind.py in.npy S out.npy minutes seed  (regions = 'auto': near walls where tilted squares touch)"""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from relax import relax, free_space_point
from slp import feasible


def grind(P, S, minutes, rng, regions, radius=3.0, verbose=True):
    t0 = time.time()
    tries = succ = 0
    while time.time() - t0 < 60 * minutes:
        reg = regions[rng.integers(len(regions))]
        pts, _ = free_space_point(P, S, k=12, region=reg)
        if not pts:
            continue
        x, y, dval = pts[rng.integers(len(pts))]
        near = np.hypot(P[:, 0] - x, P[:, 1] - y)
        ang = P[np.argmin(near), 2] if rng.random() < 0.6 else rng.choice([0.0, rng.uniform(-0.8, 0.8)])
        Q = np.vstack([P, [[x + rng.normal(0, 0.1), y + rng.normal(0, 0.1), ang]]])
        r = radius * (1 + rng.random())
        free = np.where(np.hypot(Q[:, 0] - x, Q[:, 1] - y) < r)[0]
        Q2, E = relax(Q, S, free=free, iters=3000, rounds=5, mu=1e-10)
        tries += 1
        if E < 1e-22:
            P = Q2; succ += 1
            if verbose:
                print(f'  +1 at ({x:.2f},{y:.2f}) n={len(P)} tries={tries} t={time.time()-t0:.0f}', flush=True)
    return P, succ, tries


if __name__ == '__main__':
    P = np.load(sys.argv[1]); S = float(sys.argv[2])
    rng = np.random.default_rng(int(sys.argv[5]))
    regs = [tuple(map(float, r.split(','))) for r in sys.argv[6:]] or [(0, S, 0, S)]
    P, s, t = grind(P, S, float(sys.argv[4]), rng, regs)
    np.save(sys.argv[3], P)
    print('done', len(P), s, t, feasible(P, S, tol=1e-9))
