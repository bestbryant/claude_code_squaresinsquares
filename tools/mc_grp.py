"""Melt-and-recompress the non-grid part of a packing: rigid upright groups move with f,
tilted squares (and grid squares within R of them) are free hard squares under MC compression."""
import sys, os, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mc import run
from fault_opt import classify
from slp import polish, side, normalize
from common import verify, finalize
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(__file__), '..')


def attempt(P0, melt, R, steps, rng, sched=((0.04, 30.0), (0.02, 300.0), (0.008, 3000.0), (0.003, 3e4), (0.001, 3e5))):
    L0 = side(P0)
    K, f, gx, gy, grouped = classify(P0, L0)
    tilted = np.where(~grouped)[0]
    tree = cKDTree(P0[grouped, :2])
    near = set()
    gi = np.where(grouped)[0]
    for i in tilted:
        for j in tree.query_ball_point(P0[i, :2], R):
            near.add(gi[j])
    free = np.array(sorted(set(tilted) | near), np.int64)
    gxf = np.maximum(gx, 0).astype(np.float64); gyf = np.maximum(gy, 0).astype(np.float64)
    gxf[free] = 0; gyf[free] = 0
    P = P0.copy()
    fs = f + melt
    P[:, 0] += gxf * melt; P[:, 1] += gyf * melt
    # free squares: spread affinely within the box to use the slack
    for dxy, pr in sched:
        P, fs, acc = run(P, free, gxf, gyf, K, K, fs, steps, dxy, dxy, pr, -1.0, int(rng.integers(1 << 30)), 1e-12)
    return P, K + fs, len(free)


if __name__ == '__main__':
    out = sys.argv[1]; rounds = int(sys.argv[2]); steps = int(sys.argv[3]); seed = int(sys.argv[4])
    ns = list(map(int, sys.argv[5:]))
    rng = np.random.default_rng(seed)
    for r in range(rounds):
      for n in ns:
        P0 = normalize(np.load(os.path.join(ROOT, 'solutions', f'n{n}.npy'))); L0 = side(P0)
        best = L0
        melt = float(rng.uniform(0.03, 0.3)); R = float(rng.choice([1.2, 1.6, 2.2, 3.0]))
        t = time.time()
        P, S, nf = attempt(P0, melt, R, steps, rng)
        if S < L0 + 0.003:
            Q, L = polish(P, iters=300, time_limit=120)
        else:
            Q, L = P, S
        ok = False
        if L < best - 1e-9:
            Q, L = finalize(Q)
            ok = Q is not None and L < best - 1e-9
        print(n, r, 'melt %.3f R %.1f' % (melt, R), 'free %d mc %.6f polished %.6f (cur %.6f) ok=%s %.0fs' % (nf, S, L, L0, ok, time.time() - t), flush=True)
        if ok and L < best - 1e-9:
            best = L
            np.save(os.path.join(ROOT, 'solutions', f'n{n}.npy'), Q); print('SAVED', n, L, flush=True)
