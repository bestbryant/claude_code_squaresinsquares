"""Insert one extra square near a chosen point (e.g. a strip end at a corner), relax at an expanded
f, then hard-square MC compression of the local region (rigid continents elsewhere).
Success = the (n+1)-packing reaches a side below the current best for n+1.

usage: python mc_insert.py n_src rounds steps seed
"""
import sys, os, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mc import run, grow_one
from fault_opt import classify
from relax import relax, free_space_point
from slp import polish, side, normalize
from common import finalize
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOL = os.path.join(ROOT, 'solutions')


def ends(P, K):
    """Tilted squares nearest to each of the four box corners (strip ends)."""
    t = np.mod(P[:, 2], math.pi / 2)
    tl = np.where(np.minimum(t, math.pi / 2 - t) > 1e-9)[0]
    if not len(tl):
        return []
    out = []
    for cx, cy in ((0, 0), (K, 0), (0, K), (K, K)):
        d = np.hypot(P[tl, 0] - cx, P[tl, 1] - cy)
        out.append((d.min(), P[tl[np.argmin(d)], :2]))
    out.sort(key=lambda z: z[0])
    return [p for _, p in out[:2]]


def attempt(P0, center, R, melt, steps, rng):
    L0 = side(P0)
    K, f, gx, gy, grouped = classify(P0, L0)
    gxf = np.maximum(gx, 0).astype(np.float64); gyf = np.maximum(gy, 0).astype(np.float64)
    fs = f + melt
    P = P0.copy()
    P[:, 0] += gxf * melt; P[:, 1] += gyf * melt
    S = K + fs
    reg = (center[0] - R, center[0] + R, center[1] - R, center[1] + R)
    pts, _ = free_space_point(P, S, region=reg, k=3)
    if not pts:
        return None, None
    x, y, _ = pts[int(rng.integers(len(pts)))]
    P = np.vstack([P, [[x, y, rng.uniform(0, math.pi / 2)]]])
    gxf = np.append(gxf, 0.0); gyf = np.append(gyf, 0.0)
    d = np.hypot(P[:, 0] - x, P[:, 1] - y)
    free = np.where((d < R) | np.append(~grouped, True))[0]
    free = free[np.hypot(P[free, 0] - x, P[free, 1] - y) < R + 2]
    gxf[free] = 0; gyf[free] = 0
    k = len(P) - 1
    Q, h = grow_one(P.copy(), free.astype(np.int64), k, 0.05, S, S, steps * 5, 0.03, 0.03, int(rng.integers(1 << 30)), 1e-12)
    if h < 0.5:
        return None, ('grow', h)
    for dxy, pr in ((0.03, 30.0), (0.015, 300.0), (0.006, 3e3), (0.002, 3e4), (0.0007, 3e5)):
        Q, fs, acc = run(Q, free.astype(np.int64), gxf, gyf, K, K, fs, steps, dxy, dxy, pr, -1.0,
                         int(rng.integers(1 << 30)), 1e-12)
    return Q, K + fs


if __name__ == '__main__':
    n_src = int(sys.argv[1]); rounds = int(sys.argv[2]); steps = int(sys.argv[3]); seed = int(sys.argv[4])
    rng = np.random.default_rng(seed)
    P0 = normalize(np.load(os.path.join(SOL, f'n{n_src}.npy')))
    K = int(side(P0))
    tgt_path = os.path.join(SOL, f'n{n_src + 1}.npy')
    for r in range(rounds):
        cur = side(np.load(tgt_path)) if os.path.exists(tgt_path) else math.ceil(math.sqrt(n_src + 1))
        E = ends(P0, K)
        if not E:
            break
        c = E[int(rng.integers(len(E)))]
        R = float(rng.choice([2.0, 3.0, 4.0]))
        melt = float(rng.uniform(0.05, 0.3))
        t = time.time()
        Q, S = attempt(P0, c, R, melt, steps, rng)
        if Q is None:
            print(r, 'no start', S, flush=True)
            continue
        msg = '%d end (%.1f,%.1f) R %.0f melt %.2f -> %.6f (src %.6f, cur n+1 %.6f) %.0fs' % (
            r, c[0], c[1], R, melt, S, side(P0), cur, time.time() - t)
        if S < cur - 1e-9:
            Q2, L = finalize(Q)
            if Q2 is not None and L < cur - 1e-9:
                np.save(tgt_path, Q2)
                msg += ' SAVED %.9f' % L
        print(msg, flush=True)
