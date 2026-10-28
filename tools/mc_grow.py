"""Fixed-box insertion: take the best packing for n (side L), add a tiny square in one of the
largest holes and let MC (free squares within R of the hole) make room while it grows to unit
size.  Success gives an (n+1)-packing with side L; saved if that beats the current n+1.

usage: python mc_grow.py seed steps tries n1 [n2 ...]
"""
import sys, os, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mc import grow_one
from relax import free_space_point
from slp import side, normalize
from common import finalize

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOL = os.path.join(ROOT, 'solutions')


def cur_side(n):
    p = os.path.join(SOL, f'n{n}.npy')
    return side(np.load(p)) if os.path.exists(p) else math.ceil(math.sqrt(n))


def try_grow(P, S, pt, R, steps, rng):
    x, y = pt
    Q = np.vstack([P, [[x, y, rng.uniform(0, math.pi / 2)]]])
    d = np.hypot(Q[:, 0] - x, Q[:, 1] - y)
    free = np.where(d < R)[0].astype(np.int64)
    k = len(Q) - 1
    Q, h = grow_one(Q, free, k, 0.04, S, S, steps, 0.02, 0.02, int(rng.integers(1 << 30)), 1e-12)
    return Q, h


if __name__ == '__main__':
    seed = int(sys.argv[1]); steps = int(sys.argv[2]); tries = int(sys.argv[3])
    ns = list(map(int, sys.argv[4:]))
    rng = np.random.default_rng(seed)
    for n in ns:
        P = normalize(np.load(os.path.join(SOL, f'n{n}.npy')))
        S = side(P)
        tgt = cur_side(n + 1)
        if S >= tgt - 1e-9:
            continue
        pts, _ = free_space_point(P, S, k=tries)
        best_h = 0
        t0 = time.time()
        for (x, y, dd) in pts:
            for R in (3.0, 6.0):
                Q, h = try_grow(P, S, (x, y), R, steps, rng)
                best_h = max(best_h, h)
                if h >= 0.5:
                    Q2, L = finalize(Q)
                    if Q2 is not None and L < cur_side(n + 1) - 1e-9:
                        np.save(os.path.join(SOL, f'n{n + 1}.npy'), Q2)
                        print('SAVED n=%d L=%.9f (was %.9f)' % (n + 1, L, tgt), flush=True)
                    break
            if best_h >= 0.5:
                break
        print(n, '-> n+1', 'L=%.6f tgt=%.6f best_h=%.3f %.0fs' % (S, tgt, best_h, time.time() - t0), flush=True)
