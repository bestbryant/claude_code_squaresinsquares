"""Region re-pack by LS growth: remove the squares whose centres lie in a disc/rect, then try to
grow k+extra squares from random tiny seeds inside the hole.  Success -> packing for n+extra at
the same side.

usage: python ls_region.py seed steps n R tries [cx cy]
"""
import sys, os, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mc import grow_all
from relax import relax
from slp import side, normalize, corner_arrays
from common import finalize

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOL = os.path.join(ROOT, 'solutions')


def cur_side(n):
    p = os.path.join(SOL, f'n{n}.npy')
    return side(np.load(p)) if os.path.exists(p) else math.ceil(math.sqrt(n))


def holes_score(P, S, cx, cy, R):
    """free area inside the disc (approx by grid sampling)."""
    from matplotlib.path import Path
    g = np.linspace(-R, R, 60)
    X, Y = np.meshgrid(cx + g, cy + g)
    m = (np.hypot(X - cx, Y - cy) < R) & (X > 0) & (Y > 0) & (X < S) & (Y < S)
    pts = np.c_[X[m], Y[m]]
    occ = np.zeros(len(pts), bool)
    ox, oy = corner_arrays(P)
    near = np.where(np.hypot(P[:, 0] - cx, P[:, 1] - cy) < R + 1)[0]
    for i in near:
        occ |= Path(np.c_[P[i, 0] + ox[i], P[i, 1] + oy[i]]).contains_points(pts)
    return (~occ).mean() * m.sum() / m.size * (2 * R) ** 2


def inside_any(F, x, y):
    c, s = np.cos(F[:, 2]), np.sin(F[:, 2])
    u = (x - F[:, 0]) * c + (y - F[:, 1]) * s
    v = -(x - F[:, 0]) * s + (y - F[:, 1]) * c
    return bool(((np.abs(u) < 0.56) & (np.abs(v) < 0.56)).any())


def attempt(P, S, cx, cy, R, extra, steps, rng, rect=None):
    d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    rem = d < R
    if rect is not None:
        x0, x1, y0, y1 = rect
        rem = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        R = 0.5 * math.hypot(x1 - x0, y1 - y0)
    k = int(rem.sum())
    F = P[~rem]
    Fn = F[np.hypot(F[:, 0] - cx, F[:, 1] - cy) < R + 2.5]
    m = k + extra
    # random seeds inside the disc
    Q = np.empty((m, 3))
    for i in range(m):
        while True:
            if rect is None:
                r = R * math.sqrt(rng.random()); a = rng.random() * 2 * math.pi
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            else:
                x, y = rng.uniform(rect[0], rect[1]), rng.uniform(rect[2], rect[3])
            if 0.1 < x < S - 0.1 and 0.1 < y < S - 0.1 and not inside_any(Fn, x, y) and \
                    (i == 0 or np.hypot(Q[:i, 0] - x, Q[:i, 1] - y).min() > 0.08):
                break
        Q[i] = (x, y, rng.random() * math.pi / 2)
    Q, h = grow_all(Fn.copy(), Q, 0.02, S, steps, 0.5, 0.15, int(rng.integers(1 << 30)), 1e-12, 50, 0.002)
    allP = np.vstack([F, Q])
    if 0.47 < h < 0.5:
        free = np.arange(len(F), len(allP))
        R2, E = relax(allP, S, free=free, iters=4000, rounds=8, mu=1e-11)
        if E < 1e-24:
            return R2, 0.5, k
    return allP, h, k


if __name__ == '__main__':
    seed = int(sys.argv[1]); steps = int(sys.argv[2]); n = int(sys.argv[3]); R = float(sys.argv[4]); tries = int(sys.argv[5])
    EXTRA = int(os.environ.get('EXTRA', 1))
    RECT = tuple(map(float, os.environ['RECT'].split(','))) if os.environ.get('RECT') else None
    rng = np.random.default_rng(seed)
    P = normalize(np.load(os.path.join(SOL, f'n{n}.npy'))); S = side(P)
    if len(sys.argv) > 7:
        centres = [(float(sys.argv[6]), float(sys.argv[7]))]
    else:
        # sample centres, keep the ones with the most free area
        c = [(rng.uniform(R * 0.5, S - R * 0.5), rng.uniform(R * 0.5, S - R * 0.5)) for _ in range(300)]
        sc = [holes_score(P, S, x, y, R) for x, y in c]
        centres = [c[i] for i in np.argsort(sc)[::-1][:tries]]
    for t_ in range(tries):
        cx, cy = centres[t_ % len(centres)]
        t0 = time.time()
        newP, h, k = attempt(P, S, cx, cy, R, EXTRA, steps, rng, RECT)
        msg = 'n=%d c=(%.1f,%.1f) R=%.1f k=%d h=%.4f %.0fs' % (n, cx, cy, R, k, h, time.time() - t0)
        if h >= 0.5 and EXTRA > 0:
            Q2, L = finalize(newP)
            if Q2 is not None and L < cur_side(n + 1) - 1e-9:
                np.save(os.path.join(SOL, f'n{n + 1}.npy'), Q2)
                msg += ' SAVED n+1 L=%.9f' % L
        print(msg, flush=True)
