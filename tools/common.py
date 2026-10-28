"""Shared helpers: load/write submissions, compute L, verify with the official metric."""
import csv, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import metric  # official Kaggle metric (copied verbatim)


def load_submission(path):
    """Return {n: np.array (n,3) float64 [x,y,angle]} for every complete puzzle in a CSV."""
    import pandas as pd
    df = pd.read_csv(path, usecols=lambda c: c in ('n', 'square_id', 'x', 'y', 'angle'))
    out = {}
    for n, g in df.groupby('n'):
        n = int(n)
        if len(g) != n:
            continue
        g = g.sort_values('square_id')
        if list(g['square_id']) != list(range(n)):
            continue
        a = g[['x', 'y', 'angle']].to_numpy(dtype=np.float64)
        if not np.isfinite(a).all():
            continue
        out[n] = a
    return out


def corners(a):
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    h = 0.5
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    xs = np.stack([a[:, 0] + dx1, a[:, 0] + dx2, a[:, 0] - dx1, a[:, 0] - dx2], 1)
    ys = np.stack([a[:, 1] + dy1, a[:, 1] + dy2, a[:, 1] - dy1, a[:, 1] - dy2], 1)
    return xs, ys


def side(a):
    xs, ys = corners(a)
    return max(xs.max() - xs.min(), ys.max() - ys.min())


def fmt(v):
    r = repr(float(v))
    return r


def verify(a):
    """Official verification on the float64-repr strings (what Kaggle sees after pandas round-trip)."""
    sq = []
    for x, y, t in a:
        sq.append(metric.Square(x=float(fmt(x)), y=float(fmt(y)), angle=float(fmt(t)),
                                x_str=fmt(x), y_str=fmt(y), angle_str=fmt(t)))
    ok, _ = metric.verify_puzzle(sq)
    if not ok:
        return False, None
    return True, float(metric.measure_side_mp(sq))


def score_n(n, L):
    return L / math.sqrt(n) - 1


def grid(n):
    k = math.ceil(math.sqrt(n))
    return np.array([[i % k + 0.5, i // k + 0.5, 0.0] for i in range(n)])


def write_submission(path, sol):
    rows = 0
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
        for n in range(1, 1001):
            a = sol.get(n)
            if a is None:
                a = grid(n)
            for i, (x, y, t) in enumerate(a):
                w.writerow([rows, n, i, fmt(x), fmt(y), fmt(t)])
                rows += 1


def finalize(P):
    """Return (P', L) passing the official verify, scaling centres apart slightly if contacts are
    undecidable; (None, None) if no small scaling works."""
    P = np.asarray(P, float)
    ok, L = verify(P)
    if ok:
        return P, L
    c = P[:, :2].mean(0)
    for e in (1e-11, 1e-10, 1e-9, 1e-8, 1e-7):
        Q = P.copy()
        Q[:, :2] = c + (P[:, :2] - c) * (1 + e)
        ok, L = verify(Q)
        if ok:
            return Q, L
    return None, None
