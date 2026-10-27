"""45-degree strips of tilted squares along an anti-diagonal fault.

Grid cells (i, j) with i + j < c0 sit at offset (0, 0), cells with i + j > c0 + w - 1 at offset
(f, f); the w diagonals in between (the corridor, optionally limited to a range of i) are removed
and replaced by `rows` rows of ~45-degree squares.  Per diagonal step the corridor frees w cells
and a row holds ~sqrt(2) squares, so w=4, rows=3 gains ~0.243 per step once f >= ~0.605.
"""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from relax import relax
from fault_opt import squeeze
from slp import side, normalize


def build(K, f, c0, w=4, rows=3, extra=0, irange=None, rng=None, jitter=0.0):
    rng = rng or np.random.default_rng(0)
    cells = []
    removed = 0
    lo, hi = irange if irange else (-1, K + 1)
    for i in range(K):
        for j in range(K):
            s = i + j
            if c0 <= s < c0 + w and lo <= i < hi:
                removed += 1
                continue
            if s < c0 + w and not (s >= c0):
                cells.append((i + 0.5, j + 0.5, 0.0))
            elif s >= c0 + w:
                cells.append((i + 0.5 + f, j + 0.5 + f, 0.0))
            else:  # corridor cell outside irange: keep in the lower-left continent
                cells.append((i + 0.5, j + 0.5, 0.0))
    nt = removed + extra
    # rows: lines x + y = mid + k*sqrt(2), k centred
    mid = c0 + w / 2 + f
    ks = np.arange(rows) - (rows - 1) / 2
    S = K + f
    segs = []
    for k in ks:
        cst = mid + k * math.sqrt(2)
        # param u along (1,-1)/sqrt2: x = cst/2 + u/sqrt2, y = cst/2 - u/sqrt2
        # keep the square (half-diagonal 0.71) inside the box
        m = 0.70
        umin = max((m - cst / 2) * math.sqrt(2), (cst / 2 - (S - m)) * math.sqrt(2))
        umax = min(((S - m) - cst / 2) * math.sqrt(2), (cst / 2 - m) * math.sqrt(2))
        if irange:
            # restrict to the corridor's i range
            umin = max(umin, (lo - cst / 2) * math.sqrt(2))
            umax = min(umax, (hi - cst / 2) * math.sqrt(2))
        segs.append((cst, umin, umax))
    lens = np.array([max(0.0, b - a) for _, a, b in segs])
    cnt = np.floor(lens / lens.sum() * nt).astype(int)
    while cnt.sum() < nt:
        cnt[np.argmax(lens / np.maximum(cnt, 1e-9) - 0)] += 1
    T = []
    for (cst, a, b), c in zip(segs, cnt):
        if c <= 0:
            continue
        us = np.linspace(a, b, c) if c > 1 else np.array([(a + b) / 2])
        for u in us:
            x = cst / 2 + u / math.sqrt(2); y = cst / 2 - u / math.sqrt(2)
            T.append((x, y, math.pi / 4 + jitter * rng.normal()))
    P = np.vstack([np.array(cells).reshape(-1, 3), np.array(T).reshape(-1, 3)])
    return P, removed


def make(K, extra, c0=None, w=4, rows=3, f0=0.72, irange=None, time_limit=120, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    if c0 is None:
        c0 = K - 1 - w // 2
    P, removed = build(K, f0, c0, w, rows, extra, irange, rng, jitter=0.02)
    n = len(P)
    tilt = np.arange(n - removed - extra, n)
    # first relax only the tilted squares in the big box
    S = K + f0
    P, E = relax(P, S, free=tilt, iters=3000, rounds=6, mu=1e-11)
    if E > 1e-24:
        # widen
        for f1 in (0.8, 0.9, 0.98):
            P2, removed = build(K, f1, c0, w, rows, extra, irange, rng, jitter=0.02)
            P2, E = relax(P2, K + f1, free=tilt, iters=3000, rounds=6, mu=1e-11)
            if E < 1e-24:
                P = P2
                break
        else:
            return None, None
    Q, L = squeeze(P, time_limit=time_limit, rng=rng, verbose=verbose)
    return Q, L


if __name__ == '__main__':
    K = int(sys.argv[1]); extra = int(sys.argv[2])
    Q, L = make(K, extra, verbose=True, time_limit=float(sys.argv[3]) if len(sys.argv) > 3 else 120)
    print(K, extra, len(Q) if Q is not None else None, L)
