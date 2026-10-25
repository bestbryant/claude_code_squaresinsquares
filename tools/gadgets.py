"""Rectangular gadgets + anti-diagonal fault composition.

A gadget (w, h, dm, f) is a packing of w*h+dm unit squares inside [0, w+f] x [0, h+f].
Composition (Deotte's fault-line construction): box side K+f, lower-left continent on Z^2,
upper-right continent on Z^2+(f,f); gadgets placed along a monotone chain from the top-left
to the bottom-right, consecutive gadgets separated by one buffer column or row.
Total squares K^2 + sum(dm), side K + max f.

Extraction: corner crops of existing packings under the 8 symmetries; for a footprint (w,h)
square i needs f >= r_i = max(0, xhi_i - w, yhi_i - h); the k-th order statistic of r gives the
best f for w*h+dm = k squares.
"""
import math
import numpy as np

V = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def sym(P, k):
    """Apply symmetry k in 0..7 (dihedral group) to packing P (centres + angle), then normalise."""
    Q = P.copy()
    if k & 1:
        Q[:, 0] = -Q[:, 0]; Q[:, 2] = -Q[:, 2]
    if k & 2:
        Q[:, 1] = -Q[:, 1]; Q[:, 2] = -Q[:, 2]
    if k & 4:
        Q = Q[:, [1, 0, 2]]; Q[:, 2] = math.pi / 2 - Q[:, 2]
    c, s = np.cos(Q[:, 2]), np.sin(Q[:, 2])
    X = Q[:, 0:1] + c[:, None] * V[:, 0] - s[:, None] * V[:, 1]
    Y = Q[:, 1:2] + s[:, None] * V[:, 0] + c[:, None] * V[:, 1]
    Q[:, 0] -= X.min(); Q[:, 1] -= Y.min()
    return Q, X.max(1) - X.min(), Y.max(1) - Y.min()


def extract(P, maxdim=31, fmax=0.999):
    """Yield (w, h, dm, f, idx_symmetry, kept_indices) best crops of packing P."""
    out = {}
    n = len(P)
    for k in range(8):
        Q, xhi, yhi = sym(P, k)
        L = max(xhi.max(), yhi.max())
        K = int(math.floor(L + 1e-12))
        for w in range(1, min(K, maxdim) + 1):
            hs = np.arange(1, min(K, maxdim) + 1)
            # r[h, i]
            r = np.maximum(0.0, np.maximum(xhi[None, :] - w, yhi[None, :] - hs[:, None]))
            rs = np.sort(r, axis=1)
            for hi, h in enumerate(hs):
                base = w * h
                if base >= n:
                    break
                vals = rs[hi, base:]
                ok = vals < fmax
                if not ok.any():
                    continue
                for dm in range(1, int(ok.sum()) + 1):
                    f = vals[dm - 1]
                    if f >= fmax:
                        break
                    key = (w, h, dm)
                    if key not in out or f < out[key][0] - 1e-13:
                        out[key] = (f, k)
    return out


def crop(P, k, w, h, f):
    Q, xhi, yhi = sym(P, k)
    keep = (xhi <= w + f + 1e-12) & (yhi <= h + f + 1e-12)
    return Q[keep]


def pareto(gads):
    """gads: dict (w,h,dm) -> (f, src). Remove dominated entries."""
    items = sorted(gads.items(), key=lambda kv: kv[1][0])
    kept = {}
    best = {}  # (w,h) -> max dm seen at lower f
    for (w, h, dm), v in items:
        dom = False
        for (w2, h2), d2 in best.items():
            if w2 <= w and h2 <= h and d2 >= dm:
                dom = True
                break
        if not dom:
            kept[(w, h, dm)] = v
            if best.get((w, h), 0) < dm:
                best[(w, h)] = dm
    return kept


def compose_dp(items, Kmax=31):
    """items: list of (w, h, dm). Unbounded chain knapsack.
    Returns dp[cw][ch] = max dm-sum for chains whose footprints+buffers use cw columns, ch rows,
    where every gadget but the last pays one buffer (column or row).  We realise this by
    allowing capacity K+1 in one dimension."""
    N = Kmax + 2
    NEG = -10 ** 9
    dp = np.full((N, N), NEG, dtype=np.int64)
    arg = np.full((N, N), -1, dtype=np.int64)
    dp[0, 0] = 0
    costs = []
    for gi, (w, h, dm) in enumerate(items):
        costs.append((w + 1, h, dm, gi))
        costs.append((w, h + 1, dm, gi))
    for cw in range(N):
        for ch in range(N):
            if dp[cw, ch] == NEG:
                continue
            for a, b, dm, gi in costs:
                x, y = cw + a, ch + b
                if x < N and y < N and dp[cw, ch] + dm > dp[x, y]:
                    dp[x, y] = dp[cw, ch] + dm
                    arg[x, y] = len(costs) and (gi * 2 + (0 if a == items[gi][0] + 1 else 1))
    return dp, arg


def best_for_K(dp, K):
    a = dp[:K + 2, :K + 1].max()
    b = dp[:K + 1, :K + 2].max()
    return max(a, b)
