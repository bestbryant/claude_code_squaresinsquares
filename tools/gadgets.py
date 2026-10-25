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


import numba as nb

NEG = -1000000000


@nb.njit(cache=True)
def _add_item4(D, w, h, v):
    """D[layer, x, y], layer = 2*anycol + anyrow. Item may pay a column or a row buffer."""
    N = D.shape[1]
    for x in range(N):
        for y in range(N):
            for l in range(4):
                c, r = l // 2, l % 2
                # column buffer: cost (w+1, h) -> layer (1, r)
                if x >= w + 1 and y >= h:
                    val = D[l, x - w - 1, y - h]
                    if val > NEG:
                        t = 2 + r
                        if val + v > D[t, x, y]:
                            D[t, x, y] = val + v
                if x >= w and y >= h + 1:
                    val = D[l, x - w, y - h - 1]
                    if val > NEG:
                        t = 2 * c + 1
                        if val + v > D[t, x, y]:
                            D[t, x, y] = val + v


@nb.njit(cache=True)
def _best_K(D, K):
    b = 0
    for x in range(min(K + 2, D.shape[1])):
        for y in range(min(K + 2, D.shape[1])):
            for l in range(4):
                c, r = l // 2, l % 2
                ok = (c == 1 and x <= K + 1 and y <= K) or (r == 1 and x <= K and y <= K + 1)
                if ok and D[l, x, y] > b:
                    b = D[l, x, y]
    return b


@nb.njit(cache=True)
def _dp4_arg(ws, hs, vs, N):
    D = np.full((4, N, N), NEG, np.int64)
    A = np.full((4, N, N), -1, np.int64)
    D[0, 0, 0] = 0
    for x in range(N):
        for y in range(N):
            for l in range(4):
                if D[l, x, y] == NEG:
                    continue
                c, r = l // 2, l % 2
                for i in range(len(ws)):
                    for o in range(2):
                        a = ws[i] + (1 - o)
                        b = hs[i] + o
                        X, Y = x + a, y + b
                        t = (2 + r) if o == 0 else (2 * c + 1)
                        if X < N and Y < N and D[l, x, y] + vs[i] > D[t, X, Y]:
                            D[t, X, Y] = D[l, x, y] + vs[i]
                            A[t, X, Y] = (i * 2 + o) * 4 + l
    return D, A


def frontier(G, Kmin=4, Kmax=31):
    N = Kmax + 3
    items = sorted(((f, w, h, dm) for (w, h, dm), (f, _) in G.items()))
    D = np.full((4, N, N), NEG, np.int64)
    D[0, 0, 0] = 0
    best = {K: 0 for K in range(Kmin, Kmax + 1)}
    rec = {}
    for f, w, h, dm in items:
        if w > Kmax or h > Kmax:
            continue
        _add_item4(D, w, h, dm)
        for K in range(max(Kmin, max(w, h)), Kmax + 1):
            v = _best_K(D, K)
            if v > best[K]:
                for d in range(best[K] + 1, v + 1):
                    rec[(K, d)] = f
                best[K] = v
    return rec


def reconstruct(G, K, delta, f):
    keys = [k for k, v in G.items() if v[0] <= f + 1e-15 and k[0] <= K and k[1] <= K]
    ws = np.array([k[0] for k in keys], np.int64)
    hs = np.array([k[1] for k in keys], np.int64)
    vs = np.array([k[2] for k in keys], np.int64)
    N = K + 3
    D, A = _dp4_arg(ws, hs, vs, N)
    best = None
    for x in range(N):
        for y in range(N):
            for l in range(4):
                c, r = l // 2, l % 2
                ok = (c == 1 and x <= K + 1 and y <= K) or (r == 1 and x <= K and y <= K + 1)
                if ok and D[l, x, y] >= delta and (best is None or D[l, x, y] < D[best]):
                    best = (l, x, y)
    if best is None:
        return None
    chain = []
    l, x, y = best
    while (x, y) != (0, 0):
        a = A[l, x, y]
        io, pl = a // 4, a % 4
        i, o = io // 2, io % 2
        w, h, dm = keys[i]
        chain.append((w, h, dm, o))
        x -= w + (1 - o)
        y -= h + o
        l = pl
    chain.reverse()
    # the last gadget needs no buffer: move its buffer onto an earlier gadget is unnecessary;
    # build() simply ignores the final buffer.
    return chain

def build(chain, K, f, G, sols):
    """Place chain gadgets top-left -> bottom-right; continents fill the rest. Returns packing."""
    chain = list(chain)
    cols = sum(c[0] for c in chain) + sum(1 for c in chain if c[3] == 0)
    rows = sum(c[1] for c in chain) + sum(1 for c in chain if c[3] == 1)
    # the last gadget's buffer is dropped: make sure it is the one that overflows
    if cols > K:
        j = max(i for i, c in enumerate(chain) if c[3] == 0)
        chain.append(chain.pop(j))
    elif rows > K:
        j = max(i for i, c in enumerate(chain) if c[3] == 1)
        chain.append(chain.pop(j))
    out = []
    ox, top = 0, K
    placements = []
    for idx, (w, h, dm, o) in enumerate(chain):
        oy = top - h
        if oy < 0 or ox + w > K:
            return None
        placements.append((ox, oy, w, h, dm))
        if o == 0:
            ox, top = ox + w + 1, oy
        else:
            ox, top = ox + w, oy - 1
    split = np.full(K, -1)   # A rows < a[c]; B rows >= b[c]
    a = np.zeros(K, int); b = np.zeros(K, int)
    for (ox_, oy_, w, h, dm) in placements:
        a[ox_:ox_ + w] = oy_
        b[ox_:ox_ + w] = oy_ + h
        split[ox_:ox_ + w] = 1
    last_oy = placements[-1][1]
    for c in range(K):
        if split[c] == 1:
            continue
        # buffer column (after gadget m) or trailing column: A below previous gadget's bottom
        prv = [p for p in placements if p[0] + p[2] <= c]
        v = prv[-1][1] if prv else K
        a[c] = b[c] = v
    for (ox_, oy_, w, h, dm) in placements:
        fg, (n, k) = G[(w, h, dm)]
        P = crop(sols[n] if not isinstance(n, str) else sols[n], k, w, h, fg)
        ext = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
        r = np.maximum(0, np.maximum(P[:, 0] + ext - w, P[:, 1] + ext - h))
        P = P[np.argsort(r)[:w * h + dm]].copy()
        if len(P) < w * h + dm:
            return None
        P[:, 0] += ox_; P[:, 1] += oy_
        out.append(P)
    cells = []
    for c in range(K):
        for j in range(a[c]):
            cells.append((c + 0.5, j + 0.5, 0.0))
        for j in range(b[c], K):
            cells.append((c + 0.5 + f, j + 0.5 + f, 0.0))
    out.append(np.array(cells).reshape(-1, 3))
    return np.vstack(out)
