"""Constructive band packings.

Container S = K + f.  An upright grid core plus bands of tilted 'columns' along the
top and right walls.  A column is a stack of unit squares tilted by theta, stacked
along e2 = (-sin t, cos t); neighbouring columns are horizontal translates by 1/cos t.
"""
import math
import numpy as np

EPS = 1e-9


def fill_rect_columns(x0, x1, y0, y1, t, phase=0.0, align='bottom'):
    """Fill [x0,x1]x[y0,y1] with tilted columns (tilt t>0). Returns (m,3) array."""
    c, s = math.cos(t), math.sin(t)
    hx = 0.5 * (c + s)  # half extent of a tilted square in x and y
    e2 = np.array([-s, c])
    out = []
    dx = 1.0 / c
    H = y1 - y0
    # base square center y
    if align == 'bottom':
        by = y0 + hx
    else:
        # top aligned: the column top square touches y1; compute count per column below
        by = None
    # x range of base centers: column squares drift left by s per step; allow bases to the right of x1
    nmax = int(H / c) + 2
    xs = np.arange(x0 + hx + phase, x1 + nmax * s + dx, dx)
    for bx in xs:
        if align == 'bottom':
            j = np.arange(nmax)
            cx = bx + j * e2[0]
            cy = by + j * e2[1]
        else:
            j = np.arange(nmax)
            cx = bx - j * e2[0]
            cy = (y1 - hx) - j * e2[1]
        ok = (cx - hx >= x0 - EPS) & (cx + hx <= x1 + EPS) & (cy - hx >= y0 - EPS) & (cy + hx <= y1 + EPS)
        for a, b in zip(cx[ok], cy[ok]):
            out.append((a, b, t))
    return np.array(out).reshape(-1, 3)


def grid_rect(x0, x1, y0, y1):
    nx = int(math.floor(x1 - x0 + EPS))
    ny = int(math.floor(y1 - y0 + EPS))
    xs = x0 + 0.5 + np.arange(nx)
    ys = y0 + 0.5 + np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]


def transpose(P):
    Q = P[:, [1, 0, 2]].copy()
    Q[:, 2] = math.pi / 2 - Q[:, 2]
    return Q


def two_band(K, f, h1, t1, h2, t2, ph1=0.0, ph2=0.0):
    """Top band of height h1+f across full width, right band of width h2+f below it."""
    S = K + f
    core = grid_rect(0, K - h2, 0, K - h1)
    top = fill_rect_columns(0, S, K - h1, S, t1, ph1) if h1 > 0 else np.zeros((0, 3))
    # right band: transpose a horizontal-band problem: region x in [K-h2, S], y in [0, K-h1]
    if h2 > 0:
        r = fill_rect_columns(0, K - h1, K - h2, S, t2, ph2)  # in transposed coords (x<->y)
        right = transpose(r)
    else:
        right = np.zeros((0, 3))
    return np.vstack([core, top, right]), S


def resolve_conflicts(P, tol=1e-9):
    """Greedily drop squares until no pair overlaps (gap < -tol)."""
    from scipy.spatial import cKDTree
    from slp import pair_gaps_vec
    P = np.asarray(P)
    if len(P) < 2:
        return P
    pairs = cKDTree(P[:, :2]).query_pairs(math.sqrt(2) + 1e-9, output_type='ndarray')
    if len(pairs) == 0:
        return P
    g = pair_gaps_vec(P, pairs)
    bad = pairs[g < -tol]
    keep = np.ones(len(P), bool)
    while len(bad):
        cnt = np.bincount(bad.ravel(), minlength=len(P))
        i = cnt.argmax()
        keep[i] = False
        bad = bad[(bad[:, 0] != i) & (bad[:, 1] != i)]
    return P[keep]


def in_box(P, S, tol=1e-9):
    from slp import corner_arrays
    ox, oy = corner_arrays(P)
    X = P[:, 0:1] + ox
    Y = P[:, 1:2] + oy
    ok = (X.min(1) >= -tol) & (Y.min(1) >= -tol) & (X.max(1) <= S + tol) & (Y.max(1) <= S + tol)
    return P[ok]


def fill_holes(P, S, step=0.05, tol=1e-9):
    """Greedy: add upright unit squares wherever they fit (scan bottom-left first)."""
    from slp import corner_arrays
    P = np.asarray(P)
    ox, oy = corner_arrays(P)
    X = P[:, 0:1] + ox
    Y = P[:, 1:2] + oy
    # occupancy raster for quick rejection
    res = 0.05
    N = int(math.ceil(S / res)) + 1
    from matplotlib.path import Path
    occ = np.zeros((N, N), bool)
    gx = (np.arange(N) + 0.5) * res
    GX, GY = np.meshgrid(gx, gx, indexing='ij')
    pts = np.c_[GX.ravel(), GY.ravel()]
    for i in range(len(P)):
        x0, x1 = X[i].min(), X[i].max()
        y0, y1 = Y[i].min(), Y[i].max()
        i0, i1 = max(int(x0 / res) - 1, 0), min(int(x1 / res) + 2, N)
        j0, j1 = max(int(y0 / res) - 1, 0), min(int(y1 / res) + 2, N)
        sub = np.c_[GX[i0:i1, j0:j1].ravel(), GY[i0:i1, j0:j1].ravel()]
        m = Path(np.c_[X[i], Y[i]]).contains_points(sub, radius=-1e-6)
        occ[i0:i1, j0:j1] |= m.reshape(i1 - i0, j1 - j0)
    added = []
    cur = P
    # candidate positions: free raster cells
    cand = []
    for a in np.arange(0.5, S - 0.5 + 1e-12, step):
        for b in np.arange(0.5, S - 0.5 + 1e-12, step):
            i0, i1 = int((a - 0.5) / res) + 1, int((a + 0.5) / res) - 1
            j0, j1 = int((b - 0.5) / res) + 1, int((b + 0.5) / res) - 1
            if not occ[i0:i1, j0:j1].any():
                cand.append((a, b))
    if not cand:
        return P
    from slp import pair_gaps_vec
    for a, b in sorted(cand, key=lambda t: (t[1] + t[0], t[0])):
        q = np.array([[a, b, 0.0]])
        d = np.hypot(cur[:, 0] - a, cur[:, 1] - b)
        near = np.where(d < math.sqrt(2) + 1e-9)[0]
        if len(near):
            T = np.vstack([cur[near], q])
            pr = np.c_[np.arange(len(near)), np.full(len(near), len(near))]
            if pair_gaps_vec(T, pr).min() < -tol:
                continue
        cur = np.vstack([cur, q])
    return cur


def stick_theta(m, W):
    """Smallest tilt t with m cos t + sin t <= W (sticks of m squares across width W)."""
    from scipy.optimize import brentq
    if m <= W:
        return 0.0
    g = lambda t: m * math.cos(t) + math.sin(t) - W
    tm = math.atan(1.0 / m)  # maximiser of g
    if g(tm) < 0:
        return None
    return brentq(g, tm, math.pi / 2) + 1e-12


def mirror_x(P, x0, x1):
    Q = P.copy()
    Q[:, 0] = x0 + x1 - Q[:, 0]
    Q[:, 2] = -Q[:, 2]
    return Q


def band(x0, x1, y0, y1, vertical_sticks=True, lean=1, phase=0.0, m=None, align='bottom'):
    """Fill rectangle with sticks crossing its short side. vertical_sticks: sticks span y-extent."""
    if not vertical_sticks:
        Q = band(y0, y1, x0, x1, True, lean, phase, m, align)
        return transpose(Q)
    W = y1 - y0
    if m is None:
        m = int(math.floor(W + 1e-9)) + 1
    t = stick_theta(m, W)
    if t is None or t == 0.0:
        return grid_rect(x0, x1, y0, y1)
    P = fill_rect_columns(x0, x1, y0, y1, t, phase, align)
    if lean < 0:
        P = mirror_x(P, x0, x1)
    return P


def pinwheel(K, f, h1, h2, l1=1, l2=1, p1=0.0, p2=0.0, fill=True):
    S = K + f
    parts = [grid_rect(0, h1, 0, h2),
             band(h1, S, 0, h2 + f, True, l1, p1),
             band(0, h1 + f, h2, S, False, l2, p2),
             grid_rect(h1 + f, S, h2 + f, S)]
    P = np.vstack([p for p in parts if len(p)])
    P = in_box(P, S)
    P = resolve_conflicts(P)
    if fill:
        P = fill_holes(P, S)
    return P, S


def lshape(K, f, h1, h2, l1=1, l2=1, p1=0.0, p2=0.0, fill=True, m1=None, m2=None):
    """Top band (full width, vertical sticks) + right band below it (horizontal sticks) + grid core."""
    S = K + f
    parts = [grid_rect(0, K - h2, 0, K - h1),
             band(0, S, K - h1, S, True, l1, p1, m1),
             band(K - h2, S, 0, K - h1, False, l2, p2, m2)]
    P = np.vstack([p for p in parts if len(p)])
    P = in_box(P, S)
    P = resolve_conflicts(P)
    if fill:
        P = fill_holes(P, S)
    return P, S


def grow(P, S, tol=1e-9, max_rounds=20, slide=(0.0,)):
    """Greedy: add squares at edge-adjacent lattice positions of existing squares (same angle),
    optionally slid along the shared edge by fractions in `slide`. Repeats until nothing fits."""
    from slp import pair_gaps_vec
    from scipy.spatial import cKDTree
    P = np.asarray(P, dtype=float)
    for _ in range(max_rounds):
        added = 0
        tree = cKDTree(P[:, :2])
        cands = []
        for i in range(len(P)):
            x, y, t = P[i]
            c, s = math.cos(t), math.sin(t)
            for ux, uy in ((c, s), (-c, -s), (-s, c), (s, -c)):
                for sl in slide:
                    vx, vy = -uy, ux
                    cands.append((x + ux + sl * vx, y + uy + sl * vy, t))
        cands = np.array(cands)
        # in box
        cands = in_box(cands, S, tol)
        for q in cands:
            idx = np.where(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) < math.sqrt(2) + 1e-9)[0]
            if len(idx):
                if np.min(np.hypot(P[idx, 0] - q[0], P[idx, 1] - q[1])) < 1 - 1e-9:
                    continue
                T = np.vstack([P[idx], q])
                pr = np.c_[np.arange(len(idx)), np.full(len(idx), len(idx))]
                if pair_gaps_vec(T, pr).min() < -tol:
                    continue
            P = np.vstack([P, q])
            added += 1
        if not added:
            break
    return P


def resolve_priority(groups, S, tol=1e-9):
    """groups: list of arrays in decreasing priority. A square is kept if it is inside the box and
    does not overlap any kept square of higher or equal priority (greedy in order)."""
    from slp import pair_gaps_vec
    kept = np.zeros((0, 3))
    for G in groups:
        G = in_box(np.asarray(G).reshape(-1, 3), S, tol)
        for q in G:
            if len(kept):
                d = np.hypot(kept[:, 0] - q[0], kept[:, 1] - q[1])
                idx = np.where(d < math.sqrt(2) + 1e-9)[0]
                if len(idx):
                    if d[idx].min() < 1 - 1e-9:
                        continue
                    T = np.vstack([kept[idx], q])
                    pr = np.c_[np.arange(len(idx)), np.full(len(idx), len(idx))]
                    if pair_gaps_vec(T, pr).min() < -tol:
                        continue
            kept = np.vstack([kept, q])
    return kept


def lcorner(K, f, h, R, ph_top, qext=3, order='rqt'):
    """L-band with lattice corner: right band rows (along e1) left-aligned at x=K-h, a pure lattice
    block (rows shifted by e2) above them up to the top wall, top band sticks floor-aligned at y=K-h."""
    S = K + f
    W = h + f
    m = h + 1
    t = stick_theta(m, W)
    c, s = math.cos(t), math.sin(t)
    p = 1 / c
    X0 = K - h
    core = grid_rect(0, X0, 0, X0)
    rows = []
    for r in range(R):
        y_r = r * p
        cx0 = X0 + 0.5 * (c + s)
        cy0 = y_r + 0.5 * (c + s)
        rows.append([(cx0 + i * c, cy0 + i * s, t) for i in range(m)])
    rb = np.array(rows).reshape(-1, 3)
    # lattice block above the top row: shift top row by q*e2, extended along e1 both ways
    top_row = np.array(rows[-1])
    Q = []
    for q in range(1, int(S) + 2):
        for i in range(-qext, m + qext):
            cx = top_row[0, 0] + i * c - q * s
            cy = top_row[0, 1] + i * s + q * c
            Q.append((cx, cy, t))
    Q = np.array(Q)
    top = band(-2, S, X0, S, True, 1, ph_top)  # lean left sticks, extend past the left wall then clip
    groups = {'r': rb, 'q': Q, 't': top}
    P = resolve_priority([core] + [groups[k] for k in order], S)
    return P, S


def lcorner2(K, f, h, R, qext=3, extra_shift=0.0):
    """As lcorner, but top band sticks phased so the last stick is flush with the lattice block."""
    S = K + f
    W = h + f
    m = h + 1
    t = stick_theta(m, W)
    c, s = math.cos(t), math.sin(t)
    p = 1 / c
    X0 = K - h
    core = grid_rect(0, X0, 0, X0)
    rows = []
    for r in range(R):
        y_r = r * p
        rows.append([(X0 + 0.5 * (c + s) + i * c, y_r + 0.5 * (c + s) + i * s, t) for i in range(m)])
    rb = np.array(rows).reshape(-1, 3)
    top_row = np.array(rows[-1])
    yL = (R - 1) * p + c
    Q = []
    for q in range(1, int(S / c) + 2):
        for i in range(-qext, m + qext):
            Q.append((top_row[0, 0] + i * c - q * s, top_row[0, 1] + i * s + q * c, t))
    Q = np.array(Q)
    xQ = X0 - (X0 - yL) * math.tan(t)  # Q left line at height Y0 = X0
    xk = xQ - p + extra_shift
    stks = []
    j = 0
    while xk - j * p > -W * math.tan(t) - 2:
        x = xk - j * p
        cx = x + 0.5 * (c - s); cy = X0 + 0.5 * (s + c)
        for i in range(m):
            stks.append((cx - i * s, cy + i * c, t))
        j += 1
    top = np.array(stks)
    P = resolve_priority([core, rb, Q, top], S)
    return P, S


def stick_ribbon(K, f, m, t, T, c0, gap=1e-9, u=None):
    """Ribbon of sticks (m squares stacked along e2 of tilt t) translated by T from c0, crossing the
    box; continent A at lattice offset (f,0) (lower-right), B at (0,f) (upper-left), greedy."""
    S = K + f
    c, s = math.cos(t), math.sin(t)
    e2 = np.array([-s, c]) if u is None else np.array(u)
    rib = []
    for j in range(-int(3 * S), int(3 * S)):
        base = np.array(c0) + j * np.array(T)
        for i in range(m):
            p = base + i * e2
            rib.append((p[0], p[1], t))
    rib = in_box(np.array(rib), S)
    # continents: A below the ribbon line, B above. Use the ribbon direction to decide sides.
    d = np.array(T) / np.hypot(*T)
    nrm = np.array([-d[1], d[0]])
    mid = np.array(c0) + (m - 1) / 2 * e2
    A = []; B = []
    for i in range(K):
        for j in range(K):
            pa = np.array([i + 0.5 + f, j + 0.5])
            pb = np.array([i + 0.5, j + 0.5 + f])
            if (pa - mid) @ nrm < 0:
                A.append((pa[0], pa[1], 0.0))
            if (pb - mid) @ nrm > 0:
                B.append((pb[0], pb[1], 0.0))
    P = resolve_priority([rib, np.array(A), np.array(B)], S)
    return P, S
