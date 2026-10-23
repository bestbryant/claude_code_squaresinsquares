"""Sequential linear programming polish for a packing of unit squares.

Variables per square: (dx, dy, dtheta); plus the container side L (box is [0, L]^2).
Non-overlap of a near pair (a, b) is linearised on the currently best separating
edge-normal axis u of one of the two squares: every corner of b must satisfy
u . (corner - c_a) >= 0.5 + margin.  Steps are accepted only if the true (non-linear)
configuration is feasible and L decreases; otherwise the trust region shrinks.
"""
import math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial import cKDTree

V = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def corner_arrays(P):
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    # rotated offsets: (n,4,2)
    ox = c[:, None] * V[None, :, 0] - s[:, None] * V[None, :, 1]
    oy = s[:, None] * V[None, :, 0] + c[:, None] * V[None, :, 1]
    return ox, oy


def side(P):
    ox, oy = corner_arrays(P)
    X = P[:, 0:1] + ox
    Y = P[:, 1:2] + oy
    return max(X.max() - X.min(), Y.max() - Y.min())


def normalize(P):
    ox, oy = corner_arrays(P)
    X = P[:, 0:1] + ox
    Y = P[:, 1:2] + oy
    Q = P.copy()
    Q[:, 0] -= X.min()
    Q[:, 1] -= Y.min()
    return Q


def pair_axes(P, i, j, ox, oy):
    """Best separating axis among edge normals of i and j. Returns (gap, owner, other, ux, uy)."""
    best = None
    for a, b in ((i, j), (j, i)):
        t = P[a, 2]
        for m in range(2):
            ang = t + m * math.pi / 2
            ux, uy = math.cos(ang), math.sin(ang)
            d = (P[b, 0] - P[a, 0]) * ux + (P[b, 1] - P[a, 1]) * uy
            if d < 0:
                ux, uy, d = -ux, -uy, -d
            proj = d + (ox[b] * ux + oy[b] * uy).min()
            gap = proj - 0.5
            if best is None or gap > best[0]:
                best = (gap, a, b, ux, uy)
    return best


def min_gap(P, pairs=None):
    ox, oy = corner_arrays(P)
    if pairs is None:
        tree = cKDTree(P[:, :2])
        pairs = tree.query_pairs(math.sqrt(2) + 1e-9, output_type='ndarray')
    g = np.inf
    for i, j in pairs:
        g = min(g, pair_axes(P, i, j, ox, oy)[0])
    return g


def pair_gaps_vec(P, pairs):
    """Vectorised best separating gap for an array of pairs (m,2)."""
    if len(pairs) == 0:
        return np.zeros(0)
    ox, oy = corner_arrays(P)
    best = np.full(len(pairs), -np.inf)
    for a_col, b_col in ((0, 1), (1, 0)):
        a = pairs[:, a_col]
        b = pairs[:, b_col]
        for m in range(2):
            ang = P[a, 2] + m * math.pi / 2
            ux, uy = np.cos(ang), np.sin(ang)
            d = (P[b, 0] - P[a, 0]) * ux + (P[b, 1] - P[a, 1]) * uy
            sgn = np.where(d < 0, -1.0, 1.0)
            ux, uy, d = ux * sgn, uy * sgn, d * sgn
            proj = d + (ox[b] * ux[:, None] + oy[b] * uy[:, None]).min(1)
            best = np.maximum(best, proj - 0.5)
    return best


def feasible(P, L, margin=0.0, tol=0.0):
    ox, oy = corner_arrays(P)
    X = P[:, 0:1] + ox
    Y = P[:, 1:2] + oy
    if X.min() < -tol or Y.min() < -tol or X.max() > L + tol or Y.max() > L + tol:
        return False
    tree = cKDTree(P[:, :2])
    pairs = tree.query_pairs(math.sqrt(2) + 1e-9, output_type='ndarray')
    if len(pairs) == 0:
        return True
    return pair_gaps_vec(P, pairs).min() >= margin - tol


def lp_step(P, L, t_xy, t_th, margin, fix_angle=None, eps_obj=1e-9):
    n = len(P)
    ox, oy = corner_arrays(P)
    X = P[:, 0:1] + ox
    Y = P[:, 1:2] + oy
    nv = 3 * n + 1
    iL = 3 * n
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    # derivative of corner offsets wrt theta: perp(o) = (-oy, ox)
    dX = -oy
    dY = ox
    near = 2 * t_xy + 1.0 * t_th + 1e-6
    # containment
    for i in range(n):
        for k in range(4):
            # x >= 0 :  -(x + dx + dXdθ) <= 0  ->  -dx - dX dθ <= X
            if X[i, k] < near + 0.0:
                rows += [r, r]; cols += [3 * i, 3 * i + 2]; vals += [-1.0, -dX[i, k]]; rhs.append(X[i, k]); r += 1
            if Y[i, k] < near:
                rows += [r, r]; cols += [3 * i + 1, 3 * i + 2]; vals += [-1.0, -dY[i, k]]; rhs.append(Y[i, k]); r += 1
            # x <= L: x + dx + dX dθ - L' <= 0   (L' variable absolute)
            if X[i, k] > L - near - 0.5:
                rows += [r, r, r]; cols += [3 * i, 3 * i + 2, iL]; vals += [1.0, dX[i, k], -1.0]; rhs.append(-X[i, k]); r += 1
            if Y[i, k] > L - near - 0.5:
                rows += [r, r, r]; cols += [3 * i + 1, 3 * i + 2, iL]; vals += [1.0, dY[i, k], -1.0]; rhs.append(-Y[i, k]); r += 1
    # pairs
    tree = cKDTree(P[:, :2])
    pairs = tree.query_pairs(math.sqrt(2) + 2 * t_xy + t_th * 1.5 + 1e-6, output_type='ndarray')
    for i, j in pairs:
        gap, a, b, ux, uy = pair_axes(P, i, j, ox, oy)
        if gap > 2 * t_xy + 1.5 * t_th + 1e-6:
            continue
        vx, vy = -uy, ux  # du/dθa
        for k in range(4):
            px = P[b, 0] + ox[b, k] - P[a, 0]
            py = P[b, 1] + oy[b, k] - P[a, 1]
            f = ux * px + uy * py
            if f - 0.5 > 2 * t_xy + 1.5 * t_th + 1e-6:
                continue
            # f + (-u)·dc_a + u·dc_b + (v·p) dθa + (u·perp(o_bk)) dθb >= 0.5 + margin
            gA = vx * px + vy * py
            gB = ux * dX[b, k] + uy * dY[b, k]
            rows += [r] * 6
            cols += [3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2]
            vals += [ux, uy, -gA, -ux, -uy, -gB]
            rhs.append(f - 0.5 - margin)
            r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    c = np.zeros(nv)
    c[iL] = 1.0
    bounds = []
    for i in range(n):
        bounds += [(-t_xy, t_xy), (-t_xy, t_xy)]
        if fix_angle is not None and fix_angle[i]:
            bounds.append((0, 0))
        else:
            bounds.append((-t_th, t_th))
    bounds.append((0, None))
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method='highs')
    if res.status != 0:
        return None, None
    d = res.x
    Q = P.copy()
    Q[:, 0] += d[0:3 * n:3]
    Q[:, 1] += d[1:3 * n:3]
    Q[:, 2] += d[2:3 * n:3]
    return Q, d[iL]


def polish(P, iters=200, t_xy=0.05, t_th=0.05, margin=2e-12, min_t=1e-9, fix_axis_aligned=False,
           verbose=False, time_limit=None):
    import time
    t0 = time.time()
    P = normalize(np.asarray(P, dtype=float))
    L = side(P)
    fix = None
    if fix_axis_aligned:
        fix = np.abs(np.mod(P[:, 2] + math.pi / 4, math.pi / 2) - math.pi / 4) < 1e-12
    best = (L, P.copy())
    for it in range(iters):
        if time_limit and time.time() - t0 > time_limit:
            break
        Q, Lq = lp_step(P, L, t_xy, t_th, margin, fix)
        if Q is None:
            t_xy *= 0.5; t_th *= 0.5
            if t_xy < min_t:
                break
            continue
        Q = normalize(Q)
        Lr = side(Q)
        ok = feasible(Q, Lr, margin=0.5 * margin)
        if ok and Lr < L - 1e-15:
            gain = L - Lr
            P, L = Q, Lr
            if L < best[0]:
                best = (L, P.copy())
            if verbose:
                print(f'it {it} L={L:.12f} gain={gain:.3e} t={t_xy:.2e}', flush=True)
            if gain < 1e-13:
                break
            t_xy = min(t_xy * 1.5, 0.1); t_th = min(t_th * 1.5, 0.1)
        else:
            t_xy *= 0.3; t_th *= 0.3
            if t_xy < min_t:
                break
    return best[1], best[0]
