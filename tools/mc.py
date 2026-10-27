"""Hard-square Monte Carlo compression (NPT-like) for gadgets / packings with rigid upright groups.

State: squares (x, y, theta); a box (w+f) x (h+f) (or square K+f).  Upright squares grouped by
lattice offset (gx, gy in {0,1}) translate rigidly with f; 'free' squares move by MC.
Moves: single free square translate/rotate (accept iff no overlap & inside box);
box move f -> f - d (groups shift), accepted iff no overlap; with pressure p also allow
expansion with probability exp(-p * dArea).
"""
import math, sys, os
import numpy as np
import numba as nb

sys.path.insert(0, os.path.dirname(__file__))


@nb.njit(cache=True)
def corners(x, y, t):
    c = math.cos(t); s = math.sin(t)
    X = np.empty(4); Y = np.empty(4)
    vx = (0.5, -0.5, -0.5, 0.5); vy = (0.5, 0.5, -0.5, -0.5)
    for k in range(4):
        X[k] = x + c * vx[k] - s * vy[k]
        Y[k] = y + s * vx[k] + c * vy[k]
    return X, Y


@nb.njit(cache=True)
def overlap(x1, y1, t1, x2, y2, t2, tol):
    dx = x2 - x1; dy = y2 - y1
    if dx * dx + dy * dy >= 2.0:
        return False
    X1, Y1 = corners(x1, y1, t1)
    X2, Y2 = corners(x2, y2, t2)
    for which in range(2):
        t = t1 if which == 0 else t2
        cx = x1 if which == 0 else x2
        cy = y1 if which == 0 else y2
        for m in range(2):
            a = t + m * math.pi / 2
            ux = math.cos(a); uy = math.sin(a)
            # projections
            mn1 = 1e9; mx1 = -1e9; mn2 = 1e9; mx2 = -1e9
            for k in range(4):
                p = X1[k] * ux + Y1[k] * uy
                mn1 = min(mn1, p); mx1 = max(mx1, p)
                q = X2[k] * ux + Y2[k] * uy
                mn2 = min(mn2, q); mx2 = max(mx2, q)
            if mx1 <= mn2 + tol or mx2 <= mn1 + tol:
                return False
    return True


@nb.njit(cache=True)
def inside(x, y, t, W, H, tol):
    X, Y = corners(x, y, t)
    for k in range(4):
        if X[k] < -tol or Y[k] < -tol or X[k] > W + tol or Y[k] > H + tol:
            return False
    return True


@nb.njit(cache=True)
def any_overlap_with(P, i, x, y, t, tol):
    for j in range(P.shape[0]):
        if j != i and overlap(x, y, t, P[j, 0], P[j, 1], P[j, 2], tol):
            return True
    return False


@nb.njit(cache=True)
def run(P, free, gx, gy, w, h, f, steps, dxy, dth, pressure, fmin_stop, seed, tol):
    np.random.seed(seed)
    n = P.shape[0]
    nf = free.shape[0]
    acc = 0
    best_f = f
    best_P = P.copy()
    for it in range(steps):
        if np.random.random() < 0.98 or nf == 0:
            i = free[np.random.randint(nf)]
            x = P[i, 0] + dxy * (2 * np.random.random() - 1)
            y = P[i, 1] + dxy * (2 * np.random.random() - 1)
            t = P[i, 2] + dth * (2 * np.random.random() - 1)
            if inside(x, y, t, w + f, h + f, tol) and not any_overlap_with(P, i, x, y, t, tol):
                P[i, 0] = x; P[i, 1] = y; P[i, 2] = t
                acc += 1
        else:
            # box move
            d = 0.01 * dxy * (2 * np.random.random() - 1.2)
            nf_ = f + d
            if d > 0 and np.random.random() > math.exp(-pressure * d):
                continue
            Q = P.copy()
            for i in range(n):
                Q[i, 0] += gx[i] * d
                Q[i, 1] += gy[i] * d
            ok = True
            for i in range(n):
                if not inside(Q[i, 0], Q[i, 1], Q[i, 2], w + nf_, h + nf_, tol):
                    ok = False
                    break
            if ok:
                for i in range(n):
                    for j in range(i + 1, n):
                        if overlap(Q[i, 0], Q[i, 1], Q[i, 2], Q[j, 0], Q[j, 1], Q[j, 2], tol):
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                P[:, :] = Q
                f = nf_
                if f < best_f:
                    best_f = f
                    best_P = P.copy()
                    if best_f < fmin_stop:
                        break
    return best_P, best_f, acc


@nb.njit(cache=True)
def run_box(P, S, steps, dxy, dth, pressure, seed, tol, box_every):
    """All squares free in [0,S]^2; affine box moves (scale centres).  Returns best (P, S)."""
    np.random.seed(seed)
    n = P.shape[0]
    best_S = S
    best_P = P.copy()
    acc = 0
    for it in range(steps):
        if it % box_every != 0:
            i = np.random.randint(n)
            x = P[i, 0] + dxy * (2 * np.random.random() - 1)
            y = P[i, 1] + dxy * (2 * np.random.random() - 1)
            t = P[i, 2] + dth * (2 * np.random.random() - 1)
            if inside(x, y, t, S, S, tol) and not any_overlap_with(P, i, x, y, t, tol):
                P[i, 0] = x; P[i, 1] = y; P[i, 2] = t
                acc += 1
        else:
            d = 0.002 * dxy * (2 * np.random.random() - 1.2) * S
            if d > 0 and np.random.random() > math.exp(-pressure * d):
                continue
            S2 = S + d
            lam = S2 / S
            Q = P.copy()
            Q[:, 0] *= lam; Q[:, 1] *= lam
            ok = True
            for i in range(n):
                if not inside(Q[i, 0], Q[i, 1], Q[i, 2], S2, S2, tol):
                    ok = False
                    break
            if ok and d < 0:
                for i in range(n):
                    for j in range(i + 1, n):
                        if overlap(Q[i, 0], Q[i, 1], Q[i, 2], Q[j, 0], Q[j, 1], Q[j, 2], tol):
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                P[:, :] = Q
                S = S2
                if S < best_S:
                    best_S = S
                    best_P = P.copy()
    return best_P, best_S, acc
