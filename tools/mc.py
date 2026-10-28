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


@nb.njit(cache=True)
def corners_s(x, y, t, h):
    c = math.cos(t); s = math.sin(t)
    X = np.empty(4); Y = np.empty(4)
    vx = (h, -h, -h, h); vy = (h, h, -h, -h)
    for k in range(4):
        X[k] = x + c * vx[k] - s * vy[k]
        Y[k] = y + s * vx[k] + c * vy[k]
    return X, Y


@nb.njit(cache=True)
def overlap_s(x1, y1, t1, h1, x2, y2, t2, h2, tol):
    dx = x2 - x1; dy = y2 - y1
    r = (h1 + h2) * 1.4143
    if dx * dx + dy * dy >= r * r:
        return False
    X1, Y1 = corners_s(x1, y1, t1, h1)
    X2, Y2 = corners_s(x2, y2, t2, h2)
    for which in range(2):
        t = t1 if which == 0 else t2
        for m in range(2):
            a = t + m * math.pi / 2
            ux = math.cos(a); uy = math.sin(a)
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
def grow_one(P, free, k, h0, W, H, steps, dxy, dth, seed, tol):
    """Free squares move; square k (half-size h0 initially) grows toward 0.5 whenever possible.
    Returns (P, h_final)."""
    np.random.seed(seed)
    n = P.shape[0]
    nf = free.shape[0]
    h = h0
    for it in range(steps):
        if it % 20 == 0 and h < 0.5:
            h2 = min(0.5, h + 0.002)
            ok = True
            X, Y = corners_s(P[k, 0], P[k, 1], P[k, 2], h2)
            for q in range(4):
                if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > H + tol:
                    ok = False
            if ok:
                for j in range(n):
                    if j != k and overlap_s(P[k, 0], P[k, 1], P[k, 2], h2, P[j, 0], P[j, 1], P[j, 2], 0.5, tol):
                        ok = False
                        break
            if ok:
                h = h2
            continue
        i = free[np.random.randint(nf)]
        x = P[i, 0] + dxy * (2 * np.random.random() - 1)
        y = P[i, 1] + dxy * (2 * np.random.random() - 1)
        t = P[i, 2] + dth * (2 * np.random.random() - 1)
        hi = h if i == k else 0.5
        X, Y = corners_s(x, y, t, hi)
        ok = True
        for q in range(4):
            if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > H + tol:
                ok = False
        if not ok:
            continue
        for j in range(n):
            if j != i:
                hj = h if j == k else 0.5
                if overlap_s(x, y, t, hi, P[j, 0], P[j, 1], P[j, 2], hj, tol):
                    ok = False
                    break
        if ok:
            P[i, 0] = x; P[i, 1] = y; P[i, 2] = t
        if h >= 0.5:
            break
    return P, h


@nb.njit(cache=True)
def grow_all(F, Q, h0, W, steps, dxy, dth, seed, tol, grow_every, dh):
    """LS growth: Q (m,3) free squares with common half-size h (start h0) among fixed F (k,3)
    unit squares, box [0,W]^2.  Returns (Q, h)."""
    np.random.seed(seed)
    m = Q.shape[0]; k = F.shape[0]
    h = h0
    for it in range(steps):
        if it % grow_every == 0 and h < 0.5:
            h2 = min(0.5, h + dh)
            ok = True
            for i in range(m):
                X, Y = corners_s(Q[i, 0], Q[i, 1], Q[i, 2], h2)
                for q in range(4):
                    if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > W + tol:
                        ok = False
                if not ok:
                    break
                for j in range(i + 1, m):
                    if overlap_s(Q[i, 0], Q[i, 1], Q[i, 2], h2, Q[j, 0], Q[j, 1], Q[j, 2], h2, tol):
                        ok = False
                        break
                if not ok:
                    break
                for j in range(k):
                    if overlap_s(Q[i, 0], Q[i, 1], Q[i, 2], h2, F[j, 0], F[j, 1], F[j, 2], 0.5, tol):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                h = h2
                dh = min(dh * 1.3, 0.004)
            else:
                dh *= 0.7
                if dh < 1e-7:
                    dh = 1e-7
            if h >= 0.5:
                break
            continue
        i = np.random.randint(m)
        x = Q[i, 0] + dxy * h * (2 * np.random.random() - 1)
        y = Q[i, 1] + dxy * h * (2 * np.random.random() - 1)
        t = Q[i, 2] + dth * (2 * np.random.random() - 1)
        X, Y = corners_s(x, y, t, h)
        ok = True
        for q in range(4):
            if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > W + tol:
                ok = False
        if not ok:
            continue
        for j in range(m):
            if j != i and overlap_s(x, y, t, h, Q[j, 0], Q[j, 1], Q[j, 2], h, tol):
                ok = False
                break
        if not ok:
            continue
        for j in range(k):
            if overlap_s(x, y, t, h, F[j, 0], F[j, 1], F[j, 2], 0.5, tol):
                ok = False
                break
        if ok:
            Q[i, 0] = x; Q[i, 1] = y; Q[i, 2] = t
    return Q, h


@nb.njit(cache=True)
def grow_push(P, isfree, k, h0, W, steps, dxy, seed, tol, maxc):
    """Grow square k (half-size h) among unit squares; moves are push-chain translations:
    move square i by d, every free square it then overlaps is pushed by d too (recursively).
    Rejected if the cluster hits a non-free square, a wall, or exceeds maxc squares."""
    np.random.seed(seed)
    n = P.shape[0]
    h = h0
    cl = np.empty(n, np.int64)
    inc = np.zeros(n, np.bool_)
    fidx = np.where(isfree)[0]
    nf = fidx.shape[0]
    dh = 0.002
    for it in range(steps):
        if it % 10 == 0:
            h2 = min(0.5, h + dh)
            ok = True
            X, Y = corners_s(P[k, 0], P[k, 1], P[k, 2], h2)
            for q in range(4):
                if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > W + tol:
                    ok = False
            if ok:
                for j in range(n):
                    if j != k and overlap_s(P[k, 0], P[k, 1], P[k, 2], h2, P[j, 0], P[j, 1], P[j, 2], 0.5, tol):
                        ok = False
                        break
            if ok:
                h = h2
                dh = min(dh * 1.5, 0.01)
                if h >= 0.5:
                    break
            else:
                dh = max(dh * 0.7, 1e-6)
            continue
        i = fidx[np.random.randint(nf)]
        if np.random.random() < 0.3 and i != k:
            # small rotation, single particle
            t = P[i, 2] + 0.3 * dxy * (2 * np.random.random() - 1)
            ok = True
            X, Y = corners_s(P[i, 0], P[i, 1], t, 0.5)
            for q in range(4):
                if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > W + tol:
                    ok = False
            if ok:
                for j in range(n):
                    hj = h if j == k else 0.5
                    if j != i and overlap_s(P[i, 0], P[i, 1], t, 0.5, P[j, 0], P[j, 1], P[j, 2], hj, tol):
                        ok = False
                        break
            if ok:
                P[i, 2] = t
            continue
        a = np.random.random() * 2 * math.pi
        r = dxy * np.random.random()
        dx = r * math.cos(a); dy = r * math.sin(a)
        # build cluster
        nc = 1; cl[0] = i; inc[i] = True
        ok = True
        p = 0
        while p < nc and ok:
            a_ = cl[p]; p += 1
            ha = h if a_ == k else 0.5
            xa = P[a_, 0] + dx; ya = P[a_, 1] + dy
            X, Y = corners_s(xa, ya, P[a_, 2], ha)
            for q in range(4):
                if X[q] < -tol or Y[q] < -tol or X[q] > W + tol or Y[q] > W + tol:
                    ok = False
            if not ok:
                break
            for j in range(n):
                if inc[j]:
                    continue
                hj = h if j == k else 0.5
                if overlap_s(xa, ya, P[a_, 2], ha, P[j, 0], P[j, 1], P[j, 2], hj, tol):
                    if not isfree[j] or nc >= maxc:
                        ok = False
                        break
                    cl[nc] = j; inc[j] = True; nc += 1
        # a pushed square may overlap with squares already checked at old positions: verify final
        if ok:
            for c in range(nc):
                a_ = cl[c]
                P[a_, 0] += dx; P[a_, 1] += dy
            for c in range(nc):
                a_ = cl[c]
                ha = h if a_ == k else 0.5
                for j in range(n):
                    if j == a_:
                        continue
                    hj = h if j == k else 0.5
                    if overlap_s(P[a_, 0], P[a_, 1], P[a_, 2], ha, P[j, 0], P[j, 1], P[j, 2], hj, tol):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                for c in range(nc):
                    a_ = cl[c]
                    P[a_, 0] -= dx; P[a_, 1] -= dy
        for c in range(nc):
            inc[cl[c]] = False
    return P, h
