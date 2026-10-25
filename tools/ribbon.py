"""Periodic ribbon search along a staircase fault of slope q/p.

Continent A (lower-right) = cells (i,j) with j <= hA(i)-1, squares at (i+f+.5, j+.5).
Continent B (upper-left)  = cells (i,j) with j >= hB(i),   squares at (i+.5, j+f+.5).
Profiles are periodic: hA(i+p) = hA(i)+q, hB likewise.  Per period the continents lose
sum_i (hB(i)-hA(i)) cells relative to the full grid; the ribbon holds k free squares per period
(period vector T=(p,q)).  Gain per period = k - lost.  We minimise f for given (profile, k).
"""
import math
import numpy as np
from scipy.optimize import minimize
from relax import energy_grad

OFF = 50.0  # shift everything to positive coordinates (energy_grad has a box at [0,S])


def obstacles(hA, hB, p, q, f, reps=4):
    pts = []
    P = len(hA)
    for r in range(-reps, reps + 1):
        for i in range(p):
            I = i + r * p
            a = hA[i] + r * q
            b = hB[i] + r * q
            for j in range(a - 10, a):
                pts.append((I + f + 0.5, j + 0.5, 0.0))
            for j in range(b, b + 10):
                pts.append((I + 0.5, j + f + 0.5, 0.0))
    return np.array(pts)


def periodic_energy(z, k, T, obs, mu, band=None, imgs=(1, -1, 2, -2)):
    X = z.reshape(k, 3)
    allp = [X]
    for s in imgs:
        Y = X.copy(); Y[:, 0] += s * T[0]; Y[:, 1] += s * T[1]
        allp.append(Y)
    A = np.vstack(allp + [obs]) + np.array([OFF, OFF, 0.0])
    n = len(A)
    # pairs: free-free (within cell), free-image, free-obstacle
    fi = np.arange(k)
    others = np.arange(k, n)
    I, J = np.meshgrid(fi, np.arange(n), indexing='ij')
    I = I.ravel(); J = J.ravel()
    m = (J > I)
    I, J = I[m], J[m]
    d = np.hypot(A[I, 0] - A[J, 0], A[I, 1] - A[J, 1])
    sel = d < math.sqrt(2) + 0.3
    pairs = np.c_[I[sel], J[sel]]
    E, G = energy_grad(A.ravel(), n, 1e6, pairs, mu=mu)
    G = G.reshape(n, 3)
    # fold image gradients: pairs free-image counted once with gradient on both; image grads map back
    g = G[:k].copy()
    for idx, s in enumerate(imgs):
        g += G[k * (idx + 1):k * (idx + 2)]
    if band is not None:
        lo, hi, sl = band
        v = X[:, 1] - sl * X[:, 0]
        e1 = np.minimum(v - lo, 0.0); e2 = np.maximum(v - hi, 0.0)
        e = e1 + e2
        E += 10 * (e ** 2).sum()
        g[:, 1] += 20 * e
        g[:, 0] += -20 * e * sl
    # free-image pairs (i, image of j) and (j, image of i) are both present -> energy double counted
    return E, g.ravel()


def fit(X0, T, hA, hB, p, q, f, mu=1e-9, iters=3000):
    obs = obstacles(hA, hB, p, q, f)
    k = len(X0)
    sl = q / p
    band = (min(hA[i] - i * sl for i in range(p)) - 1.5, max(hB[i] - i * sl for i in range(p)) + f + 1.5, sl)
    res = minimize(periodic_energy, X0.ravel(), args=(k, T, obs, mu, band), jac=True, method='L-BFGS-B',
                   options={'maxiter': iters, 'ftol': 1e-30, 'gtol': 1e-15, 'maxcor': 40})
    return res.x.reshape(k, 3), res.fun


def random_init(k, T, hA, hB, p, q, f, rng):
    # sample points in the gap of one period
    pts = []
    while len(pts) < k:
        i = rng.uniform(0, p)
        ii = int(math.floor(i))
        lo = hA[ii] + (i - ii) * 0  # gap between hA and hB (+f)
        hi = hB[ii] + f
        y = rng.uniform(lo, hi)
        pts.append((i + f * 0.5, y, rng.uniform(0, math.pi / 2)))
    return np.array(pts)


def search(hA, hB, p, q, k, f_start, rng, restarts=40, f_tol=1e-4, time_limit=60):
    """Find small f admitting k ribbon squares per period. Returns (f, X) or (None, None)."""
    import time
    T = (p, q)
    t0 = time.time()
    best = (None, None)
    f = f_start
    # find a feasible configuration at f_start
    for r in range(restarts):
        X = random_init(k, T, hA, hB, p, q, f, rng)
        X, E = fit(X, T, hA, hB, p, q, f)
        if E < 1e-22:
            best = (f, X)
            break
        if time.time() - t0 > time_limit:
            break
    if best[0] is None:
        return best
    step = 0.05
    while step > f_tol and time.time() - t0 < time_limit:
        ft = best[0] - step
        if ft <= 0:
            step /= 2; continue
        X, E = fit(best[1], T, hA, hB, p, q, ft)
        ok = E < 1e-22
        tries = 0
        while not ok and tries < 8:
            Y = best[1] + np.c_[rng.normal(0, 0.1, (k, 2)), rng.normal(0, 0.1, k)]
            X, E = fit(Y, T, hA, hB, p, q, ft)
            ok = E < 1e-22
            tries += 1
        if ok:
            best = (ft, X)
        else:
            step /= 2
    return best


def build_puzzle(K, f, m, Ls, X, c=0, phase=0):
    """Diagonal periodic ribbon (period (m,m)) in a (K+f) box.
    A (lower-right) cells j < i - phase + c + hA..., B (upper-left) cells j >= hB."""
    from bands import resolve_priority, in_box
    S = K + f
    hA = lambda i: (i - phase) + c
    def hB(i):
        r, q = divmod(i - phase, m)
        return r * m + q + Ls[q] + c
    A = [(i + f + 0.5, j + 0.5, 0.0) for i in range(K) for j in range(K) if j < hA(i)]
    B = [(i + 0.5, j + f + 0.5, 0.0) for i in range(K) for j in range(K) if j >= hB(i)]
    rib = []
    for r in range(-2 * K // m - 3, 2 * K // m + 4):
        Y = X.copy()
        Y[:, 0] += r * m + phase
        Y[:, 1] += r * m + c
        rib.append(Y)
    rib = in_box(np.vstack(rib), S)
    P = resolve_priority([np.array(A).reshape(-1, 3), np.array(B).reshape(-1, 3), rib], S)
    return P, S


def search2(hA, hB, p, q, k, rng, lo=0.5, hi=0.995, iters=13, tries=30, time_limit=300, warm=None):
    import time
    T = (p, q)
    t0 = time.time()
    best = (None, None)
    if warm is not None:
        best = warm
        hi = warm[0]
    while hi - lo > 2e-4 and time.time() - t0 < time_limit:
        mid = 0.5 * (lo + hi) if best[0] is not None else hi
        ok = None
        cands = []
        if best[1] is not None:
            cands.append(best[1])
            for _ in range(4):
                cands.append(best[1] + np.c_[rng.normal(0, 0.15, (k, 2)), rng.normal(0, 0.2, k)])
        for _ in range(tries):
            cands.append(random_init(k, T, hA, hB, p, q, mid, rng))
        for X0 in cands:
            X, E = fit(X0, T, hA, hB, p, q, mid)
            if E < 1e-22:
                ok = X
                break
        if ok is not None:
            best = (mid, ok)
            hi = mid
        else:
            if best[0] is None:
                return best
            lo = mid
    return best


def _entities(X, T, hA, hB, p, q, f):
    """Free squares, their images and obstacles with variable maps: (pos(3), xvar, yvar, tvar)."""
    k = len(X)
    ents = []
    for i in range(k):
        for s in (0, 1, -1, 2, -2):
            ents.append((X[i] + np.array([s * T[0], s * T[1], 0.0]), 3 * i, 3 * i + 1, 3 * i + 2, i, s))
    obs = obstacles(hA, hB, p, q, f)
    fi = 3 * k
    # A obstacles (x depends on f) vs B (y depends on f): A have x - 0.5 - f integer
    for o in obs:
        isA = abs(((o[0] - 0.5 - f) + 1e-9) % 1.0) < 1e-6
        if isA:
            ents.append((o, fi, None, None, -1, 0))
        else:
            ents.append((o, None, fi, None, -1, 0))
    return ents


def periodic_feasible(X, T, hA, hB, p, q, f, margin=0.0):
    from slp import pair_gaps_vec
    ents = _entities(X, T, hA, hB, p, q, f)
    P = np.array([e[0] for e in ents])
    k = len(X)
    pairs = []
    for i in range(k):  # entity index of free i, shift 0 is ents[5*i]
        a = 5 * i
        d = np.hypot(P[:, 0] - P[a, 0], P[:, 1] - P[a, 1])
        for b in np.where(d < math.sqrt(2) + 1e-9)[0]:
            if b != a and not (ents[b][4] == i):
                pairs.append((a, b))
            elif ents[b][4] == i and ents[b][5] != 0:
                pairs.append((a, b))
    if not pairs:
        return True
    return pair_gaps_vec(P, np.array(pairs)).min() >= margin - 1e-15


def periodic_lp_step(X, T, hA, hB, p, q, f, tr, margin=1e-10):
    import scipy.sparse as sp
    from scipy.optimize import linprog
    from slp import pair_axes, corner_arrays
    k = len(X)
    ents = _entities(X, T, hA, hB, p, q, f)
    P = np.array([e[0] for e in ents])
    ox, oy = corner_arrays(P)
    nv = 3 * k + 1
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    reach = math.sqrt(2) + 3 * tr
    for i in range(k):
        a0 = 5 * i
        d = np.hypot(P[:, 0] - P[a0, 0], P[:, 1] - P[a0, 1])
        for b0 in np.where(d < reach)[0]:
            if b0 == a0 or (ents[b0][4] == i and ents[b0][5] == 0):
                continue
            if ents[b0][4] >= 0 and ents[b0][4] < i and ents[b0][5] == 0:
                continue  # free-free pair counted once
            gap, a, b, ux, uy = pair_axes(P, a0, b0, ox, oy)
            if gap > 3 * tr:
                continue
            vx, vy = -uy, ux
            for kk in range(4):
                px = P[b, 0] + ox[b, kk] - P[a, 0]
                py = P[b, 1] + oy[b, kk] - P[a, 1]
                fval = ux * px + uy * py
                if fval - 0.5 > 3 * tr:
                    continue
                gA = vx * px + vy * py
                gB = ux * (-oy[b, kk]) + uy * ox[b, kk]
                coef = {}
                def add(var, v):
                    if var is not None:
                        coef[var] = coef.get(var, 0.0) + v
                ea, eb = ents[a], ents[b]
                add(ea[1], ux); add(ea[2], uy); add(ea[3], -gA)
                add(eb[1], -ux); add(eb[2], -uy); add(eb[3], -gB)
                for var, v in coef.items():
                    rows.append(r); cols.append(var); vals.append(v)
                rhs.append(fval - 0.5 - margin)
                r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(max(r, 1), nv))
    c = np.zeros(nv); c[-1] = 1.0
    bounds = [(-tr, tr)] * (3 * k) + [(-tr, tr)]
    res = linprog(c, A_ub=A, b_ub=np.array(rhs) if r else np.zeros(1), bounds=bounds, method='highs')
    if res.status != 0:
        return None, None
    dx = res.x
    Y = X.copy()
    Y[:, 0] += dx[0:3 * k:3]; Y[:, 1] += dx[1:3 * k:3]; Y[:, 2] += dx[2:3 * k:3]
    return Y, f + dx[-1]


def periodic_slp(X, T, hA, hB, p, q, f, iters=100, tr=0.05):
    for it in range(iters):
        Y, fn = periodic_lp_step(X, T, hA, hB, p, q, f, tr)
        if Y is None:
            tr *= 0.5
            if tr < 1e-9:
                break
            continue
        if fn < f - 1e-13 and periodic_feasible(Y, T, hA, hB, p, q, fn):
            gain = f - fn
            X, f = Y, fn
            tr = min(tr * 1.5, 0.1)
            if gain < 1e-11:
                break
        else:
            tr *= 0.3
            if tr < 1e-9:
                break
    return X, f


def basin_hop(X, T, hA, hB, p, q, f, rng, n_hops=200, kick=0.15, up=0.02, verbose=False, time_limit=600):
    import time
    t0 = time.time()
    X, f = periodic_slp(X, T, hA, hB, p, q, f)
    best = (f, X)
    for h in range(n_hops):
        if time.time() - t0 > time_limit:
            break
        f1 = best[0] + up * rng.random()
        Y = best[1] + np.c_[rng.normal(0, kick, (len(X), 2)), rng.normal(0, kick, len(X))]
        Y, E = fit(Y, T, hA, hB, p, q, f1)
        if E > 1e-22:
            continue
        Y, fy = periodic_slp(Y, T, hA, hB, p, q, f1)
        if fy < best[0] - 1e-9:
            best = (fy, Y)
            if verbose:
                print(f'  hop {h}: f={fy:.6f}', flush=True)
    return best


def build_puzzle_pq(K, f, p, q, hA0, hB0, X, c=0, phase=0):
    """General-slope periodic ribbon (period (p,q)) in a (K+f) box."""
    from bands import resolve_priority, in_box
    S = K + f
    def H(h0, i):
        r, s = divmod(i - phase, p)
        return h0[s] + r * q + c
    A = [(i + f + 0.5, j + 0.5, 0.0) for i in range(K) for j in range(K) if j < H(hA0, i)]
    B = [(i + 0.5, j + f + 0.5, 0.0) for i in range(K) for j in range(K) if j >= H(hB0, i)]
    rib = []
    for r in range(-3 * K // max(p, 1) - 3, 3 * K // max(p, 1) + 4):
        Y = X.copy()
        Y[:, 0] += r * p + phase
        Y[:, 1] += r * q + c
        rib.append(Y)
    rib = in_box(np.vstack(rib), S)
    P = resolve_priority([np.array(A).reshape(-1, 3), np.array(B).reshape(-1, 3), rib], S)
    return P, S
