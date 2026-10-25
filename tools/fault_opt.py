"""Squeeze a packing with rigid upright continents.

Upright squares are classified by their lattice offset: x-offset 0 (flush left) or f (flush right),
y-offset 0 or f.  When the box side K+f shrinks, each group translates rigidly with f.
Everything else (tilted / floating squares) is free and relaxed with the overlap penalty.
"""
import math, time
import numpy as np
from relax import relax
from slp import side, normalize


def classify(P, L, tol=1e-6):
    K = int(math.floor(L + 1e-9))
    f = L - K
    t = np.mod(P[:, 2], math.pi / 2)
    up = (np.minimum(t, math.pi / 2 - t) < 1e-9)
    gx = np.full(len(P), -1); gy = np.full(len(P), -1)
    for i in np.where(up)[0]:
        for ax, g in ((0, gx), (1, gy)):
            r = (P[i, ax] - 0.5) - math.floor(P[i, ax] - 0.5 + tol)
            if abs(r) < tol or abs(r - 1) < tol:
                g[i] = 0
            elif abs(r - f) < tol:
                g[i] = 1
    grouped = (gx >= 0) & (gy >= 0)
    return K, f, gx, gy, grouped


def place(P, gx, gy, grouped, f_old, f_new):
    Q = P.copy()
    d = f_new - f_old
    Q[grouped, 0] += gx[grouped] * d
    Q[grouped, 1] += gy[grouped] * d
    return Q


def squeeze(P, time_limit=120, rng=None, verbose=False, delta0=0.01, hops=6, kick=0.08, mu=1e-11):
    rng = rng or np.random.default_rng(0)
    P = normalize(np.asarray(P, float))
    L = side(P)
    K, f, gx, gy, grouped = classify(P, L)
    free = np.where(~grouped)[0]
    if len(free) == 0:
        return P, L
    t0 = time.time()
    best_f, best_P = f, P.copy()
    delta = delta0
    while time.time() - t0 < time_limit and delta > 1e-6:
        ft = best_f - delta
        Q = place(best_P, gx, gy, grouped, best_f, ft)
        # free squares: scale towards the box
        S = K + ft
        Q2, E = relax(Q, S, free=free, iters=2000, rounds=4, mu=mu)
        ok = E < 1e-24
        h = 0
        while not ok and h < hops and time.time() - t0 < time_limit:
            R = Q.copy()
            R[free, :2] += rng.normal(0, kick, (len(free), 2))
            R[free, 2] += rng.normal(0, kick / 2, len(free))
            Q2, E = relax(R, S, free=free, iters=2000, rounds=4, mu=mu)
            ok = E < 1e-24
            h += 1
        if ok:
            best_f, best_P = ft, Q2
            if verbose:
                print(f'  f={best_f:.6f} (delta {delta:.4g}) t={time.time()-t0:.0f}', flush=True)
            delta *= 1.5
        else:
            delta *= 0.4
    return best_P, K + best_f


def classify_rect(P, w, h, f, tol=1e-6):
    """Group upright squares by lattice offset along x and y: offset v0 (group 0) or v0+f (group 1),
    where v0 is the offset of the squares touching the low wall."""
    t = np.mod(P[:, 2], math.pi / 2)
    up = (np.minimum(t, math.pi / 2 - t) < 1e-9)
    gx = np.full(len(P), -1); gy = np.full(len(P), -1)
    for ax, g in ((0, gx), (1, gy)):
        idx = np.where(up)[0]
        if not len(idx):
            continue
        fr = np.mod(P[idx, ax] - 0.5, 1.0)
        lowest = idx[np.argmin(P[idx, ax])]
        v0 = np.mod(P[lowest, ax] - 0.5, 1.0)
        v1 = np.mod(v0 + f, 1.0)
        for i, r in zip(idx, fr):
            if min(abs(r - v0), 1 - abs(r - v0)) < tol:
                g[i] = 0
            elif min(abs(r - v1), 1 - abs(r - v1)) < tol:
                g[i] = 1
    return gx, gy, (gx >= 0) & (gy >= 0)


def squeeze_rect(P, w, h, f, time_limit=120, rng=None, verbose=False, delta0=0.01, hops=8, kick=0.1, mu=1e-11):
    """Minimise f for a gadget packing inside [0,w+f]x[0,h+f] with rigid upright groups."""
    rng = rng or np.random.default_rng(0)
    P = np.asarray(P, float)
    gx, gy, grouped = classify_rect(P, w, h, f)
    free = np.where(~grouped)[0]
    t0 = time.time()
    best_f, best_P = f, P.copy()
    delta = delta0
    while time.time() - t0 < time_limit and delta > 1e-6:
        ft = best_f - delta
        Q = place(best_P, gx, gy, grouped, best_f, ft)
        S = (w + ft, h + ft)
        Q2, E = relax(Q, S, free=free, iters=2000, rounds=4, mu=mu)
        ok = E < 1e-24
        hh = 0
        while not ok and hh < hops and time.time() - t0 < time_limit:
            R = Q.copy()
            sel = free if rng.random() < 0.5 else rng.choice(free, max(1, len(free) // 4), replace=False)
            R[sel, :2] += rng.normal(0, kick, (len(sel), 2))
            R[sel, 2] += rng.normal(0, kick / 2, len(sel))
            Q2, E = relax(R, S, free=free, iters=2000, rounds=4, mu=mu)
            ok = E < 1e-24
            hh += 1
        if ok:
            best_f, best_P = ft, Q2
            if verbose:
                print(f'  f={best_f:.6f} (delta {delta:.4g}) t={time.time()-t0:.0f}', flush=True)
            delta *= 1.5
        else:
            delta *= 0.4
    return best_P, best_f
