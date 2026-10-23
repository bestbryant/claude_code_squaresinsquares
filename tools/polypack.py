"""Pack unit squares into a convex polygon given by half-planes a.x <= b (rows of A, b).

Penalty relaxation + basin hopping.  Used to optimise band-end regions.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from relax import energy_grad as pair_energy_grad

V = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def poly_energy_grad(z, n, A, b, pairs, mu, fixed=None):
    P = z.reshape(n, 3)
    if fixed is not None and len(fixed):
        Pall = np.vstack([P, fixed])
        E, G = pair_energy_grad(Pall.ravel(), len(Pall), 1e9, pairs, mu=mu)
        G = G.reshape(-1, 3)[:n].ravel()
    else:
        E, G = pair_energy_grad(z, n, 1e9, pairs, mu=mu)
    G = G.reshape(n, 3)
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    c, s = np.cos(t), np.sin(t)
    for vx, vy in V:
        cx = x + c * vx - s * vy
        cy = y + s * vx + c * vy
        dcx = -s * vx - c * vy
        dcy = c * vx - s * vy
        for (ax, ay), bb in zip(A, b):
            e = np.maximum(ax * cx + ay * cy - bb, 0.0)
            if e.any():
                E += (e ** 2).sum()
                G[:, 0] += 2 * e * ax
                G[:, 1] += 2 * e * ay
                G[:, 2] += 2 * e * (ax * dcx + ay * dcy)
    return E, G.ravel()


def _pairs(allP, n, fixed, r):
    pairs = cKDTree(allP[:, :2]).query_pairs(r, output_type='ndarray')
    if fixed is not None and len(pairs):
        pairs = pairs[(pairs[:, 0] < n) | (pairs[:, 1] < n)]
    return pairs


def relax_poly(P, A, b, mu=1e-9, iters=4000, fixed=None, passes=6):
    P = np.array(P, float)
    n = len(P)
    E = np.inf
    for p in range(passes):
        allP = P if fixed is None else np.vstack([P, fixed])
        pairs = _pairs(allP, n, fixed, math.sqrt(2) + (0.6 if p == 0 else 0.25))
        res = minimize(poly_energy_grad, P.ravel(), args=(n, A, b, pairs, mu, fixed), jac=True,
                       method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-30, 'gtol': 1e-15, 'maxcor': 40})
        Pn = res.x.reshape(n, 3)
        moved = np.abs(Pn[:, :2] - P[:, :2]).max()
        P = Pn
        if moved < 0.1:
            break
    # exact energy with a full fresh pair list
    allP = P if fixed is None else np.vstack([P, fixed])
    pairs = _pairs(allP, n, fixed, math.sqrt(2) + 1e-6)
    E, _ = poly_energy_grad(P.ravel(), n, A, b, pairs, mu, fixed)
    return P, E


def sample_in_poly(A, b, bbox, rng, k):
    out = []
    x0, x1, y0, y1 = bbox
    while len(out) < k:
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if np.all(A @ p <= b - 0.3):
            out.append(p)
    return np.array(out)


def try_count(A, b, bbox, count, rng, tries=50, angles=(0.0,), init=None, mu=1e-9, fixed=None, hop=30):
    """Try to place `count` squares. Returns (P, E) best found."""
    best = (None, np.inf)
    for t in range(tries):
        if init is not None and t % 2 == 0:
            P = init.copy()
            extra = count - len(P)
            if extra > 0:
                pts = sample_in_poly(A, b, bbox, rng, extra)
                P = np.vstack([P, np.c_[pts, rng.choice(angles, extra)]])
            P[:, :2] += rng.normal(0, 0.05, (len(P), 2))
        else:
            pts = sample_in_poly(A, b, bbox, rng, count)
            P = np.c_[pts, rng.choice(angles, count) + rng.normal(0, 0.02, count)]
        P, E = relax_poly(P, A, b, mu, fixed=fixed)
        # basin hopping on the worst squares
        for h in range(hop):
            if E < 1e-22:
                break
            Q = P.copy()
            k = rng.integers(1, 4)
            idx = rng.choice(len(Q), k, replace=False)
            if rng.random() < 0.5:
                pts = sample_in_poly(A, b, bbox, rng, k)
                Q[idx, :2] = pts
            else:
                Q[idx, :2] += rng.normal(0, 0.3, (k, 2))
            Q[idx, 2] = rng.choice(angles, k) + rng.normal(0, 0.05, k)
            Q, E2 = relax_poly(Q, A, b, mu, fixed=fixed)
            if E2 < E:
                P, E = Q, E2
        if E < best[1]:
            best = (P, E)
        if E < 1e-22:
            break
    return best
