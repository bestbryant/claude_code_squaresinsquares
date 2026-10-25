"""Overlap-penalty relaxation for unit squares in a fixed box [0,S]^2.

energy = sum_pairs max(0, depth)^2 + sum_corners outside^2, where depth is the SAT
penetration (min over the 4 edge-normal axes of projected overlap).
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

V = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def energy_grad(z, n, S, pairs, free=None, base=None, mu=0.0):
    if free is not None:
        P = base.copy()
        P[free] = z.reshape(-1, 3)
    else:
        P = z.reshape(n, 3)
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    G = np.zeros((n, 3))
    E = 0.0
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        dx, dy = x[j] - x[i], y[j] - y[i]
        best = np.full(len(i), np.inf)
        gi = np.zeros((len(i), 3)); gj = np.zeros((len(i), 3))
        for a_is_i in (True, False):
            ta = t[i] if a_is_i else t[j]
            tb = t[j] if a_is_i else t[i]
            for k in (0, 1):
                phi = ta + k * math.pi / 2
                ux, uy = np.cos(phi), np.sin(phi)
                ud = ux * dx + uy * dy
                sg = np.where(ud >= 0, 1.0, -1.0)
                al = phi - tb
                ca, sa = np.cos(al), np.sin(al)
                rb = 0.5 * (np.abs(ca) + np.abs(sa))
                ov = 0.5 + rb - np.abs(ud) + mu
                m = ov < best
                if not m.any():
                    continue
                best = np.where(m, ov, best)
                drb = 0.5 * (-np.sign(ca) * sa + np.sign(sa) * ca)  # d rb / d alpha
                upd = -uy * dx + ux * dy  # u' . d
                d_ta = drb - sg * upd
                d_tb = -drb
                # d ov / d c_j = -sg*u ; d c_i = +sg*u
                gix = sg * ux; giy = sg * uy
                if a_is_i:
                    gi_t, gj_t = d_ta, d_tb
                else:
                    gi_t, gj_t = d_tb, d_ta
                gi[m] = np.c_[gix, giy, gi_t][m]
                gj[m] = np.c_[-gix, -giy, gj_t][m]
        pos = best > 0
        if pos.any():
            ov = best[pos]
            E += (ov ** 2).sum()
            w = 2 * ov[:, None]
            np.add.at(G, i[pos], w * gi[pos])
            np.add.at(G, j[pos], w * gj[pos])
    # box
    c, s = np.cos(t), np.sin(t)
    for vx, vy in V:
        cx = x + c * vx - s * vy
        cy = y + s * vx + c * vy
        dcx_dt = -s * vx - c * vy
        dcy_dt = c * vx - s * vy
        Sx, Sy = (S, S) if np.isscalar(S) else S
        for cc, dd, col, SS in ((cx, dcx_dt, 0, Sx), (cy, dcy_dt, 1, Sy)):
            lo = np.minimum(cc, 0.0)
            hi = np.maximum(cc - SS, 0.0)
            e = lo + hi
            if np.any(e != 0):
                E += (e ** 2).sum()
                G[:, col] += 2 * e
                G[:, 2] += 2 * e * dd
    if free is not None:
        return E, G[free].ravel()
    return E, G.ravel()


def near_pairs(P, r=math.sqrt(2) + 0.05):
    return cKDTree(P[:, :2]).query_pairs(r, output_type='ndarray')


def relax(P, S, free=None, iters=3000, rounds=6, tol=1e-28, mu=0.0):
    """Minimise overlap energy; pairs list refreshed between rounds. Returns (P, E)."""
    P = np.array(P, dtype=float)
    n = len(P)
    E = None
    for r in range(rounds):
        pairs = near_pairs(P, math.sqrt(2) + 0.3)
        if free is not None:
            fm = np.zeros(n, bool); fm[free] = True
            pairs = pairs[fm[pairs[:, 0]] | fm[pairs[:, 1]]]
            z0 = P[free].ravel()
            res = minimize(energy_grad, z0, args=(n, S, pairs, free, P, mu), jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-30, 'gtol': 1e-16, 'maxcor': 30})
            P[free] = res.x.reshape(-1, 3)
        else:
            res = minimize(energy_grad, P.ravel(), args=(n, S, pairs, None, None, mu), jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-30, 'gtol': 1e-16, 'maxcor': 30})
            P = res.x.reshape(n, 3)
        E = res.fun
        if E < tol:
            break
    return P, E


def free_space_point(P, S, res=0.05, rng=None, k=1, region=None):
    """Return up to k points in [0.5,S-0.5]^2 farthest from occupied area (distance-transform maxima)."""
    from matplotlib.path import Path
    from scipy.ndimage import distance_transform_edt, maximum_filter
    N = int(math.ceil(S / res))
    gx = (np.arange(N) + 0.5) * res
    GX, GY = np.meshgrid(gx, gx, indexing='ij')
    occ = np.zeros((N, N), bool)
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    for i in range(len(P)):
        X = P[i, 0] + c[i] * V[:, 0] - s[i] * V[:, 1]
        Y = P[i, 1] + s[i] * V[:, 0] + c[i] * V[:, 1]
        i0, i1 = max(int(X.min() / res) - 1, 0), min(int(X.max() / res) + 2, N)
        j0, j1 = max(int(Y.min() / res) - 1, 0), min(int(Y.max() / res) + 2, N)
        sub = np.c_[GX[i0:i1, j0:j1].ravel(), GY[i0:i1, j0:j1].ravel()]
        occ[i0:i1, j0:j1] |= Path(np.c_[X, Y]).contains_points(sub).reshape(i1 - i0, j1 - j0)
    pad = np.ones((N + 2, N + 2), bool)
    pad[1:-1, 1:-1] = ~occ
    D = distance_transform_edt(pad)[1:-1, 1:-1] * res
    mx = maximum_filter(D, size=int(0.6 / res)) == D
    sel = mx & (D > 0.1)
    if region is not None:
        x0, x1, y0, y1 = region
        sel &= (GX >= x0) & (GX <= x1) & (GY >= y0) & (GY <= y1)
    idx = np.argwhere(sel)
    vals = D[idx[:, 0], idx[:, 1]]
    order = np.argsort(-vals)[:k]
    return [(gx[idx[o, 0]], gx[idx[o, 1]], vals[o]) for o in order], D


def try_insert(P, S, pt, angle=0.0, radius=3.0, mu=1e-10, iters=2000):
    x, y = pt
    Q = np.vstack([P, [[min(max(x, 0.5), S - 0.5), min(max(y, 0.5), S - 0.5), angle]]])
    d = np.hypot(Q[:, 0] - x, Q[:, 1] - y)
    free = np.where(d < radius)[0]
    Q2, E = relax(Q, S, free=free, iters=iters, rounds=4, mu=mu)
    return Q2, E
