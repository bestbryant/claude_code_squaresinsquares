"""Rebuild a band end: continue the band's (sheared) tilted lattice exactly up to the walls inside a
region, then greedily fill what is left with upright squares, and try MC growth for extras.
Reports the count change versus the original region.
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from slp import side, normalize, pair_gaps_vec, corner_arrays
from scipy.spatial import cKDTree


def fit_lattice(P, ref_box):
    x0, x1, y0, y1 = ref_box
    sel = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)
    T = P[sel]
    a = np.mod(T[:, 2], math.pi / 2); a = np.where(a > math.pi / 4, a - math.pi / 2, a)
    th = float(np.median(a))
    c, s = math.cos(th), math.sin(th)
    u = T[:, 0] * c + T[:, 1] * s; v = -T[:, 0] * s + T[:, 1] * c
    # columns: round u relative to a reference
    u0 = u[0]; iu = np.round(u - u0).astype(int)
    # stagger: v offset between consecutive columns
    vs = {}
    for k, vv in zip(iu, v):
        vs.setdefault(k, []).append(vv)
    ks = sorted(vs)
    fr = [np.median(np.mod(np.array(vs[k]) - vs[ks[0]][0], 1.0)) for k in ks]
    # delta per column (mod 1): fit by successive differences
    d = np.diff([fr[i] for i in range(len(ks))])
    d = np.mod(d, 1.0)
    delta = float(np.median(d))
    ustep = float(np.median(np.diff([np.median(np.array([uu for kk, uu in zip(iu, u) if kk == k])) for k in ks])))
    vref = vs[ks[0]][0]
    return th, u0 + ks[0] * 0, vref, delta, ustep


def overl(P, Q, tol=1e-10):
    """indices of Q rows overlapping any row of P (or outside nothing)."""
    if len(P) == 0:
        return np.zeros(len(Q), bool)
    tree = cKDTree(P[:, :2])
    bad = np.zeros(len(Q), bool)
    allP = np.vstack([P, Q])
    for k in range(len(Q)):
        js = tree.query_ball_point(Q[k, :2], 1.4143)
        if js:
            pr = np.array([[len(P) + k, j] for j in js])
            if pair_gaps_vec(allP, pr).min() < tol:
                bad[k] = True
    return bad


def inbox(Q, S, tol=1e-10):
    ox, oy = corner_arrays(Q)
    X = Q[:, 0:1] + ox; Y = Q[:, 1:2] + oy
    return (X.min(1) >= tol) & (Y.min(1) >= tol) & (X.max(1) <= S - tol) & (Y.max(1) <= S - tol)


def rebuild(P, region, ref_box, scale=1 + 3e-9, v_shift=0.0, u_shift=0.0):
    P = normalize(P); S = side(P)
    x0, x1, y0, y1 = region
    inR = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)
    fixed = P[~inR]
    th, u0, v0, delta, ustep = fit_lattice(P, ref_box)
    c, s = math.cos(th), math.sin(th)
    cand = []
    for i in range(-80, 80):
        for j in range(-80, 80):
            u = u0 + u_shift + i * ustep * scale
            v = v0 + v_shift + (j + i * delta) * scale
            x = u * c - v * s; y = u * s + v * c
            if x0 - 0.3 < x < x1 + 0.3 and y0 - 0.3 < y < y1 + 0.3 and x0 < x < x1 and y0 < y < y1:
                cand.append((x, y, th))
    Q = np.array(cand).reshape(-1, 3)
    Q = Q[inbox(Q, S)]
    Q = Q[~overl(fixed, Q)]
    placed = Q
    # greedy upright fill on a fine grid, bottom-left first
    added = []
    cur = np.vstack([fixed, placed])
    for yy in np.arange(y0 + 0.5, y1, 0.02):
        for xx in np.arange(x1 - 0.5, x0, -0.02):
            q = np.array([[xx, yy, 0.0]])
            if not inbox(q, S)[0]:
                continue
            if not overl(cur, q)[0]:
                cur = np.vstack([cur, q]); added.append(q[0])
    newP = cur
    return newP, int(inR.sum()), len(placed), len(added), (th, delta, ustep)
