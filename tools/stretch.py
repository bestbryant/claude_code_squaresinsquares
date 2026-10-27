"""Stretch a packing that contains a diagonal strip: insert t columns at x=c and t rows at y=r
(their crossing on the strip), copy the neighbouring grid column/row into the new bands, and
replicate the strip's tilted layers to fill the new length.  Then resolve overlaps and squeeze.

Box K -> K+t; the strip gains ~ rows*(t*sqrt2/a) - corridor*t squares.
"""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from slp import side, normalize, pair_gaps_vec
from scipy.spatial import cKDTree


def is_up(P):
    t = np.mod(P[:, 2], math.pi / 2)
    return np.minimum(t, math.pi / 2 - t) < 1e-9


def overlaps(P, idx_new, others, tol=1e-9):
    """Return mask over idx_new of squares that overlap any square in `others` (indices)."""
    tree = cKDTree(P[others, :2])
    bad = np.zeros(len(idx_new), bool)
    for k, i in enumerate(idx_new):
        js = tree.query_ball_point(P[i, :2], math.sqrt(2) + 1e-6)
        js = [others[j] for j in js if others[j] != i]
        if js:
            pr = np.array([[i, j] for j in js])
            if pair_gaps_vec(P, pr).min() < -tol:
                bad[k] = True
    return bad


def stretch(P, c, r, t, layer_dir=(1, 1), s_cut=None, nl=None, a=None):
    """P normalised; (c, r) crossing point; returns new packing (unsqueezed) and n added."""
    P = normalize(np.asarray(P, float))
    up = is_up(P)
    x, y = P[:, 0], P[:, 1]
    sx = (x > c).astype(float) * t
    sy = (y > r).astype(float) * t
    d = np.array(layer_dir, float); d /= np.linalg.norm(d)
    s = P[:, :2] @ d
    if s_cut is None:
        s_cut = np.array([c, r]) @ d
    Q = P.copy()
    # upright squares: quadrant shift.  tilted: shift by (t,t) if beyond the cut along the strip
    Q[up, 0] += sx[up]; Q[up, 1] += sy[up]
    tl = ~up
    beyond = tl & (s > s_cut)
    Q[beyond, 0] += t; Q[beyond, 1] += t
    new = []
    # column band copies: squares whose cell is the last column before c
    U = np.where(up)[0]
    colsrc = U[(x[U] < c) & (x[U] > c - 1)]
    for i in colsrc:
        for k in range(1, t + 1):
            new.append((Q[i, 0] + k, Q[i, 1], Q[i, 2]))
    rowsrc = U[(y[U] < r) & (y[U] > r - 1)]
    for i in rowsrc:
        for k in range(1, t + 1):
            new.append((Q[i, 0], Q[i, 1] + k, Q[i, 2]))
    # strip layers: tilted squares with s in [s_cut - a_eff*?, s_cut) form the last layer(s)
    L = math.sqrt(2) * t if tuple(layer_dir) in ((1, 1), (1, -1)) else t * np.linalg.norm(layer_dir)
    if a is None:
        a = 1.0016
    if nl is None:
        nl = int(math.floor(L / a + 1e-9))
    T = np.where(tl & (s <= s_cut))[0]
    # last layer: the tilted squares within one spacing of the cut
    last = T[s[T] > s_cut - a * 0.999]
    step = L / nl if nl else 0
    tilted_new = []
    for j in range(1, nl + 1):
        for i in last:
            tilted_new.append((P[i, 0] + d[0] * step * j, P[i, 1] + d[1] * step * j, P[i, 2]))
    R = np.vstack([Q, np.array(new).reshape(-1, 3), np.array(tilted_new).reshape(-1, 3)])
    n0 = len(Q)
    i_up_new = np.arange(n0, n0 + len(new))
    i_t_new = np.arange(n0 + len(new), len(R))
    # drop duplicate upright copies (same centre)
    keep = np.ones(len(R), bool)
    cent = {}
    for i in list(np.where(is_up(R))[0]):
        key = (round(R[i, 0], 6), round(R[i, 1], 6))
        if key in cent:
            keep[i] = False
        else:
            cent[key] = i
    R = R[keep]
    # drop upright squares overlapping tilted ones
    upR = is_up(R)
    iu = np.where(upR)[0]; it = np.where(~upR)[0]
    if len(it):
        bad = overlaps(R, iu, it, tol=1e-6)
        R = np.delete(R, iu[bad], axis=0)
    return normalize(R)


if __name__ == '__main__':
    pass
