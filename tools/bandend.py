"""Bands of tilted sticks with optimised end gadgets.

Canonical band: x in [0,B], y in [0,W]; sticks of m squares tilted by +theta, leaning left
(stack direction e2 = (-sin t, cos t)); m cos t + sin t = W.
Stick k has its lowest corner at (X0 + k p, 0), p = 1/cos t.

Left end region (trapezoid): {0<=y<=W, x>=0, x <= X0 - y tan t}, X0 = W tan t + D.
The right end region, rotated by 180 degrees about the band centre, is the same shape.
An end gadget is a packing of that trapezoid for a given D.
"""
import math
import numpy as np


def stick_params(W, m=None):
    from bands import stick_theta
    if m is None:
        m = int(math.floor(W + 1e-9)) + 1
    t = stick_theta(m, W)
    return m, t, 1.0 / math.cos(t)


def sticks(X0, N, W, m, t):
    c, s = math.cos(t), math.sin(t)
    out = []
    for k in range(N):
        cx = X0 + k / c + 0.5 * (c - s)
        cy = 0.5 * (s + c)
        for j in range(m):
            out.append((cx - j * s, cy + j * c, t))
    return np.array(out).reshape(-1, 3)


def trapezoid(W, t, D):
    c, s = math.cos(t), math.sin(t)
    X0 = W * math.tan(t) + D
    A = np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [c, s]])
    b = np.array([0.0, 0.0, W, c * X0])
    return A, b, X0


def rot180(P, cx, cy):
    Q = P.copy()
    Q[:, 0] = 2 * cx - Q[:, 0]
    Q[:, 1] = 2 * cy - Q[:, 1]
    Q[:, 2] = Q[:, 2] + math.pi
    return Q


def assemble(B, W, m, t, gadgets):
    """gadgets: list of (D, count, packing).  Returns best band packing of length B."""
    p = 1.0 / math.cos(t)
    base = B - W * math.tan(t)
    best = (-1, None)
    if base < 0:
        return best
    Nmax = int(math.floor(base / p + 1e-12))
    for N in range(max(Nmax - 2, 0), Nmax + 1):
        r = base - N * p
        for D0, c0, P0 in gadgets:
            if D0 > r + 1e-12:
                continue
            rem = r - D0
            # best right gadget with D1 <= rem
            cand = [(c1, D1, P1) for D1, c1, P1 in gadgets if D1 <= rem + 1e-12]
            if not cand:
                continue
            c1, D1, P1 = max(cand, key=lambda z: z[0])
            tot = c0 + c1 + N * m
            if tot > best[0]:
                best = (tot, (N, D0, P0, D1, P1))
    if best[1] is None:
        return best
    N, D0, P0, D1, P1 = best[1]
    X0 = W * math.tan(t) + D0
    S_ = sticks(X0, N, W, m, t)
    # right gadget: built for depth D1 in canonical coords; rotate 180 about centre so its wall sits at x=B
    _, _, X0r = trapezoid(W, t, D1)
    R = rot180(P1, B / 2.0, W / 2.0)
    parts = [P0, S_, R]
    P = np.vstack([q for q in parts if len(q)])
    return best[0], P


def load_gadgets(path, W, t):
    """Load endsearch npz -> list of (D, count, packing); also add D+k*p variants by sticks."""
    z = np.load(path)
    out = []
    for k in z.files:
        if k.endswith('_D'):
            c = int(k[1:-2])
            out.append((float(z[k]), c, z[f'c{c}_P']))
    return out


def stick_gadgets(W, m, t, Dmax=3.0, step=0.01):
    """Baseline gadgets from partial sticks (bottom aligned) for D in [0, Dmax]."""
    from bands import fill_rect_columns
    from slp import corner_arrays
    c, s = math.cos(t), math.sin(t)
    out = []
    best_c = -1
    for D in np.arange(0, Dmax + 1e-9, step):
        A, b, X0 = trapezoid(W, t, D)
        bestP = None
        for ph in np.linspace(0, 1 / c, 25, endpoint=False):
            P = fill_rect_columns(-3, X0 + 3, 0, W, t + 1e-12, ph)
            if not len(P):
                continue
            ox, oy = corner_arrays(P)
            X = P[:, 0:1] + ox; Y = P[:, 1:2] + oy
            ok = (X >= -1e-9).all(1) & (c * X + s * Y <= c * X0 + 1e-9).all(1) & (Y <= W + 1e-9).all(1) & (Y >= -1e-9).all(1)
            if bestP is None or ok.sum() > len(bestP):
                bestP = P[ok]
        if bestP is not None and len(bestP) > best_c:
            best_c = len(bestP)
            out.append((float(D), best_c, bestP))
    return out


def lshape_gadget(K, f, h1, h2, gad1, gad2, m1=None, m2=None):
    """L-shape: top band (full width) + right band below it, using end gadgets.
    gad1/gad2: gadget lists for widths h1+f / h2+f."""
    S = K + f
    W1, W2 = h1 + f, h2 + f
    mm1, t1, _ = stick_params(W1, m1)
    mm2, t2, _ = stick_params(W2, m2)
    from bands import grid_rect, transpose
    core = grid_rect(0, K - h2, 0, K - h1)
    c1, top = assemble(S, W1, mm1, t1, gad1)
    c2, right = assemble(K - h1, W2, mm2, t2, gad2)
    if top is None or right is None:
        return None, S
    top = top.copy(); top[:, 1] += K - h1
    right = transpose(right)  # band coords (x along band) -> x'=y
    right[:, 0] += K - h2
    P = np.vstack([core, top, right])
    return P, S
