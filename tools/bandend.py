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
