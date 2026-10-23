"""Gravity compaction: translate squares one at a time along a direction until contact."""
import math
import numpy as np

V = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def corners(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.c_[p[0] + c * V[:, 0] - s * V[:, 1], p[1] + s * V[:, 0] + c * V[:, 1]]


def toi(A, B, u, ta, tb):
    """Time of impact of polygon A (corners) translating along unit u against static B. inf if never."""
    axes = []
    for t in (ta, tb):
        axes.append((math.cos(t), math.sin(t)))
        axes.append((-math.sin(t), math.cos(t)))
    enter, exit_ = -np.inf, np.inf
    for nx, ny in axes:
        pa = A[:, 0] * nx + A[:, 1] * ny
        pb = B[:, 0] * nx + B[:, 1] * ny
        a0, a1, b0, b1 = pa.min(), pa.max(), pb.min(), pb.max()
        v = u[0] * nx + u[1] * ny
        if abs(v) < 1e-15:
            if a1 <= b0 or b1 <= a0:
                return np.inf
            continue
        if v > 0:
            e, x = (b0 - a1) / v, (b1 - a0) / v
        else:
            e, x = (b1 - a0) / v, (b0 - a1) / v
        enter = max(enter, e)
        exit_ = min(exit_, x)
    if enter >= exit_:
        return np.inf
    return enter


def slide(P, order, u, S, gap=0.0, max_t=50.0):
    """Slide squares in `order` along u (unit) as far as possible inside [0,S]^2. Modifies a copy."""
    P = np.array(P, dtype=float)
    u = np.asarray(u, dtype=float)
    C = [corners(p) for p in P]
    for i in order:
        A = C[i]
        # wall limit
        t = max_t
        for k in range(2):
            if u[k] < 0:
                t = min(t, (A[:, k].min() - 0.0) / -u[k])
            elif u[k] > 0:
                t = min(t, (S - A[:, k].max()) / u[k])
        # neighbours roughly ahead
        d = P[:, :2] - P[i, :2]
        ahead = d @ u
        lat = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
        cand = np.where((ahead > -1.5) & (lat < 1.5) & (np.arange(len(P)) != i) & (ahead < t + 1.5))[0]
        for j in cand:
            tt = toi(A, C[j], u, P[i, 2], P[j, 2])
            if tt < t:
                t = tt
        t = max(t - gap, 0.0)
        if t > 0:
            P[i, 0] += t * u[0]
            P[i, 1] += t * u[1]
            C[i] = corners(P[i])
    return P
