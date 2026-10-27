"""Rotated rectangular block of a x b squares (lattice rotated by theta) with four grid
quadrants around it, shifted by (0|f, 0|f) away from the block.  Exact count:
  Delta = a*b - (K^2 - kept cells).
For each K, scan a, b, centre and find the smallest f giving each Delta.
"""
import math, sys
import numpy as np


def kept_cells(K, f, a, b, cx, cy, th, tol=1e-9):
    """Count grid cells (with quadrant offsets) lying entirely outside the rotated rectangle."""
    c, s = math.cos(th), math.sin(th)
    S = K + f
    I, J = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
    # quadrant: right of cx -> x offset f ; above cy -> y offset f
    ox = np.where(I + 0.5 + f / 2 > cx, f, 0.0)
    oy = np.where(J + 0.5 + f / 2 > cy, f, 0.0)
    x0 = I + ox; y0 = J + oy
    # cell corners in rectangle frame
    out = np.zeros_like(I, bool)
    # separating axis test between cell (axis aligned) and rectangle (rotated)
    # rect axes u=(c,s), v=(-s,c); half sizes a/2, b/2
    corners = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
    pu = [((X - cx) * c + (Y - cy) * s) for X, Y in corners]
    pv = [(-(X - cx) * s + (Y - cy) * c) for X, Y in corners]
    umin = np.minimum.reduce(pu); umax = np.maximum.reduce(pu)
    vmin = np.minimum.reduce(pv); vmax = np.maximum.reduce(pv)
    sep = (umin >= a / 2 - tol) | (umax <= -a / 2 + tol) | (vmin >= b / 2 - tol) | (vmax <= -b / 2 + tol)
    # cell axes: rectangle projection on x / y
    hx = 0.5 * (a * abs(c) + b * abs(s)); hy = 0.5 * (a * abs(s) + b * abs(c))
    sep |= (x0 >= cx + hx - tol) | (x0 + 1 <= cx - hx + tol) | (y0 >= cy + hy - tol) | (y0 + 1 <= cy - hy + tol)
    inside = (x0 >= -tol) & (x0 + 1 <= S + tol) & (y0 >= -tol) & (y0 + 1 <= S + tol)
    # quadrant consistency inside each row/col holds by construction (split at cx / cy)
    return int((sep & inside).sum()), (hx, hy)


def scan(K, th=math.pi / 4, fgrid=None, step=0.25):
    fgrid = fgrid if fgrid is not None else np.round(np.arange(0.50, 1.0, 0.0005), 6)
    best = {}
    for a in range(2, 2 * K):
        for b in range(a, 2 * K):
            hx = 0.5 * (a * abs(math.cos(th)) + b * abs(math.sin(th)))
            if 2 * hx > K + 1:
                break
            for f in fgrid:
                S = K + f
                if 2 * hx > S + 1e-9:
                    continue
                d_best = None
                for cx in np.arange(hx, S - hx + 1e-9, step):
                    for cy in np.arange(hx, S - hx + 1e-9, step):
                        kept, _ = kept_cells(K, f, a, b, cx, cy, th)
                        d = a * b - (K * K - kept)
                        if d_best is None or d > d_best:
                            d_best = d
                if d_best is not None and d_best > 0 and (d_best not in best or f < best[d_best][0]):
                    best[d_best] = (f, a, b)
    return best


if __name__ == '__main__':
    K = int(sys.argv[1])
    b = scan(K)
    for d in sorted(b):
        print(K, d, b[d])
