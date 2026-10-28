"""Insert a full grid row + column into a (K, D) packing along lines that cut no square,
giving a (K+1, D) packing with the same f.  Also tries chains K -> K+2, ... via repetition.
"""
import sys, os, math, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from slp import corner_arrays, side, normalize
from common import finalize

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOL = os.path.join(ROOT, 'solutions')


def cut_lines(P, axis, eps=1e-9):
    ox, oy = corner_arrays(P)
    C = (P[:, 0:1] + ox) if axis == 0 else (P[:, 1:2] + oy)
    lo = C.min(1); hi = C.max(1)
    # candidate cut positions: square edges
    cands = np.unique(np.round(np.r_[lo, hi], 12))
    out = []
    for c in cands:
        if not ((lo < c - eps) & (hi > c + eps)).any():
            out.append(c)
    return out


def insert(P, cx, cy):
    P = P.copy()
    S = side(P)
    Q = P.copy()
    Q[P[:, 0] > cx, 0] += 1
    Q[P[:, 1] > cy, 1] += 1
    S2 = S + 1
    K2 = int(math.floor(S2 + 1e-9))
    new = []
    # row strip y in [cy, cy+1]: squares across x in [0, S2]
    for i in range(K2):
        new.append((i + 0.5, cy + 0.5, 0.0))
    # column strip x in [cx, cx+1], excluding the row strip
    for j in range(K2):
        y = j + 0.5
        if cy - 1e-12 < y < cy + 1 + 1e-12:
            continue
        new.append((cx + 0.5, y, 0.0))
    return np.vstack([Q, np.array(new)])


def best_side(n):
    p = os.path.join(SOL, f'n{n}.npy')
    return side(np.load(p)) if os.path.exists(p) else math.ceil(math.sqrt(n))


if __name__ == '__main__':
    gain = 0
    for n in range(2, 1001):
        p = os.path.join(SOL, f'n{n}.npy')
        if not os.path.exists(p):
            continue
        P = normalize(np.load(p)); S = side(P)
        K = int(math.floor(S + 1e-9))
        if K * K == n:
            continue
        xs = cut_lines(P, 0); ys = cut_lines(P, 1)
        if not xs or not ys:
            continue
        n2 = n + 2 * K + 1
        if n2 > 1000 or S + 1 >= best_side(n2) - 1e-9:
            continue
        done = False
        for cx in xs:
            for cy in ys:
                Q = insert(P, cx, cy)
                # the row/col squares must not exceed the new box; drop any that do and check count
                if len(Q) < n2:
                    continue
                Q = Q[:n2] if len(Q) > n2 else Q
                R, L = finalize(Q)
                if R is not None and L < best_side(n2) - 1e-9:
                    g = (best_side(n2) - L) / math.sqrt(n2)
                    np.save(os.path.join(SOL, f'n{n2}.npy'), R)
                    gain += g
                    print(f'n={n} -> n={n2}: {L:.9f} gain {g:.6f}', flush=True)
                    done = True
                    break
            if done:
                break
    print('total gain', gain)
