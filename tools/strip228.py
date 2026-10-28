"""Main-diagonal 3-row strip (n=228 family) for any K: upright cells by region rule, tilted squares by
stretching the n=228 strip (cut between layers, replicate one layer)."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from slp import side, normalize
from stretch import is_up

ROOT = os.path.join(os.path.dirname(__file__), '..')


def upright(K, f):
    cells = []
    for i in range(K):
        for j in range(K):
            d = j - i
            if i + j <= 2:
                cells.append((i + 0.5, j + 0.5, 0.0))
            elif i + j >= 2 * K - 3:
                cells.append((i + 0.5 + f, j + 0.5 + f, 0.0))
            elif d >= 3:
                cells.append((i + 0.5, j + 0.5 + f, 0.0))
            elif d <= -2:
                cells.append((i + 0.5 + f, j + 0.5, 0.0))
    return np.array(cells)


def tilted(t, nl, scut=10.68, src=228, layer=None):
    P = normalize(np.load(os.path.join(ROOT, 'solutions', f'n{src}.npy')))
    T = P[~is_up(P)]
    s = (T[:, 0] + T[:, 1]) / math.sqrt(2)
    a = 1.0
    last = T[(s < scut) & (s > scut - a)]
    keep = T[s < scut]
    mov = T[s >= scut].copy(); mov[:, :2] += t
    L = t * math.sqrt(2)
    step = L / nl if nl else 0
    new = []
    for j in range(1, nl + 1):
        q = last.copy(); q[:, :2] += step * j / math.sqrt(2)
        new.append(q)
    return np.vstack([keep] + new + [mov]) if nl else np.vstack([keep, mov])


def build(K, nl, f=None):
    t = K - 15
    P0 = np.load(os.path.join(ROOT, 'solutions', 'n228.npy'))
    f = f if f is not None else side(P0) - 15
    U = upright(K, f)
    T = tilted(t, nl)
    return np.vstack([U, T]), f
