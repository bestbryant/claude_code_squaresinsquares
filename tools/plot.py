import sys, os, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.insert(0, os.path.dirname(__file__))
from slp import corner_arrays, side, normalize

def plot(P, path, title=''):
    P = normalize(P); L = side(P)
    ox, oy = corner_arrays(P)
    fig, ax = plt.subplots(figsize=(9, 9))
    for i in range(len(P)):
        t = P[i, 2] % (math.pi / 2)
        tilt = min(t, math.pi / 2 - t) > 1e-6
        ax.add_patch(Polygon(np.c_[P[i, 0] + ox[i], P[i, 1] + oy[i]], closed=True,
                             fc='#e8a33d' if tilt else '#9fb3c8', ec='k', lw=0.3))
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], 'r-', lw=0.8)
    ax.set_xlim(-0.2, L + 0.2); ax.set_ylim(-0.2, L + 0.2); ax.set_aspect('equal')
    ax.set_title(title or f'n={len(P)} L={L:.6f}')
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    out = sys.argv[1]
    for n in sys.argv[2:]:
        P = np.load(os.path.join(os.path.dirname(__file__), '..', 'solutions', f'n{n}.npy'))
        plot(P, os.path.join(out, f'n{n}.png'))
