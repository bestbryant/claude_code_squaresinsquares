"""Melt-and-recompress whole packings with hard-square MC; polish best with SLP; save if better."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mc import run_box
from slp import polish, side, normalize
from common import verify
ROOT = os.path.join(os.path.dirname(__file__), '..')
out = sys.argv[1]; n = int(sys.argv[2]); melt = float(sys.argv[3]); rounds = int(sys.argv[4]); steps = int(sys.argv[5])
P0 = normalize(np.load(os.path.join(ROOT, 'solutions', f'n{n}.npy'))); L0 = side(P0)
best = L0
rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
for r in range(rounds):
    S = L0 * (1 + melt)
    P = P0 * np.array([1 + melt, 1 + melt, 1.0])
    t = time.time()
    for stage, (dxy, pr) in enumerate([(0.05, 50.0), (0.02, 300.0), (0.008, 3000.0), (0.003, 30000.0)]):
        P, S, acc = run_box(P, S, steps, dxy, dxy, pr, int(rng.integers(1 << 30)), 1e-12, 50)
    Q, L = polish(P, iters=300, time_limit=120)
    ok = verify(Q)[0]
    print(r, 'mc S=%.6f polished %.6f (cur %.6f) ok=%s %.0fs' % (S, L, L0, ok, time.time() - t), flush=True)
    if ok and L < best - 1e-9:
        best = L
        np.save(os.path.join(out, f'n{n}_mc.npy'), Q)
