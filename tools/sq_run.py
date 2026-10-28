"""Long squeeze runs with rigid continents on chosen n; saves improvements to scratchpad/sq."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from fault_opt import squeeze
from slp import side
from common import finalize
OUT = sys.argv[1]
tl = float(sys.argv[2]); kick = float(sys.argv[3]); seed = int(sys.argv[4])
for n in map(int, sys.argv[5:]):
    P = np.load(os.path.join(os.path.dirname(__file__), '..', 'solutions', f'n{n}.npy'))
    L0 = side(P)
    Q, L = squeeze(P, time_limit=tl, rng=np.random.default_rng(seed), hops=25, kick=kick, delta0=0.004)
    good = False
    if L < L0 - 1e-9:
        np.save(os.path.join(OUT, f'cand_n{n}_{seed}.npy'), Q)
        Q, L = finalize(Q)
        good = Q is not None and L < L0 - 1e-9
    print(n, L0, L, good, flush=True)
    if good:
        np.save(os.path.join(OUT, f'n{n}_{seed}.npy'), Q)
