"""Start from a generated band packing and repeatedly grow extra squares into holes (MC grow_one).
usage: python grow_loop.py K f h1 h2 l1 l2 kind seed"""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bands import pinwheel, lshape
from mc_grow import try_grow, cur_side
from relax import free_space_point
from common import finalize
from slp import side
ROOT = os.path.join(os.path.dirname(__file__), '..')
K = int(sys.argv[1]); f = float(sys.argv[2]); h1, h2, l1, l2 = map(int, sys.argv[3:7]); kind = sys.argv[7]; seed = int(sys.argv[8])
rng = np.random.default_rng(seed)
P, S = (pinwheel if kind == 'pin' else lshape)(K, f, h1, h2, l1, l2)
print('start delta', len(P) - K * K, flush=True)
fails = 0
while fails < 3:
    pts, _ = free_space_point(P, S, k=25)
    got = False
    for (x, y, dd) in pts:
        for R in (2.0, 3.5):
            Q, h = try_grow(P, S, (x, y), R, 2000000, rng)
            if h >= 0.5:
                P = Q; got = True; break
        if got:
            break
    if not got:
        fails += 1
        continue
    n = len(P)
    msg = 'delta %d' % (n - K * K)
    if S < cur_side(n) - 1e-9:
        Q2, L = finalize(P)
        if Q2 is not None and L < cur_side(n) - 1e-9:
            np.save(os.path.join(ROOT, 'solutions', f'n{n}.npy'), Q2)
            msg += ' SAVED n=%d L=%.9f' % (n, L)
    print(msg, flush=True)
print('final delta', len(P) - K * K)
