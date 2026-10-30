"""Spread every packing slightly (centres scaled by 1+e about their mean) so exact contacts survive
ulp-level parsing differences in Kaggle's scorer.  Checks the official metric under random
1-ulp-scale perturbations; escalates e until it passes."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import metric
from common import fmt
from multiprocessing import Pool
SOL = os.path.join(os.path.dirname(__file__), '..', 'solutions')


def ok(P):
    sq = [metric.Square(x=float(fmt(a)), y=float(fmt(b)), angle=float(fmt(c)), x_str=fmt(a), y_str=fmt(b),
                        angle_str=fmt(c)) for a, b, c in P]
    return metric.verify_puzzle(sq)[0]


def robust(P, n):
    rng = np.random.default_rng(n)
    if not ok(P):
        return False
    for _ in range(4):
        Q = P.copy(); Q[:, :2] *= 1 + rng.normal(0, 2e-16, (len(P), 2)); Q[:, 2] += rng.normal(0, 1e-16, len(P))
        if not ok(Q):
            return False
    return True


def job(p):
    n = int(os.path.basename(p)[1:-4]); P = np.load(p)
    if robust(P, n):
        return n, 0.0
    c = P[:, :2].mean(0)
    for e in (1e-10, 1e-9, 1e-8, 1e-7):
        Q = P.copy(); Q[:, :2] = c + (P[:, :2] - c) * (1 + e)
        if robust(Q, n):
            np.save(p, Q)
            return n, e
    return n, None


if __name__ == '__main__':
    files = sys.argv[1:] or glob.glob(os.path.join(SOL, 'n*.npy'))
    with Pool(4) as pool:
        res = pool.map(job, files)
    fixed = [(n, e) for n, e in res if e]
    bad = [n for n, e in res if e is None]
    print('spread', len(fixed), 'unfixable', bad)
