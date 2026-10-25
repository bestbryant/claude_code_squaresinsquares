"""Turn periodic diagonal ribbons into full puzzles; save verified improvements to solutions/.

usage: python ribbon_pipeline.py ribbons.pkl [more.pkl ...]
pkl: dict (p,q,Ls,k) -> (f, X)
"""
import sys, os, glob, math, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ribbon import build_puzzle, build_puzzle_pq
from common import side, verify, score_n

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOL = os.path.join(ROOT, 'solutions')


def current():
    return {int(os.path.basename(p)[1:-4]): side(np.load(p)) for p in glob.glob(os.path.join(SOL, 'n*.npy'))}


def main():
    rib = {}
    for p in sys.argv[1:]:
        rib.update(pickle.load(open(p, 'rb')))
    cur = current()
    gain = 0
    for key, (f, X) in rib.items():
        if f is None:
            continue
        if len(key) == 4:
            p, q, Ls, k = key
            hA0 = list(range(p)); hB0 = [i + Ls[i] for i in range(p)]
        else:
            p, q, hA0, hB0, k = key
            Ls = key
        m = p
        lost = sum(b - a for a, b in zip(hB0, hA0)) if False else sum(hB0[i] - hA0[i] for i in range(p))
        g = (k - lost) / p
        for K in range(8, 32):
            n_est = int(K * K + g * K)
            if all(cur.get(nn, math.ceil(math.sqrt(nn))) <= K + f + 1e-9 for nn in range(max(1, n_est - 8), min(1000, n_est + 2) + 1)):
                continue
            best = {}
            for c in range(-6, 7):
                for ph in range(m):
                    P, S = build_puzzle_pq(K, f, p, q, list(hA0), list(hB0), X, c, ph)
                    n = len(P)
                    if n > 1000:
                        continue
                    if n not in best:
                        best[n] = P
            for n, P in best.items():
                # the packing also serves every smaller n
                L = side(P)
                for nn in range(n, max(n - 3, 0), -1):
                    old = cur.get(nn, math.ceil(math.sqrt(nn)))
                    if L < old - 1e-9:
                        Q = P[:nn]
                        ok, Lv = verify(Q)
                        if ok and Lv < old - 1e-9:
                            np.save(os.path.join(SOL, f'n{nn}.npy'), Q)
                            gain += (old - Lv) / math.sqrt(nn)
                            print(f'n={nn} {old:.6f} -> {Lv:.6f}  ribbon {(p, Ls, k)} f={f:.5f}', flush=True)
                            cur[nn] = Lv
    print('total gain', gain)


if __name__ == '__main__':
    main()
