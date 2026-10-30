"""Merge many candidate submissions: for each n keep the smallest-L packing that passes official verification.

usage: python tools/merge.py out_dir sub1.csv sub2.csv ...   (out_dir gets best.npz + submission.csv)
Candidate pool is also persisted in solutions/ as one .npy per n (best verified so far).
"""
import sys, os, math, glob
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from common import load_submission, side, verify, write_submission, score_n, grid

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOLDIR = os.path.join(ROOT, 'solutions')


def load_store():
    sol = {}
    for p in glob.glob(os.path.join(SOLDIR, 'n*.npy')):
        n = int(os.path.basename(p)[1:-4])
        sol[n] = np.load(p)
    return sol


def best_valid(args):
    n, cands = args
    cands = sorted(cands, key=lambda a: side(a))
    for a in cands:
        ok, L = verify(a)
        if ok:
            return n, a, L
    return n, None, None


def main():
    paths = sys.argv[1:]
    store = load_store()
    pool_c = {n: [] for n in range(1, 1001)}
    for n, a in store.items():
        pool_c[n].append(a)
    for p in paths:
        try:
            s = load_submission(p)
        except Exception as e:
            print('skip', p, e); continue
        for n, a in s.items():
            if 1 <= n <= 1000:
                pool_c[n].append(a)
        print('loaded', p, len(s), flush=True)
    # prune: only keep candidates better than the stored one
    work = []
    for n in range(1, 1001):
        cur = side(store[n]) if n in store else math.ceil(math.sqrt(n))
        c = [a for a in pool_c[n] if side(a) < cur - 2e-7 or (n in store and a is store[n])]
        # dedupe by L
        seen = {}
        for a in c:
            seen.setdefault(round(side(a), 13), a)
        if seen:
            work.append((n, list(seen.values())))
    os.makedirs(SOLDIR, exist_ok=True)
    with Pool(4) as pool:
        for n, a, L in pool.imap_unordered(best_valid, work, chunksize=4):
            if a is not None:
                old = side(store[n]) if n in store else math.ceil(math.sqrt(n))
                if n not in store or L < old - 2e-7:
                    store[n] = a
                    np.save(os.path.join(SOLDIR, f'n{n}.npy'), a)
    total = sum(score_n(n, side(store[n]) if n in store else math.ceil(math.sqrt(n))) for n in range(1, 1001))
    print('TOTAL', total)


if __name__ == '__main__':
    main()
