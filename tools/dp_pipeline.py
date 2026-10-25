"""Gadget extraction from solutions/ (+ optional donor dirs) -> fault-chain DP -> build -> verify -> save.
usage: python dp_pipeline.py [donor_dir ...]
"""
import sys, os, glob, math, pickle, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from gadgets import extract, frontier, reconstruct, build
from common import side, verify

ROOT = os.path.join(os.path.dirname(__file__), '..')
SOL = os.path.join(ROOT, 'solutions')


def _ex(p):
    return p, extract(np.load(p))


def main():
    files = glob.glob(os.path.join(SOL, 'n*.npy'))
    for d in sys.argv[1:]:
        files += glob.glob(os.path.join(d, '*.npy'))
    sols = {p: np.load(p) for p in files}
    G = {}
    with Pool(4) as pool:
        for p, g in pool.imap_unordered(_ex, files, chunksize=4):
            for key, (f, k) in g.items():
                if key not in G or f < G[key][0] - 1e-13:
                    G[key] = (f, (p, k))
    print('gadgets', len(G), flush=True)
    rec = frontier(G)
    cur = {int(os.path.basename(p)[1:-4]): side(np.load(p)) for p in glob.glob(os.path.join(SOL, 'n*.npy'))}
    gain = 0
    for (K, d), f in sorted(rec.items()):
        n = K * K + d
        if n > 1000:
            continue
        old = cur.get(n, math.ceil(math.sqrt(n)))
        if K + f >= old - 1e-10:
            continue
        ch = reconstruct(G, K, d, f)
        if ch is None:
            continue
        P = build(ch, K, f, G, sols)
        if P is None or len(P) < n:
            continue
        P = P[:n]
        ok, L = verify(P)
        if ok and L < old - 1e-10:
            np.save(os.path.join(SOL, f'n{n}.npy'), P)
            gain += (old - L) / math.sqrt(n)
            print(f'n={n} {old:.9f} -> {L:.9f} chain={[c[:3] for c in ch]}', flush=True)
    print('total gain', gain)


if __name__ == '__main__':
    main()
