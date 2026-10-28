"""Re-optimise library gadgets (Pareto set per dm over (cost, f)) with squeeze_rect.
Improved gadgets are saved as donor packings for dp_pipeline.py.
usage: python gadget_squeeze.py gadgets.pkl out_dir seconds [max_dm]
"""
import sys, os, pickle, math
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from gadgets import crop
from fault_opt import squeeze_rect
from common import finalize

ROOT = os.path.join(os.path.dirname(__file__), '..')


def select(G, max_dm=8, fmax=0.95):
    out = []
    for dm in range(1, max_dm + 1):
        items = sorted(((w + h + 1, f, w, h, src) for (w, h, d), (f, src) in G.items() if d == dm and f < fmax))
        bestf = 9
        for c, f, w, h, src in items:
            if f < bestf - 1e-9:
                bestf = f
                out.append((w, h, dm, f, src))
    return out


def job(args):
    w, h, dm, f, src, T, outdir, seed = args
    n, k = src
    P0 = np.load(os.path.join(ROOT, 'solutions', f'n{n}.npy')) if not isinstance(n, str) else np.load(n)
    P = crop(P0, k, w, h, f)
    ext = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    r = np.maximum(0, np.maximum(P[:, 0] + ext - w, P[:, 1] + ext - h))
    P = P[np.argsort(r)[:w * h + dm]].copy()
    best_f, best_P = f, P
    for s in range(2):
        Q, f2 = squeeze_rect(best_P, w, h, best_f, time_limit=T / 2, rng=np.random.default_rng(seed + s),
                             kick=(0.06, 0.2)[s], delta0=0.002)
        if f2 < best_f - 1e-9:
            best_f, best_P = f2, Q
    ok = False
    if best_f < f - 1e-9:
        Q, _ = finalize(best_P)
        ok = Q is not None
        best_P = Q if ok else best_P
    if ok:
        np.save(os.path.join(outdir, f'g_{w}_{h}_{dm}.npy'), best_P)
    return (w, h, dm), f, best_f, ok


if __name__ == '__main__':
    G = pickle.load(open(sys.argv[1], 'rb'))
    outdir = sys.argv[2]; T = float(sys.argv[3])
    max_dm = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(outdir, exist_ok=True)
    sel = select(G, max_dm)
    print(len(sel), 'gadgets', flush=True)
    with Pool(int(os.environ.get("NP", 4))) as pool:
        for key, f0, f1, ok in pool.imap_unordered(job, [(w, h, dm, f, src, T, outdir, i) for i, (w, h, dm, f, src) in enumerate(sel)]):
            print(key, '%.6f -> %.6f' % (f0, f1), 'SAVED' if ok else '', flush=True)
