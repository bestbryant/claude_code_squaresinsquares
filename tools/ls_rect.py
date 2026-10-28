"""LS-growth search for rectangle gadgets: N = w*h+dm squares in (w+f) x (h+f).
For each (w,h,dm) bisect f; success = LS growth to full size (finished by gradient relax)."""
import sys, os, math, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mc import grow_all_rect
from relax import relax
from common import finalize

def try_f(w, h, dm, f, rng, steps, tries):
    N = w * h + dm; W, H = w + f, h + f
    F = np.zeros((0, 3))
    for t in range(tries):
        Q = np.c_[rng.uniform(0.1, W - 0.1, N), rng.uniform(0.1, H - 0.1, N), rng.uniform(0, math.pi / 2, N)]
        Q, hh = grow_all_rect(F, Q, 0.01, W, H, steps, 0.5, 0.2, int(rng.integers(1 << 30)), 1e-12, 30, 0.002)
        if hh > 0.48:
            R, E = relax(Q, (W, H), iters=4000, rounds=8, mu=1e-11)
            if E < 1e-24:
                return R
    return None

if __name__ == '__main__':
    out = sys.argv[1]; steps = int(sys.argv[2]); tries = int(sys.argv[3]); seed = int(sys.argv[4])
    specs = [tuple(map(int, s.split(','))) for s in sys.argv[5:]]
    rng = np.random.default_rng(seed)
    res = pickle.load(open(out, 'rb')) if os.path.exists(out) else {}
    for (w, h, dm) in specs:
        if (w, h, dm) in res and os.environ.get('RESUME'):
            continue
        lo, hi = 0.3, 1.0
        best = None
        t0 = time.time()
        for it in range(12):
            mid = (lo + hi) / 2
            R = try_f(w, h, dm, mid, rng, steps, tries)
            if R is not None:
                hi = mid; best = (mid, R)
            else:
                lo = mid
        if best:
            Q, _ = finalize(best[1])
            if Q is not None:
                key = (w, h, dm)
                if key not in res or best[0] < res[key][0]:
                    res[key] = (best[0], Q); pickle.dump(res, open(out, 'wb'))
        print((w, h, dm), 'f=%.5f' % best[0] if best else None, '%.0fs' % (time.time() - t0), flush=True)
