"""Compare stored solutions against per-puzzle best L (best_L.json from the public per-puzzle leaderboard)."""
import sys, os, json, math, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import side
ROOT = os.path.join(os.path.dirname(__file__), '..')
best = {r['n']: (float(r['L']) or math.ceil(math.sqrt(r['n'])), r['team']) for r in json.load(open(sys.argv[1]))}
mine = {}
for p in glob.glob(os.path.join(ROOT, 'solutions', 'n*.npy')):
    n = int(os.path.basename(p)[1:-4]); mine[n] = side(np.load(p))
tot = 0; totb = 0; gaps = []
for n in range(1, 1001):
    L = mine.get(n, math.ceil(math.sqrt(n)))
    tot += L / math.sqrt(n) - 1; totb += best[n][0] / math.sqrt(n) - 1
    gaps.append(((L - best[n][0]) / math.sqrt(n), n, L, best[n][0], best[n][1]))
print('mine', tot, 'best', totb)
gaps.sort(reverse=True)
for g in gaps[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print('%.5f n=%d mine=%.6f best=%.6f %s' % g)
print('n where I beat best:', sum(1 for g in gaps if g[0] < -1e-12))
