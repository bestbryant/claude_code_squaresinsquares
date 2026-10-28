"""Compare ls_rect results with the gadget library; export better ones as DP donors."""
import sys, os, pickle
import numpy as np
S = sys.argv[1]
G = pickle.load(open(os.path.join(S, 'gadgets.pkl'), 'rb'))
R = pickle.load(open(os.path.join(S, 'lsrect.pkl'), 'rb'))
os.makedirs(os.path.join(S, 'gdon'), exist_ok=True)
for (w, h, dm), (f, P) in sorted(R.items()):
    lib = min([G[k][0] for k in ((w, h, dm), (h, w, dm)) if k in G] or [9])
    tag = 'BETTER' if f < lib - 1e-6 else ''
    print((w, h, dm), 'ls %.5f lib %.5f %s' % (f, lib, tag))
    if tag:
        np.save(os.path.join(S, 'gdon', f'ls_{w}_{h}_{dm}.npy'), P)
