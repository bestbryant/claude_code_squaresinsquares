import sys, math, numpy as np
from diamond import kept_cells
from multiprocessing import Pool
def run(K):
    th=math.pi/4; best={}
    fg=np.arange(0.52,0.999,0.002)
    for a in range(3,2*K):
        for b in range(a,2*K):
            hx=0.5*(a+b)/math.sqrt(2)
            if 2*hx>K+1: break
            for f in fg:
                S=K+f
                if 2*hx>S: continue
                room=S-2*hx
                dbest=-10**9
                for al in (0,0.25,0.5):
                    for be in (0,0.25,0.5,0.75,1.0):
                        cx=hx+al*room; cy=hx+be*room
                        k,_=kept_cells(K,f,a,b,cx,cy,th)
                        dbest=max(dbest,a*b-(K*K-k))
                if dbest>0 and (dbest not in best or f<best[dbest][0]): best[dbest]=(round(f,4),a,b)
                if dbest>0: break  # minimal f for this (a,b) found... record and continue scanning f for higher d
    return K,best
if __name__=='__main__':
    Ks=list(map(int,sys.argv[1:]))
    with Pool(2) as p:
        for K,b in p.imap_unordered(run,Ks):
            print(K,{d:b[d] for d in sorted(b)},flush=True)
