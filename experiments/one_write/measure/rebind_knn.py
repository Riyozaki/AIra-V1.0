"""Retrieval-канал за 1 проход счётчиком (2-gram таблица) против весовой памяти.
Никакого обучения: строим P(next | prev2,prev1) одним np.unique, смешиваем с его моделью."""
import os
import sys

LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore"))
sys.path.insert(0, LC)

import os
import sys, time, json
import numpy as np
import nano_lc as NL
LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore")); CACHE = os.environ.get("AIRA_CACHE", os.path.expanduser("~/run1/cache")); V=8000
tr=np.load(f"{LC}/data/prep/train.npy").astype(np.int64)
Hv=np.load(f"{CACHE}/Hv.npy").astype(np.float64); Yv=np.load(f"{CACHE}/Yv.npy")
m=NL.NanoGPT(V,D=192,L=4,ff=576,kind="ema"); d=np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")
for k in m.p.d:
    if k in d.files: m.p.d[k][...]=d[k]
E0=m.p.d["E"].astype(np.float64)
cnt=np.bincount(tr,minlength=V)
S=np.sort(np.random.default_rng(7).choice(np.where((cnt>=20)&(cnt<=400))[0],size=1000,replace=False))
mp=np.arange(V); mp[S]=np.roll(S,-1); sel=np.zeros(V,bool); sel[S]=True
Yr=mp[Yv]
# контекст каждой позиции val: пересобираем из исходного файла так же, как в feats()
CTX=96
va=np.load(f"{LC}/data/prep/val.npy").astype(np.int64)
st=list(range(0,len(va)-CTX-1,CTX))
X=np.stack([va[s:s+CTX] for s in st])
P1=X.reshape(-1); P2=np.concatenate([np.zeros((len(st),1),np.int64),X[:,:-1]],1).reshape(-1)
assert len(P1)==len(Yv)
def build(ids, do_rebind):
    t=mp[ids] if do_rebind else ids
    key=t[:-2].astype(np.int64)*V+t[1:-1]
    comb=key*V+t[2:]
    uq,c=np.unique(comb,return_counts=True)
    kk,tt=uq//V,uq%V
    o=np.argsort(kk,kind='stable'); ks,ts,cs=kk[o],tt[o],c[o].astype(np.float64)
    uk,s0=np.unique(ks,return_index=True); e0=np.concatenate([s0[1:],[len(ks)]])
    return uk,s0,e0,ts,cs
res={}
for name,src in (("rebound",True),("clean",False)):
    t0=time.time(); uk,s0,e0,ts,cs=build(tr, src); ttab=time.time()-t0
    Yt = Yr if src else Yv
    qk=(P2.astype(np.int64)*V+P1) if not src else (mp[P2].astype(np.int64)*V+mp[P1])
    pos=np.clip(np.searchsorted(uk,qk),0,max(0,len(uk)-1)); hit=(len(uk)>0)&(uk[pos]==qk)
    print(f"\n[{name}] таблица из train ({'перебинденного' if src else 'чистого'}): {ttab:.0f} с, "
          f"ключей {len(uk):,}, покрытие val {100*hit.mean():.1f}% (задетых "
          f"{100*hit[sel[Yt]].mean():.1f}%)",flush=True)
    res[f"{name}_build_sec"]=ttab; res[f"{name}_cover"]=float(hit.mean())
    for alpha in (0.0,0.3,0.7,0.95,1.0):
        tot=n=ssum=sn=0
        for b in range(0,len(Yt),4096):
            sl=slice(b,min(b+4096,len(Yt)))
            lg=(Hv[sl]@E0.T).astype(np.float32); mx=lg.max(-1,keepdims=True)
            P=np.exp(lg-mx); P=P.astype(np.float64); P/=P.sum(-1,keepdims=True)
            h=hit[sl]
            if h.any():
                jj=np.flatnonzero(h)
                for j in jj:
                    g0,g1=s0[pos[sl][j]],e0[pos[sl][j]]
                    if g1>g0:
                        w=cs[g0:g1]; w/=w.sum()
                        P[j,ts[g0:g1]]=(1-alpha)*P[j,ts[g0:g1]]+alpha*w
                        P[j]/=P[j].sum()
            nl=-np.log(np.clip(P[np.arange(len(P)),Yt[sl]],1e-12,None))
            tot+=float(nl.sum()); n+=len(nl)
            mk=sel[Yt[sl]]; ssum+=float(nl[mk].sum()); sn+=int(mk.sum())
        a=tot/n; s=ssum/max(1,sn)
        print(f"   α={alpha:4.2f}: all {a:.4f}  sel {s:.4f}",flush=True)
        res[f"{name}_a{alpha}"]={"all":a,"sel":s}
json.dump(res,open(os.path.expanduser("~/run1/rebind8.json"),"w"),indent=1,ensure_ascii=False)
print("\nготово",flush=True)
