import os
"""Сколько данных нужно замкнутому ремонту: кривая насыщения (важно для on-device).
Тот же LS-вид, но статистика накапливается и оценивается на 2k/4k/6k/9k окнах train."""
import os, sys, json, time
import numpy as np
LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore")); sys.path.insert(0, LC); import nano_lc as NL
V,CTX,D=8000,96,192
PATS=(".fc1",".fc2",".Wm",".qkv",".Wqkv",".Wo")
tr=np.load(f"{LC}/data/prep/train.npy").astype(np.int64); va=np.load(f"{LC}/data/prep/val.npy").astype(np.int64)
def tern(W):
    s=np.abs(W).mean(-1,keepdims=True).clip(1e-5); return (np.clip(np.rint(W/s),-1,1)*s).astype(W.dtype)
def build(q):
    m=NL.NanoGPT(V,D=D,L=4,ff=576,kind="ema"); d=np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")
    for k in m.p.d:
        if k in d.files: m.p.d[k][...]=d[k]
    if q:
        for k in list(m.p.d):
            if m.p.d[k].ndim==2 and any(p in k for p in PATS): m.p.d[k][...]=tern(m.p.d[k])
    return m
mf=build(False); mq=build(True); E0=mf.p.d["E"].astype(np.float64)
st_val=list(range(0,len(va)-CTX-1,CTX))
def nll(model,W=None,bias=None):
    tot=0.0;n=0
    for b in range(0,len(st_val),64):
        ss=st_val[b:b+64]
        x=np.stack([va[s:s+CTX] for s in ss]); y=np.stack([va[s+1:s+CTX+1] for s in ss]).reshape(-1)
        H,_=model.forward(x); Hf=H.reshape(-1,D).astype(np.float64)
        lg=Hf@(E0 if W is None else W).T
        if bias is not None: lg=lg+bias
        lg=lg.astype(np.float32); mx=lg.max(-1,keepdims=True)
        tot+=float(((mx[:,0]+np.log(np.exp(lg-mx).sum(-1)))-lg[np.arange(len(y)),y]).sum()); n+=len(y)
    return tot/n
a_fp=nll(mf); a_q=nll(mq)
print(f"fp32 {a_fp:.4f} | ternary {a_q:.4f} | урон {a_q-a_fp:.4f} nat",flush=True)
D1=D+1
st=list(range(0,min(len(tr),9000*CTX)-CTX-1,CTX))
marks={64:"6k ток",512:"49k",1024:"98k",2048:"197k",4096:"393k",8960:"860k"}
Szz=np.zeros((D1,D1)); Szt=np.zeros((D1,D))
res={"fp32":a_fp,"ternary":a_q,"curve":{}}
for i,ss in enumerate((st[b:b+64] for b in range(0,len(st),64)),1):
    x=np.stack([tr[s:s+CTX] for s in ss])
    zq,_=mq.forward(x); zf,_=mf.forward(x)
    Z=np.concatenate([zq.reshape(-1,D).astype(np.float64),np.ones((zq.size//D,1))],1)
    Szz+=Z.T@Z; Szt+=Z.T@zf.reshape(-1,D).astype(np.float64)
    if i*64 in marks:
        n=float(i*64*CTX)
        Wa=np.linalg.solve(Szz+(1e-4*n/D)*np.eye(D1),Szt)
        a2=nll(mq,W=E0@Wa[:D].T,bias=E0@Wa[D])
        rec=(a_q-a2)/(a_q-a_fp)
        print(f"  {marks[i*64]:>7s} (n={int(n):,}): отремонтировано nll {a2:.4f} PPL {np.exp(a2):7.2f}  "
              f"вернул {100*rec:5.1f}% урона",flush=True)
        res["curve"][marks[i*64]]={"nll":a2,"ppl":float(np.exp(a2)),"recover":rec,"n":int(n)}
json.dump(res,open(os.path.expanduser("~/run1/quantfix3.json"),"w"),indent=1,ensure_ascii=False)
print("готово",flush=True)
