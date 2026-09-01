import os
"""Кладём замкнутую калибровку ПОВЕРХ его QAT: складываются ли методы?
Один проход (2 форварда по train, 0 backward) + тот же LS-вид, что и для тернарного нуля."""
import os, sys, json, time
import numpy as np
LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore")); sys.path.insert(0, LC); import nano_lc as NL
V, CTX, D = 8000, 96, 192
tr=np.load(f"{LC}/data/prep/train.npy").astype(np.int64); va=np.load(f"{LC}/data/prep/val.npy").astype(np.int64)
def load(tag):
    m=NL.NanoGPT(V,D=D,L=4,ff=576,kind="ema")
    d=np.load(f"{LC}/results/{tag}.npz")
    for k in m.p.d:
        if k in d.files: m.p.d[k][...]=d[k]
    return m
mf=load("ckpt_L_mussa1500s1")                       # fp32-учитель
E0=mf.p.d["E"].astype(np.float64)
def windows(ids,stride=CTX,limit=None):
    st=list(range(0,len(ids)-CTX-1,stride)); return st[:limit] if limit else st
st_val=windows(va); st_tr=windows(tr,limit=9000)
def nll(model,W=None,bias=None,ids=va,st=None,chunk=64):
    st=st_val if st is None else st; tot=0.0; n=0
    E=E0 if W is None else W
    for b in range(0,len(st),chunk):
        ss=st[b:b+chunk]
        x=np.stack([ids[s:s+CTX] for s in ss]); y=np.stack([ids[s+1:s+CTX+1] for s in ss]).reshape(-1)
        H,_=model.forward(x); Hf=H.reshape(-1,D).astype(np.float64)
        lg=Hf@E.T
        if bias is not None: lg=lg+bias
        lg=lg.astype(np.float32); mx=lg.max(-1,keepdims=True)
        tot+=float(((mx[:,0]+np.log(np.exp(lg-mx).sum(-1)))-lg[np.arange(len(y)),y]).sum()); n+=len(y)
    return tot/n
def stats(student):
    D1=D+1; Szz=np.zeros((D1,D1)); Szt=np.zeros((D1,D)); t0=time.time()
    for b in range(0,len(st_tr),64):
        ss=st_tr[b:b+64]
        x=np.stack([tr[s:s+CTX] for s in ss])
        zq,_=student.forward(x); zf,_=mf.forward(x)
        Z=np.concatenate([zq.reshape(-1,D).astype(np.float64),np.ones((zq.size//D,1))],1)
        Szz+=Z.T@Z; Szt+=Z.T@zf.reshape(-1,D).astype(np.float64)
    return Szz,Szt,len(st_tr)*CTX,time.time()-t0
res={}
for name,tag in (("fp32","ckpt_L_mussa1500s1"),("qat200","ckpt_L_mussa1500s1_qat200"),
                 ("qat50","ckpt_L_mussa1500s1_qat50")):
    p=f"{LC}/results/{tag}.npz"
    if not os.path.exists(p):
        print(f"[{name}] чкп нета, пропуск",flush=True); continue
    m=load(tag)
    a=nll(m); print(f"\n[{name}] полный свип nll {a:.4f} PPL {np.exp(a):.2f}",flush=True)
    res[name]={"nll":a,"ppl":float(np.exp(a))}
    if name=="fp32": continue
    Szz,Szt,n,dt=stats(m)
    print(f"  статистики за {dt:.0f} с ({n:,} позиций)",flush=True)
    for lam in (1e-4,1e-2):
        Wa=np.linalg.solve(Szz+(lam*n/D)*np.eye(D+1),Szt)
        M,bias=Wa[:D],Wa[D]
        a2=nll(m,W=E0@M.T,bias=E0@bias)
        gain=res["fp32"]["nll"]
        print(f"  + калибровка 1 проход λ={lam:g}: nll {a2:.4f} PPL {np.exp(a2):.2f}  "
              f"(до: {a:.4f}; fp32: {gain:.4f})  прирост {100*(a-a2)/max(1e-9,a-gain):5.1f}% от остатка",flush=True)
        res[f"{name}_cal{lam:g}"]={"nll":a2,"ppl":float(np.exp(a2))}
        # кривая ранга: M = I + low-rank (замкнутый «LoRA» без оптимизатора, хранится как r·(V+D))
        U_,sv_,Vt_=np.linalg.svd(M-np.eye(D),full_matrices=False)
        for r in (4,8,16,32,64,192):
            Mr=np.eye(D)+(U_[:,:r]*sv_[:r])@Vt_[:r]
            ar=nll(m,W=E0@Mr.T,bias=E0@(bias@np.eye(D)))
            print(f"    ранг {r:4d}: nll {ar:.4f} PPL {np.exp(ar):7.2f}   "
                  f"парам {r*(V+D)*4/1e6:5.2f} МБ fp32",flush=True)
            res[f"{name}_rank{r}"]={"nll":ar,"ppl":float(np.exp(ar)),"params_mb":r*(V+D)*4/1e6}
json.dump(res,open(os.path.expanduser("~/run1/quantfix2.json"),"w"),indent=1,ensure_ascii=False)
print("\nготово",flush=True)
