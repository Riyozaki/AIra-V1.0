"""Честная версия: один и тот же ridge-lam, и SGD-руки оцениваются на НЕЗНАКОМЫХ отображениях."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "delta_block"))
import lc_rls as R, lc_delta as DL, probe_learn as PL

# --- 1) float32 drift при нормализованных ключах: тот же lam, что и в потоке ---
print("1) точность одного прохода RLS против решателя (одинаковый lam=1e-2)")
print(f"  {'конфигурация':44s} {'float64':>10s} {'float32':>10s}")
for Dk, n, norm in ((64, 8192, False), (64, 65536, False), (64, 8192, True), (16, 8192, True)):
    rng = np.random.default_rng(1)
    K = rng.standard_normal((n, Dk))
    if norm:
        K /= np.linalg.norm(K, axis=1, keepdims=True)
    Wt = rng.standard_normal((Dk, 16))*0.3
    V = K @ Wt + 0.01*rng.standard_normal((n, 16))
    lam = 1e-2
    out = []
    for dt in (np.float64, np.float32):
        Ka, Va = K.astype(dt), V.astype(dt)
        ref = np.linalg.solve(Ka.T@Ka + lam*np.eye(Dk, dtype=dt), Ka.T@Va)
        W, _, _ = R.rls_write_seq(Ka, Va, lam=lam)
        out.append(float(np.linalg.norm(W-ref)/np.linalg.norm(ref)))
    print(f"  {'Dk=%d n=%d %s' % (Dk, n, 'k нормированы' if norm else 'k без нормировки'):44s} {out[0]:10.2e} {out[1]:10.2e}")

# --- 2) та же задача, что probe_learn, но eval по рукам одинаковый: unseen mapping ---
print("\n2) оцениваем ВСЕ руки на незнакомых отображениях (B=16, M=12, Dk=Dv=16)")
tr, ev = np.random.default_rng(1), np.random.default_rng(2)
Xt, Yt, Xq, Yq = PL.make(tr, B=16, M=12)
Xe_t, Ye_t, Xe_q, Ye_q = PL.make(ev, B=16, M=12)
I = np.eye(16)
o = PL.rls_pred(Xe_t, Ye_t, Xe_q, I, I, I)
print(f"  {'RLS, 0 эпох (без всякого обучения)':40s} rel={PL.relerr(o, Ye_q):8.4f}   0.0 с")

def delta_eval(steps, seed=0, lr=3e-3):
    r = np.random.default_rng(seed)
    init, fwd, bwd = DL.BLOCKS["delta"]
    D = 32
    P = {k: v.astype(np.float64)*(1.0 if k.endswith(("sc","ba","bb")) else 0.5) for k, v in init(D, r).items()}
    mm = {k: np.zeros_like(v) for k, v in P.items()}; vv = dict(mm)
    X = np.concatenate([Xt, Xq], 1)
    nst = Xt.shape[1]
    t0 = time.time()
    for s in range(steps):
        Yh, ctx = fwd(X, P)
        dYf = np.zeros_like(Yh)
        dYf[:, nst:, :16] = 2.0*(Yh[:, nst:, :16]-Yq)/np.sum(Yq**2)
        _, dP = bwd(dYf, ctx, P)
        for k in P:
            mm[k] = .9*mm[k]+.1*dP[k]; vv[k] = .999*vv[k]+.001*dP[k]**2
            P[k] -= lr*mm[k]/(np.sqrt(vv[k])+1e-8)
    Xe = np.concatenate([Xe_t, Xe_q], 1)
    Yh, _ = fwd(Xe, P)
    return PL.relerr(Yh[:, Xe_t.shape[1]:, :16], Ye_q), time.time()-t0

for st in (300, 3000):
    rel, dt = delta_eval(st)
    print(f"  {'delta-rule, SGD %d шагов' % st:40s} rel={rel:8.4f}  {dt:6.1f} с")

def ema_eval(steps, seed=0, lr=3e-3):
    r = np.random.default_rng(seed)
    init, fwd, bwd = DL.BLOCKS["ema"]
    P = {k: v.astype(np.float64) for k, v in init(32, r).items()}
    X = np.concatenate([Xt, Xq], 1); nst = Xt.shape[1]
    t0 = time.time()
    for s in range(steps):
        Yh, ctx = fwd(X, P)
        dYf = np.zeros_like(Yh)
        dYf[:, nst:, :16] = 2.0*(Yh[:, nst:, :16]-Yq)/np.sum(Yq**2)
        _, dP = bwd(dYf, ctx, P)
        for k in P:
            P[k] -= lr*dP[k]/(np.sqrt(np.mean(dP[k]**2))+1e-8)
    Xe = np.concatenate([Xe_t, Xe_q], 1)
    Yh, _ = fwd(Xe, P)
    return PL.relerr(Yh[:, Xe_t.shape[1]:, :16], Ye_q), time.time()-t0

for st in (300, 3000):
    rel, dt = ema_eval(st)
    print(f"  {'EMA (его CTRL), SGD %d шагов' % st:40s} rel={rel:8.4f}  {dt:6.1f} с")
print("\n  примечание: 1 шаг = 1 проход по 16 последовательностям; у RLS проходов ноль,"
      "\n  потому что память решается, а не обучается.")
