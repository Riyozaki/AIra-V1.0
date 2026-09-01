"""Решающий тест СТАВКИ на реальных статистиках его корпуса (не на синтетике).

Память = линейное отображение k → v, где
  k_t = нормированная (whitened PCA) проекция его же H_t  (как у него: k = нормированный x)
  v_t = та же проекция эмбеддинга СЛЕДУЮЩЕГО токена   (то, что миксер и должен предсказывать)
Способы заполнить:
  RLS   — 1 проход по данным, замкнуто, 0 эпох, 0 обратных проходов;
  Adam  — N шагов спуска по MSE на том же W (это и есть delta/EMA-рука в линейном случае);
  ridge — пакетный LS = глобальный оптимум (эталон).
Вопрос один: сколько шагов спуска нужно, чтобы догнать 1 проход. Если «бесконечно» —
ставка ×100 живёт в состоянии, а не в голове (в голове я это уже померил: ×3.5 по данным).
"""
import os
import sys

LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore"))
sys.path.insert(0, LC)

import json
import time

import numpy as np

import nano_lc as NL  # noqa: E402

LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore"))
CACHE = os.environ.get("AIRA_CACHE", os.path.expanduser("~/run1/cache"))
D = 48
m = NL.NanoGPT(8000, D=192, L=4, ff=576, kind="ema")
d = np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")
for k in m.p.d:
    if k in d.files:
        m.p.d[k][...] = d[k]
E0 = m.p.d["E"].astype(np.float64)
Ht = np.load(f"{CACHE}/Ht.npy")
Yt = np.load(f"{CACHE}/Yt.npy")
Hv = np.load(f"{CACHE}/Hv.npy")
Yv = np.load(f"{CACHE}/Yv.npy")
N = min(len(Ht), 400000)
rng = np.random.default_rng(0)
sub = rng.choice(len(Ht), 20000, replace=False)
Xs = Ht[sub].astype(np.float64)
Xs -= Xs.mean(0)
U, sv_, Vt_ = np.linalg.svd(Xs, full_matrices=False)
R = Vt_[:D].T
sv = np.sqrt(np.maximum((Xs @ R).var(0), 1e-12))
print("спектр H (доли от макс.):", np.round(sv_[:8] / sv_[0], 3), flush=True)
ER = (E0 @ R) / sv                                    # (V, D) цели в том же базисе


def kv(H, Y):
    K = (H.astype(np.float64) @ R) / sv
    K /= np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-9)
    Vv = ER[Y]
    Vv = Vv / np.maximum(np.linalg.norm(Vv, axis=1, keepdims=True), 1e-9)
    return K.astype(np.float32), Vv.astype(np.float32)


t0 = time.time()
Kt, Tt = kv(Ht[:N], Yt[:N])
Kv, Tv = kv(Hv, Yv)
print(f"пара (k,v): train {N:,}, val {len(Kv):,}, {time.time()-t0:.0f} с; дисперсия v "
      f"{float(Tv.var()):.4f}", flush=True)
res = {"D": D, "n_train": int(N), "n_val": int(len(Kv)),
       "spectrum": [float(x) for x in (sv_[:8] / sv_[0])]}


def relmse(W):
    p = Kv @ W
    return float(((p - Tv) ** 2).sum() / (Tv ** 2).sum())


t0 = time.time()
Wb = np.linalg.solve(Kt.T @ Kt + 1e-3 * np.eye(D), Kt.T @ Tt)
res["ridge_batch"] = {"rel": relmse(Wb), "sec": time.time() - t0, "tokens": int(N)}
print(f"ridge (пакетный LS, 1 проход, {time.time()-t0:.1f} с): rel-MSE {relmse(Wb):.6f}", flush=True)

t0 = time.time()
W = np.zeros((D, D))
P = np.eye(D)
lam = 1.0
for i in range(N):
    k = Kt[i].astype(np.float64)
    Pk = P @ k
    den = lam + k @ Pk
    g = Tt[i].astype(np.float64) - W.T @ k
    W += np.outer(Pk, g) / den
    P -= np.outer(Pk, Pk) / den
dt = time.time() - t0
r = relmse(W)
res["rls_1pass"] = {"rel": r, "sec": dt, "tokens": int(N)}
print(f"RLS   (1 проход, {dt:.0f} с, python-цикл): rel-MSE {r:.6f}  "
      f"(×{r/res['ridge_batch']['rel']:.2f} от оптимума)", flush=True)
Wrls = W.copy()
print(f"  ‖W_rls − W_ridge‖/‖W_ridge‖ = "
      f"{np.linalg.norm(Wrls - Wb) / np.linalg.norm(Wb):.2e}", flush=True)
res["rls_vs_ridge_rel"] = float(np.linalg.norm(Wrls - Wb) / np.linalg.norm(Wb))

print("\nAdam (то же W, тот же MSE) — сколько шагов надо, чтобы догнать 1 проход:", flush=True)
for steps in (1000, 10000, 100000):
    for lr in (3e-3, 1e-2, 3e-2, 1e-1):
        Wa = np.zeros((D, D))
        mm = np.zeros_like(Wa)
        vv = np.zeros_like(Wa)
        bs = 1024
        t0 = time.time()
        for s in range(steps):
            b0 = (s * bs) % max(1, N - bs)
            K = Kt[b0:b0 + bs].astype(np.float64)
            T = Tt[b0:b0 + bs].astype(np.float64)
            g = K.T @ (K @ Wa - T) / len(K)
            mm[:] = 0.9 * mm + 0.1 * g
            vv[:] = 0.999 * vv + 0.001 * g * g
            Wa -= lr * mm / (np.sqrt(vv) + 1e-12)
        rr = relmse(Wa)
        key = f"adam{steps}_lr{lr:g}"
        if key not in res or rr < res[key]["rel"]:
            res[key] = {"rel": rr, "lr": lr, "sec": time.time() - t0, "tokens": steps * bs}
        print(f"  {steps:6d} шагов × {bs} ток, lr={lr:5g} ({time.time()-t0:6.0f} с): "
              f"rel-MSE {rr:.6f}  ×{rr/Wb.dot(Wb) if False else rr/res['ridge_batch']['rel']:7.2f} "
              f"от оптимума; RLS за 1 проход = {res['rls_1pass']['rel']:.6f}", flush=True)
print(f"\nИТОГ: 1 проход RLS = {res['rls_1pass']['rel']:.6f} за {res['rls_1pass']['sec']:.0f} с; "
      f"лучший Adam 100k шагов = {min(v['rel'] for k, v in res.items() if k.startswith('adam100000')):.6f}",
      flush=True)
json.dump(res, open(os.path.expanduser("~/run1/rebind6.json"), "w"), indent=1, ensure_ascii=False)
print("готово: rebind6.json", flush=True)
