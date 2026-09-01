"""Финальный замер: аддитивный аналитический канал (low-rank adapter, 1 проход RLS)
поверх ЗАМОРОЖЕННОЙ его модели. Строки головы не переписываются — остальной корпус не
портится; канал добавляет logits += α·(u_t Rᵀ) Eᵀ, u_t = Ŵᵀ k_t, Ŵ решён замкнуто за 1 проход.

Сравниваем: (a) α=0 (frozen, урон), (b) канал (0 эпох, 0 backward), (c) Adam по строкам S
(1000 шагов = 2.05M токенов = 2 эпохи, 223 с — измерено ранее: 102.6% возврата).
α подбирается по ОТЛОЖЕННОЙ части train (не по val!), чтобы val оставался честным.
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
D = 64
tr = np.load(f"{LC}/data/prep/train.npy").astype(np.int64)
m = NL.NanoGPT(8000, D=192, L=4, ff=576, kind="ema")
d = np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")
for k in m.p.d:
    if k in d.files:
        m.p.d[k][...] = d[k]
E0 = m.p.d["E"].astype(np.float64)
V = E0.shape[0]
cnt = np.bincount(tr, minlength=V)
S = np.sort(np.random.default_rng(7).choice(np.where((cnt >= 20) & (cnt <= 400))[0],
                                            size=1000, replace=False))
mp = np.arange(V)
mp[S] = np.roll(S, -1)
sel = np.zeros(V, bool)
sel[S] = True
Ht = np.load(f"{CACHE}/Ht.npy").astype(np.float64)
Yt = np.load(f"{CACHE}/Yt.npy")
Hv = np.load(f"{CACHE}/Hv.npy").astype(np.float64)
Yv = np.load(f"{CACHE}/Yv.npy")
Yr = mp[Yv]
res = {"D": D, "K": int(len(S)), "train_tokens": int(len(Yt)), "val_tokens": int(len(Yv))}


def basis(H, d=D):
    sub = np.arange(0, len(H), max(1, len(H) // 20000))
    X = H[sub] - H[sub].mean(0)
    R = np.linalg.svd(X, full_matrices=False)[2][:d].T
    return R


def rls_fit(K, Ytarg, lam=1.0, ridge=1e-2):
    W = np.zeros((K.shape[1], Ytarg.shape[1]))
    P = np.eye(K.shape[1]) / ridge
    for i in range(len(K)):
        k = K[i]
        Pk = P @ k
        den = lam + k @ Pk
        g = Ytarg[i] - W.T @ k
        W += np.outer(Pk, g) / den
        P -= np.outer(Pk, Pk) / den
    return W


def keys(H, R):
    Z = H @ R
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def eval_nll(H, Y, alpha, Wm, R):
    tot = n = ssum = sn = 0
    for b in range(0, len(Y), 8192):
        sl = slice(b, min(b + 8192, len(Y)))
        u = keys(H[sl], R) @ Wm
        lg = (H[sl] + alpha * (u @ R.T)) @ E0.T
        lg = lg.astype(np.float32)
        mx = lg.max(-1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(-1))
        nl = lse - lg[np.arange(len(lg)), Y[sl]]
        tot += float(nl.sum())
        n += len(nl)
        mk = sel[Y[sl]]
        ssum += float(nl[mk].sum())
        sn += int(mk.sum())
    return tot / n, ssum / max(1, sn)


t0 = time.time()
nh = int(len(Ht) * 0.15)             # первые 15% окон — только подбирать α, не учить
R = basis(Ht[nh:])
Kt = keys(Ht, R)
Vt = E0[Yt] @ R                      # цель = проекция эмбеддинга следующего токена
Wm = rls_fit(Kt[nh:], Vt[nh:], lam=1.0)
print(f"RLS-канал: 1 проход по {len(Kt):,} токенам, {time.time()-t0:.0f} с; "
      f"обратных проходов 0", flush=True)
res["rls_sec"] = time.time() - t0
print("\nподбор α на отложенном куске train (не val):", flush=True)
best = (None, None)
for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    a, s = eval_nll(Ht[:nh], Yt[:nh], alpha, Wm, R)
    print(f"  α={alpha:4.2f}: hold-train all {a:.4f}  sel {s:.4f}", flush=True)
    if best[0] is None or s < best[1]:
        best = (alpha, s)
A = best[0]
print(f"  → выбран α={A:g}", flush=True)
res["alpha"] = A
print("\nоценка на val (перебинденный корпус):", flush=True)
for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, A):
    a, s = eval_nll(Hv, Yr, alpha, Wm, R)
    ac, sc = eval_nll(Hv, Yv, alpha, Wm, R)
    print(f"  α={alpha:4.2f}: rebound all {a:.4f} sel {s:.4f} | чистый all {ac:.4f} sel {sc:.4f}",
          flush=True)
    res[f"rebound_a{alpha:g}"] = {"all": a, "sel": s, "clean_all": ac, "clean_sel": sc}
base_all, base_sel = eval_nll(Hv, Yv, 0.0, Wm, R)
dmg_all, dmg_sel = eval_nll(Hv, Yr, 0.0, Wm, R)
for alpha in (0.5, 1.0, 2.0, A):
    a, s = res[f"rebound_a{alpha:g}"]["sel"], res[f"rebound_a{alpha:g}"]["all"]
    rec = (dmg_sel - res[f"rebound_a{alpha:g}"]["sel"]) / (dmg_sel - base_sel)
    res[f"recover_a{alpha:g}"] = rec
print(f"\nбаза (чистый val): all {base_all:.4f} sel {base_sel:.4f}\n"
      f"урон (перебинд, α=0): all {dmg_all:.4f} sel {dmg_sel:.4f}  Δ {dmg_sel-base_sel:+.4f}",
      flush=True)
for alpha in (0.5, 1.0, 2.0, A):
    k = f"rebound_a{alpha:g}"
    rec = (dmg_sel - res[k]["sel"]) / (dmg_sel - base_sel)
    print(f"  α={alpha:g}: возврат {100*rec:6.1f}% урона  (чистый val при этом "
          f"{res[k]['clean_all']:.4f} vs {base_all:.4f})", flush=True)
    res[f"recover_a{alpha:g}"] = rec
json.dump(res, open(os.path.expanduser("~/run1/rebind7.json"), "w"), indent=1, ensure_ascii=False)
print("\nготово: rebind7.json", flush=True)
