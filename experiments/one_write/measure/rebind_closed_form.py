"""Замкнутая правка строки головы С ПРИОРОМ (RLS-вид: w = (XᵀX+μI)⁻¹(Xᵀt + μw₀)).

Смысл: один проход (XᵀX и Xᵀt — достаточные статистики, собираются на лету), ноль эпох,
ноль обратных проходов; μ = точность приора (P₀ = I/μ), т.е. та же алгебра, что в lc_rls.
Сравниваем с Adam-эпохами на той же цели: 50/200/1000 шагов → 27.6/77.6/102.6% возврата
(измерено) за 21/86/223 с.
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
V, CTX = 8000, 96
tr = np.load(f"{LC}/data/prep/train.npy").astype(np.int64)
va = np.load(f"{LC}/data/prep/val.npy").astype(np.int64)
m = NL.NanoGPT(V, D=192, L=4, ff=576, kind="ema")
d = np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")
for k in m.p.d:
    if k in d.files:
        m.p.d[k][...] = d[k]
E0 = m.p.d["E"].astype(np.float64).copy()
Dm = m.D
cnt = np.bincount(tr, minlength=V)
S = np.sort(np.random.default_rng(7).choice(np.where((cnt >= 20) & (cnt <= 400))[0],
                                             size=1000, replace=False))
mp = np.arange(V)
mp[S] = np.roll(S, -1)
sel = np.zeros(V, bool)
sel[S] = True
Ht = np.load(f"{CACHE}/Ht.npy")
Yt = np.load(f"{CACHE}/Yt.npy")          # уже перебинденные цели
Hv = np.load(f"{CACHE}/Hv.npy")
Yv = np.load(f"{CACHE}/Yv.npy")
Yr = mp[Yv]
res = {"K": int(len(S)), "train_tokens": int(len(Yt))}


def nll(H, Y, W):
    tot = n = ssum = sn = 0
    for b in range(0, len(Y), 8192):
        sl = slice(b, min(b + 8192, len(Y)))
        lg = (H[sl].astype(np.float64) @ W).astype(np.float32)
        mx = lg.max(-1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(-1))
        nl = lse - lg[np.arange(len(lg)), Y[sl]]
        tot += float(nl.sum())
        n += len(nl)
        mk = sel[Y[sl]]
        ssum += float(nl[mk].sum())
        sn += int(mk.sum())
    return tot / n, ssum / max(1, sn)


base, bsel = nll(Hv, Yv, E0.T)
dmg, dsel = nll(Hv, Yr, E0.T)
print(f"гейт all {base:.4f} (надо 4.7218) | урон all {dmg:.4f} sel {dsel:.4f} (Δ {dsel-bsel:+.4f})",
      flush=True)
res.update(gate_all=base, gate_sel=bsel, dmg_all=dmg, dmg_sel=dsel)
if abs(base - 4.7218) > 0.02:
    sys.exit("гейт не пройден")


def report(tag, W, t, extra=""):
    a, s = nll(Hv, Yr, W)
    ac, sc = nll(Hv, Yv, W)
    rec = (dsel - s) / (dsel - bsel)
    print(f"  {tag:30s}: sel {s:7.4f} возврат {100*rec:7.1f}%  all {a:7.4f} | "
          f"чистый all {ac:.4f} (без урона {base:.4f}) {t:7.1f} с {extra}", flush=True)
    res[tag] = {"sel": s, "all": a, "recover": rec, "clean_all": ac, "sec": t}
    return rec


def solve_rows(m_max, mu, cap, neg_scale=0.25, anchor=True):
    """XᵀX и Xᵀt по каждой строке — ровно один проход; solve 192³ на строку."""
    t0 = time.time()
    uq, cts = np.unique(Yt, return_counts=True)
    order = np.argsort(Yt, kind="stable")
    st0 = np.concatenate([[0], np.cumsum(cts)])
    grp = {int(u): order[st0[i]:st0[i + 1]] for i, u in enumerate(uq)}
    W = E0.T.copy()
    nu = 0
    for l in S:
        ip = grp.get(int(l), np.zeros(0, np.int64))
        ine = grp.get(int(mp[l]), np.zeros(0, np.int64))
        if len(ip) > cap:
            ip = ip[:: len(ip) // cap + 1]
        if len(ine) > cap:
            ine = ine[:: len(ine) // cap + 1]
        if len(ip) + len(ine) < 2:
            continue
        X = np.concatenate([Ht[ip].astype(np.float64), Ht[ine].astype(np.float64)])
        t = np.concatenate([np.full(len(ip), m_max), np.full(len(ine), -neg_scale * m_max)])
        A = X.T @ X
        b = X.T @ t
        if anchor:
            A += mu * np.eye(Dm)
            b += mu * E0[l]
        else:
            A += mu * np.eye(Dm)
        W[:, l] = np.linalg.solve(A, b)
        nu += len(ip) + len(ine)
    return W, time.time() - t0, nu


print("\nзамкнутая правка с приором (0 эпох):", flush=True)
best = None
for m_max in (2.0, 4.0, 8.0):
    for mu in (10.0, 100.0, 1000.0):
        for cap in (400, 2000):
            W, dt, nu = solve_rows(m_max, mu, cap)
            r = report(f"ridge m={m_max:g} μ={mu:g} n≤{cap}", W, dt, f"[{nu:,} строк]")
            if best is None or r > best[0]:
                best = (r, m_max, mu, cap, dt)
print(f"\n  лучшее: возврат {100*best[0]:.1f}% при m={best[1]:g} μ={best[2]:g} cap={best[3]} "
      f"за {best[4]:.1f} с", flush=True)
res["best_closed_form"] = {"recover": best[0], "m": best[1], "mu": best[2], "cap": best[3],
                           "sec": best[4]}
# уточнение вторым проходом из той же достаточной статистики (prior = решённая строка)
W1, dt, _ = solve_rows(best[1], best[2], best[3])
W = W1.copy()
for it in (2, 3):
    W2 = W.copy()
    uq, cts = np.unique(Yt, return_counts=True)
    order = np.argsort(Yt, kind="stable")
    st0 = np.concatenate([[0], np.cumsum(cts)])
    grp = {int(u): order[st0[i]:st0[i + 1]] for i, u in enumerate(uq)}
    for l in S:
        ip = grp.get(int(l), np.zeros(0, np.int64))
        ine = grp.get(int(mp[l]), np.zeros(0, np.int64))
        X = np.concatenate([Ht[ip].astype(np.float64), Ht[ine].astype(np.float64)])
        t = np.concatenate([np.full(len(ip), best[1]), np.full(len(ine), -0.25 * best[1])])
        W2[:, l] = np.linalg.solve(X.T @ X + best[2] * np.eye(Dm),
                                   X.T @ t + best[2] * W[:, l])
    r = report(f"  +итерация {it}", W2, dt)
    W = W2
    if r > 1.02:
        break
json.dump(res, open(os.path.expanduser("~/run1/rebind5.json"), "w"), indent=1, ensure_ascii=False)
print("\nготово: rebind5.json", flush=True)
