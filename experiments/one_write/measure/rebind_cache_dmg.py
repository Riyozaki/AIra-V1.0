"""Решающий замер: замкнутая попозная правка строки против эпох.

Ставка в чистом виде: связать токен с контекстом = линейная задача на 192 размера.
  * «эпохи»: Adam по строкам S, 100..1000 шагов × 2048 токенов;
  * «0 эпох»: для каждой строки l ∈ S одно решение (XᵀX+μI) w = Xᵀt, где X = позитивные
    позиции токена и его же «самоуверенные ошибки» (argmax == l, а цели другие) с
    целями ±m. Один проход по данным, обратного распространения нет вообще.
Если второе догонит первое — ставка ×100 на «один проход вместо эпох» подтверждена
на его модели и его корпусе. Если нет — ставка на стволе/голове умирает, и это надо знать.
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
os.makedirs(CACHE, exist_ok=True)
V, CTX, LIM = 8000, 96, 9000
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


def feats(ids, limit=None, chunk=48):
    st = list(range(0, len(ids) - CTX - 1, CTX))
    st = st[:limit] if limit else st
    H, Y = [], []
    for b in range(0, len(st), chunk):
        ss = st[b:b + chunk]
        x = np.stack([ids[s:s + CTX] for s in ss])
        y = np.stack([ids[s + 1:s + CTX + 1] for s in ss])
        h, _ = m.forward(x)
        H.append(h.reshape(-1, Dm).astype(np.float32))
        Y.append(y.reshape(-1))
    return np.concatenate(H), np.concatenate(Y)


if os.path.exists(f"{CACHE}/Ht.npy"):
    Ht = np.load(f"{CACHE}/Ht.npy")
    Yt = np.load(f"{CACHE}/Yt.npy")
    Hv = np.load(f"{CACHE}/Hv.npy")
    Yv = np.load(f"{CACHE}/Yv.npy")
    print("кэш H использован", flush=True)
else:
    t0 = time.time()
    Hv, Yv = feats(va)
    Ht, Yt = feats(mp[tr], limit=LIM)
    np.save(f"{CACHE}/Hv.npy", Hv)
    np.save(f"{CACHE}/Yv.npy", Yv)
    np.save(f"{CACHE}/Ht.npy", Ht)
    np.save(f"{CACHE}/Yt.npy", Yt)
    print(f"прогоны ствола: val {len(Yv):,} + train {len(Yt):,} ток за {time.time()-t0:.0f} с",
          flush=True)
Yr = mp[Yv]


def nll(H, Y, W):
    tot = n = 0
    ssum = sn = 0
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
    return tot / n, ssum / max(1, sn), sn


base, bsel, ns = nll(Hv, Yv, E0.T)
dmg, dsel, _ = nll(Hv, Yr, E0.T)
print(f"гейт all {base:.4f} (надо 4.7218) | урон: all {dmg:.4f} sel {dsel:.4f} "
      f"(Δ {dsel-bsel:+.4f}, {ns:,} позиций)", flush=True)
res = {"gate_all": base, "gate_sel": bsel, "dmg_all": dmg, "dmg_sel": dsel,
       "n_sel": int(ns), "K": int(len(S))}
if abs(base - 4.7218) > 0.02:
    sys.exit("гейт не пройден")


def report(tag, W, t):
    a, s, _ = nll(Hv, Yr, W)
    ac, sc, _ = nll(Hv, Yv, W)
    rec = (dsel - s) / (dsel - bsel)
    print(f"  {tag:34s}: sel {s:7.4f} возврат {100*rec:7.1f}%  all {a:7.4f}  "
          f"[на чистом val: all {ac:.4f} sel {sc:.4f}]  {t:6.1f} с", flush=True)
    res[tag] = {"sel": s, "all": a, "recover": rec, "clean_all": ac, "clean_sel": sc,
                "sec": t}
    return rec


Wo = E0.T.copy()
Wo[:, mp[S]] = E0[S].T
report("oracle-копия (потолок)", Wo, 0.0)


def adam_rows(steps, lr=3e-4, bs=2048):
    W = E0.T.copy()
    mm = np.zeros_like(W)
    vv = np.zeros_like(W)
    n = len(Yt)
    t0 = time.time()
    for s in range(steps):
        b0 = (s * bs) % max(1, n - bs)
        sl = slice(b0, b0 + bs)
        Hf = Ht[sl].astype(np.float64)
        lg = (Hf @ W).astype(np.float32)
        mx = lg.max(-1, keepdims=True)
        p = np.exp(lg - mx)
        p /= p.sum(-1, keepdims=True)
        p = p.astype(np.float64)
        p[np.arange(len(p)), Yt[sl]] -= 1.0
        g = np.zeros_like(W)
        g[:, S] = (Hf.T @ p)[:, S] / bs
        mm[:] = 0.9 * mm + 0.1 * g
        vv[:] = 0.999 * vv + 0.001 * g * g
        W -= lr * mm / (np.sqrt(vv) + 1e-8)
    return W, time.time() - t0, steps * bs


for steps in (50, 200, 1000):
    W, dt, tok = adam_rows(steps)
    report(f"Adam строки S, {steps} шагов", W, dt)
    res[f"adam{steps}_tok"] = tok


def closed_form(m_max=4.0, mu=1.0, cap=2000, neg_scale=0.25):
    """Ноль эпох, ноль обратных проходов: для строки l решаем попозный ridge
    на её новых позитивах и на позициях, где прежнее знание теперь неверно
    (там цель = mp[l], значит строку l надо подавить)."""
    t0 = time.time()
    uq, cnts = np.unique(Yt, return_counts=True)
    order = np.argsort(Yt, kind="stable")
    st0 = np.concatenate([[0], np.cumsum(cnts)])
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
        W[:, l] = np.linalg.solve(X.T @ X + mu * np.eye(Dm), X.T @ t)
        nu += len(ip) + len(ine)
    return W, time.time() - t0, nu


for m_max in (2.0, 4.0, 8.0, 16.0):
    for mu in (0.3, 3.0):
        W, dt, nuse = closed_form(m_max=m_max, mu=mu)
        r = report(f"ridge m={m_max:g} μ={mu:g} (0 эпох)", W, dt)
        res[f"cf_m{m_max:g}_mu{mu:g}"]["n_used"] = int(nuse)
        res[f"cf_m{m_max:g}_mu{mu:g}"]["recover"] = r
json.dump(res, open(os.path.expanduser("~/run1/rebind4.json"), "w"), indent=1, ensure_ascii=False)
print("\nготово: rebind4.json", flush=True)
