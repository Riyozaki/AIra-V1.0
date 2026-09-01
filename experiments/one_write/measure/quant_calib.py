import os
"""Квантовый ущерб = задача калибровки, а не эпох. ПРОВЕРКА на его же модели.

Его рецепт: m.p.d[k] -> tern(k) для PATS (.fc1,.fc2,.Wm,.qkv,.Wqkv,.Wo),
tern(W) = clip(round(W/rowmean|W|),-1,1)*rowmean|W|. Zero-shot это роняет PPL;
он чинит это QAT-KL: 400 шагов с backward = 916 с до 125.62 (champ4krun) / 174.05 (lean).

Ставка: H_квант ≈ A·H_fp32 + b, т.е. достаточно ОДНОГО прохода (достаточные статистики
ZᵀZ, ZᵀT накапливаются на лету, 193×193) — без backward вообще.: поправка
схлопывается в голову: E_eff = E·A (undid head) + константа bEᵀ ⇒ ноль лишней работы
на инференсе, только +6.1 МБ на 8k-словарь (или 1.5 МБ в int8).

Метрики: его 6 окон (seed 1234) + полный свип val; гейт fp32 = 111.21 / 112.37.
"""
import json
import os
import sys
import time

import numpy as np

LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore"))
sys.path.insert(0, LC)
import nano_lc as NL  # noqa: E402

V, CTX, D = 8000, 96, 192
PATS = (".fc1", ".fc2", ".Wm", ".qkv", ".Wqkv", ".Wo")
tr = np.load(f"{LC}/data/prep/train.npy").astype(np.int64)
va = np.load(f"{LC}/data/prep/val.npy").astype(np.int64)
ck = np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")


def tern(W):
    s = np.abs(W).mean(-1, keepdims=True).clip(1e-5)
    return (np.clip(np.rint(W / s), -1, 1) * s).astype(W.dtype)


def build(quantize):
    m = NL.NanoGPT(V, D=D, L=4, ff=576, kind="ema")
    for k in m.p.d:
        if k in ck.files:
            m.p.d[k][...] = ck[k]
    n_q = 0
    if quantize:
        for k in list(m.p.d):
            if m.p.d[k].ndim == 2 and any(p in k for p in PATS):
                m.p.d[k][...] = tern(m.p.d[k])
                n_q += 1
    return m, n_q


mf, _ = build(False)
mq, nq = build(True)
E0 = mf.p.d["E"].astype(np.float64)
print(f"тернаризовано матриц: {nq} (у него 17 в torch-стейдже, здесь numpy-модель)", flush=True)


def windows(ids, stride=CTX, limit=None):
    st = list(range(0, len(ids) - CTX - 1, stride))
    return st[:limit] if limit else st


def xloss(model, ids, st, W=None, bias=None, chunk=64, ret_h=False):
    """NLL по окнам; W/bias — опциональная свёрнутая калибровка. Возвращает и H, если надо."""
    tot = 0.0
    n = 0
    Hs = [] if ret_h else None
    E = E0 if W is None else W
    for b in range(0, len(st), chunk):
        ss = st[b:b + chunk]
        x = np.stack([ids[s:s + CTX] for s in ss])
        y = np.stack([ids[s + 1:s + CTX + 1] for s in ss]).reshape(-1)
        H, _ = model.forward(x)
        Hf = H.reshape(-1, D).astype(np.float64)
        if ret_h:
            Hs.append(Hf.astype(np.float32))
        lg = Hf @ E.T
        if bias is not None:
            lg = lg + bias
        lg = lg.astype(np.float32)
        mx = lg.max(-1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(-1))
        tot += float((lse - lg[np.arange(len(lg)), y]).sum())
        n += len(y)
    return tot / n, (np.concatenate(Hs) if ret_h else None)


def his_metric(model, W=None, bias=None):
    rr = np.random.default_rng(1234)
    tot = 0.0
    for _ in range(6):
        st = rr.integers(0, len(va) - CTX - 1, size=24)
        x = np.stack([va[s:s + CTX] for s in st])
        y = np.stack([va[s + 1:s + CTX + 1] for s in st])
        H, _ = model.forward(x)
        lg = H.reshape(-1, D).astype(np.float64) @ (E0 if W is None else W).T
        if bias is not None:
            lg = lg + bias
        mx = lg.max(-1, keepdims=True)
        tot += float(np.mean(mx[:, 0] + np.log(np.exp(lg - mx).sum(-1))
                              - lg[np.arange(len(y.reshape(-1))), y.reshape(-1)]))
    return tot / 6


st_val = windows(va)
res = {}
t0 = time.time()
nf, _ = xloss(mf, va, st_val)
nh = his_metric(mf)
print(f"гейт fp32: его метрика {np.exp(nh):.2f} (надо 111.21), полный свип {np.exp(nf):.4f} "
      f"/ nll {nf:.4f}  [{time.time()-t0:.0f} с]", flush=True)
res["fp32"] = {"his_ppl": float(np.exp(nh)), "sweep_nll": nf, "sweep_ppl": float(np.exp(nf))}

nq_l, _ = xloss(mq, va, st_val)
nq_h = his_metric(mq)
print(f"ternary zero-shot: его метрика {np.exp(nq_h):.2f}, свип nll {nq_l:.4f} "
      f"(PPL {np.exp(nq_l):.2f}) — урон {(nq_l-nf)/nf*100:+.1f}% NLL", flush=True)
res["ternary"] = {"his_ppl": float(np.exp(nq_h)), "sweep_nll": nq_l, "sweep_ppl": float(np.exp(nq_l))}

# ---- один проход: достаточные статистики по train ----
print("\nкалибровка в один проход (2 форварда по train, 0 обратных проходов):", flush=True)
t0 = time.time()
st_tr = windows(tr, limit=9000)
D1 = D + 1
Szz = np.zeros((D1, D1))
Szt = np.zeros((D1, D))
Stt = np.zeros((D, D))
for b in range(0, len(st_tr), 64):
    ss = st_tr[b:b + 64]
    x = np.stack([tr[s:s + CTX] for s in ss])
    zq, _ = mq.forward(x)
    zf, _ = mf.forward(x)
    Z = np.concatenate([zq.reshape(-1, D).astype(np.float64),
                        np.ones((zq.size // D, 1))], 1)
    T = zf.reshape(-1, D).astype(np.float64)
    Szz += Z.T @ Z
    Szt += Z.T @ T
    Stt += T.T @ T
    del Z, T, zq, zf
n = float(len(st_tr) * CTX)
print(f"  статистика собрана за {time.time()-t0:.0f} с на {n:,} позиций", flush=True)
res["fit_positions"] = int(n)
np.savez(os.path.expanduser("~/run1/qstats.npz"), Szz=Szz, Szt=Szt, Stt=Stt, n=n)
print("  статистики сохранены (qstats.npz)", flush=True)
for lam in (1e-4, 1e-2, 1.0):
    Wa = np.linalg.solve(Szz + (lam * n / D) * np.eye(D1), Szt)            # (D+1, D)
    M, bias = Wa[:D], Wa[D]                                                 # Ĥ = H_q M + bias
    We = E0 @ M.T                                                          # свёрнутая голова (Ĥ = H_q M)
    c = E0 @ bias
    a, _ = xloss(mq, va, st_val, W=We, bias=c)
    ah = his_metric(mq, W=We, bias=c)
    print(f"  LS-калибровка λ={lam:6g}: nll {a:.4f}  PPL {np.exp(a):.2f} "
          f"(его метрика {np.exp(ah):.2f})  возврат урона "
          f"{100*(nq_l-a)/(nq_l-nf):5.1f}%", flush=True)
    res[f"cal_lam{lam:g}"] = {"sweep_nll": a, "sweep_ppl": float(np.exp(a)),
                              "his_ppl": float(np.exp(ah)),
                              "recover": (nq_l - a) / (nq_l - nf)}
Wa = np.linalg.solve(Szz + (1e-2 * n / D) * np.eye(D1), Szt)
M, bias = Wa[:D], Wa[D]
We = E0 @ M.T
c = E0 @ bias
json.dump(res, open(os.path.expanduser("~/run1/quantfix.json"), "w"), indent=1, ensure_ascii=False)

# ---- (c) рефит ВСЕЙ головы под логиты fp32-учителя, замкнуто, 0 backward ----
# min_W ||H_q Wᵀ − H_f E0ᵀ||²  ⇒  W = E0 Gᵀ,  G = (Sqq+rI)⁻¹(Sqf+r_a I)
print("\nзамкнутый рефит головы под логиты учителя (0 эпох, 0 backward):", flush=True)
Sqq = Szz[:D, :D]
Sqf = Szt[:D, :]
for lam in (1e-3, 1e-2, 1e-1):
    for anch in (0.0, 1e-2, 1e-1):
        r = lam * n / D
        ra = anch * n / D
        G = np.linalg.solve(Sqq + (r + ra) * np.eye(D), Sqf + ra * np.eye(D))
        We2 = E0 @ G.T
        a, _ = xloss(mq, va, st_val, W=We2)
        ah = his_metric(mq, W=We2)
        print(f"  head-refit λ={lam:6g} anchor={anch:5g}: nll {a:.4f} PPL {np.exp(a):7.2f} "
              f"(его метрика {np.exp(ah):7.2f}) возврат {100*(nq_l-a)/(nq_l-nf):6.1f}%", flush=True)
        res[f"head_lam{lam:g}_a{anch:g}"] = {"sweep_nll": a, "sweep_ppl": float(np.exp(a)),
                                             "his_ppl": float(np.exp(ah)),
                                             "recover": (nq_l - a) / (nq_l - nf)}
json.dump(res, open(os.path.expanduser("~/run1/quantfix.json"), "w"), indent=1, ensure_ascii=False)

# ---- вариант «только диагональная поправка» (0.4% параметров, почти бесплатно) ----
num = np.zeros(D)
den = np.zeros(D)
for b in range(0, min(len(st_tr), 3000), 64):
    ss = st_tr[b:b + 64]
    x = np.stack([tr[s:s + CTX] for s in ss])
    zq, _ = mq.forward(x)
    zf, _ = mf.forward(x)
    Z = zq.reshape(-1, D).astype(np.float64)
    T = zf.reshape(-1, D).astype(np.float64)
    num += (Z * T).sum(0)
    den += (Z * Z).sum(0)
gamma = num / np.maximum(den, 1e-9)
a2, _ = xloss(mq, va, st_val, W=(E0 * gamma), bias=None)
print(f"  только поканальные gains γ (192 числа): nll {a2:.4f} PPL {np.exp(a2):.2f} "
      f"возврат {100*(nq_l-a2)/(nq_l-nf):5.1f}%", flush=True)
res["diag_gains"] = {"sweep_nll": a2, "sweep_ppl": float(np.exp(a2)),
                     "recover": (nq_l - a2) / (nq_l - nf), "gamma_mean": float(gamma.mean())}
json.dump(res, open(os.path.expanduser("~/run1/quantfix.json"), "w"), indent=1, ensure_ascii=False)
print("\nразмер: fp32-голова", E0.nbytes / 1e6, "МБ; int8-версия", E0.size / 1e6, "МБ "
      "(цена undid-калибровки)", flush=True)
print("готово: quantfix.json", flush=True)
