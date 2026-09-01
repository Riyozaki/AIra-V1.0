"""Кто из чего: «выучить механизм SGD» против «решить память замкнуто, обучив только адресацию».

Отображение key->value случайное НА КАЖДОЙ последовательности -> в веса его зашить нельзя,
его можно только извлечь из контекста. Тестируем именно это: сколько ПРОХОДОВ ПО ДАННЫМ
нужно, чтобы на незнакомой последовательности вспоминать значения.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "delta_block"))
import lc_rls as R, lc_delta as DL

DK, DV = 16, 16          # размерность памяти
D = 2 * DK               # размерность потока: [key | value]


def make(rng, B, M, rounds=1):
    keys = rng.standard_normal((B, M, DK)); keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    keys = keys * np.sqrt(DK)                                # масштаб как у линейных слоёв
    vals = rng.standard_normal((B, M, DV)) * 0.5
    xs, ys = [], []
    for r in range(rounds):
        for i in range(M):
            xs.append(np.concatenate([keys[:, i], vals[:, i]], 1)); ys.append(vals[:, i])
    qs = [np.concatenate([keys[:, i], np.zeros((B, DV))], 1) for i in range(M)]
    Xt = np.stack(xs, 1); Yt = np.stack(ys, 1)                # study: и ключ, и значение
    Xq = np.stack(qs, 1); Yq = np.stack([vals[:, i] for i in range(M)], 1)
    return Xt, Yt, Xq, Yq


def rls_pred(Xt, Yt, Xq, Ek, Ev, Eq, lam=1e-2, mode="solve"):
    K = np.einsum("btd,de->bte", Xt[:, :, :DK], Ek)           # адресация ключей
    V = np.einsum("btd,de->bte", Xt[:, :, DK:], Ev)
    Q = np.einsum("btd,de->bte", Xq[:, :, :DK], Eq)
    if mode == "stream":                                       # 1 проход, без решателя
        Wb = np.empty((Xt.shape[0], K.shape[-1], V.shape[-1]))
        for b in range(Xt.shape[0]):
            Wb[b], _, _ = R.rls_write_seq(K[b], V[b], lam=lam)
    else:
        Wb = R.rls_solve_batch(K, V, lam=lam)
    return np.einsum("btd,bde->bte", Q, Wb)


def relerr(P, Y):
    return float(np.linalg.norm(P - Y) / np.linalg.norm(Y))


def train_rls(Xt, Yt, Xq, Yq, steps, lr=3e-2, seed=0):
    """Учим ТОЛЬКО адресацию, через замкнутый grad (ни одного обратного прохода по T)."""
    r = np.random.default_rng(seed)
    I = np.eye(DK)
    Ek, Ev, Eq = I.copy() + 0.1*r.standard_normal((DK, DK)), I.copy(), I.copy()
    m = {k: np.zeros((DK, DK)) for k in "kve"}
    v = {k: np.zeros((DK, DK)) for k in "kve"}
    t0 = time.time(); hist = []
    for s in range(steps + 1):
        P = rls_pred(Xt, Yt, Xq, Ek, Ev, Eq)
        dP = 2.0 * (P - Yq) / np.sum(Yq**2)
        hist.append(relerr(P, Yq))
        if s == steps:
            break
        Kx = np.einsum("btd,de->bte", Xt[:, :, :DK], Ek)
        Vx = np.einsum("btd,de->bte", Xt[:, :, DK:], Ev)
        W = R.rls_solve_batch(Kx, Vx, lam=1e-2)
        dW = np.einsum("btd,bte->bde", Q_ := np.einsum("btd,de->bte", Xq[:, :, :DK], Eq), dP)
        dK, dV = R.rls_grad_batch(Kx, Vx, W, dW, lam=1e-2)
        gk = np.einsum("btd,bte->de", Xt[:, :, :DK], dK)
        gv = np.einsum("btd,bte->de", Xt[:, :, DK:], dV)
        ge = np.einsum("btd,bte->de", Xq[:, :, :DK], np.einsum("btd,bde->bte", dP, W))
        for key, g in (("k", gk), ("v", gv), ("e", ge)):
            m[key] = .9*m[key] + .1*g; v[key] = .999*v[key] + .001*g*g
            cur = {"k": Ek, "v": Ev, "e": Eq}[key]
            cur -= lr * m[key] / (np.sqrt(v[key]) + 1e-8)
    return (Ek, Ev, Eq), time.time() - t0, hist


def train_delta(Xt, Yt, Xq, Yq, steps, seed=0, lr=3e-3):
    r = np.random.default_rng(seed)
    init, fwd, bwd = DL.BLOCKS["delta"]
    P = {k: v.astype(np.float64)*(1.0 if k.endswith(("sc", "ba", "bb")) else 0.5) for k, v in init(D, r).items()}
    mm = {k: np.zeros_like(v) for k, v in P.items()}; vv = dict(mm)
    X = np.concatenate([Xt, Xq], 1); Y = np.concatenate([np.zeros_like(Yt), Yq], 1)
    msk = np.concatenate([np.zeros((*Xt.shape[:2], 1)), np.ones((*Xq.shape[:2], 1))], 1)
    t0 = time.time(); hist = []
    for s in range(steps):
        Yh, ctx = fwd(X, P)
        dY = (Yh[:, Xt.shape[1]:, :DV] - Yq)*1.0
        dYf = np.zeros_like(Yh); dYf[:, Xt.shape[1]:, :DV] = dY
        hist.append(relerr(Yh[:, Xt.shape[1]:, :DV], Yq))
        dX, dP = bwd(dYf, ctx, P)
        for k in P:
            mm[k] = .9*mm[k] + .1*dP[k]; vv[k] = .999*vv[k] + .001*dP[k]**2
            P[k] -= lr*mm[k]/(np.sqrt(vv[k]) + 1e-8)
    Yh, _ = fwd(X, P)
    return Yh[:, Xt.shape[1]:, :DV], time.time() - t0, hist


def train_ema(Xt, Yt, Xq, Yq, steps, seed=0, lr=3e-3):
    r = np.random.default_rng(seed)
    init, fwd, bwd = DL.BLOCKS["ema"]
    P = {k: v.astype(np.float64) for k, v in init(D, r).items()}
    X = np.concatenate([Xt, Xq], 1)
    t0 = time.time()
    for s in range(steps):
        Yh, ctx = fwd(X, P)
        dYf = np.zeros_like(Yh); dYf[:, Xt.shape[1]:, :DV] = 2.0*(Yh[:, Xt.shape[1]:, :DV]-Yq)/np.sum(Yq**2)
        dX, dP = bwd(dYf, ctx, P)
        for k in P:
            P[k] -= lr*dP[k]/(np.sqrt(np.mean(dP[k]**2))+1e-8)
    Yh, _ = fwd(X, P)
    return Yh[:, Xt.shape[1]:, :DV], time.time()-t0


if __name__ == '__main__':
    for M in (12, 32):
        tag = "M<=Dk (памяти хватает)" if M <= DK else "M>Dk (перегруженная память: 32 ключей в 16-мерии)"
        print(f"\n=== {tag} ===")
        tr = np.random.default_rng(1); ev = np.random.default_rng(2)
        Xt, Yt, Xq, Yq = make(tr, B=16, M=M)
        Xe_t, Ye_t, Xe_q, Ye_q = make(ev, B=16, M=M)     # другие ключИ и значения = unseen mapping
        I = np.eye(DK)
        o = rls_pred(Xe_t, Ye_t, Xe_q, I, I, I); print(f"  {'RLS, 0 эпох (адресация=тождественная)':44s} 0 шагов  rel={relerr(o, Ye_q):.4f}")
        o = rls_pred(Xe_t, Ye_t, Xe_q, I, I, I, mode="stream"); print(f"  {'то же потоком (rls_write_seq, инференсный путь)':44s} 0 шагов  rel={relerr(o, Ye_q):.4f}")
        for st in (10, 50, 200):
            (Ek, Ev, Eq), dt, h = train_rls(Xt, Yt, Xq, Yq, st)
            o = rls_pred(Xe_t, Ye_t, Xe_q, Ek, Ev, Eq)
            print(f"  {'RLS + обученная адресация (no-BPTT)':44s} {st:6d}  rel={relerr(o, Ye_q):.4f}  {dt:5.1f}s")
        for st in (50, 300, 2000):
            o, dt, h = train_delta(Xt, Yt, Xq, Yq, st)
            print(f"  {'delta-rule, SGD (обучаем весь механизм)':44s} {st:6d}  rel(train)={h[-1]:.4f}  {dt:6.1f}s")
        for st in (300, 2000):
            o, dt = train_ema(Xt, Yt, Xq, Yq, st)
            print(f"  {'EMA (его CTRL), SGD':44s} {st:6d}  rel={relerr(o, Yq):.4f}  {dt:6.1f}s")
