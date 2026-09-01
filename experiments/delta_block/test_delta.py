"""Проверки lc_delta: (1) FD-gradcheck всех backward, (2) «чанк за чанком == целиком»
(потоковый инференс обязан оставаться точным, как у EMA), (3) iso-cost таблица.

    python3 test_delta.py          # exit 0 = всё сходится
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import lc_delta as L  # noqa: E402

rng = np.random.default_rng(0)
B, T, D = 2, 9, 12
dt = np.float64
X = rng.standard_normal((B, T, D)).astype(dt)
Wt = rng.standard_normal((B, T, D)).astype(dt)


def loss(Y):
    return float(np.einsum("btd,btd->", Y, Wt))


def fd_grad(name, P, X, eps=1e-6):
    """возвращает ОТНОСИТЕЛЬНУЮ max-ошибку по каждому имени (масштаб = max|числ. градиента|)"""
    init, fwd, bwd = L.BLOCKS[name]
    Y, ctx = fwd(X, P)
    dx, dp = bwd(Wt, ctx, P)
    out = {}
    for k, arr in P.items():
        num = np.zeros_like(arr)
        for i in np.ndindex(arr.shape):
            old = arr[i]
            arr[i] = old + eps
            fp = loss(fwd(X, P)[0])
            arr[i] = old - eps
            fm = loss(fwd(X, P)[0])
            arr[i] = old
            num[i] = (fp - fm) / (2 * eps)
        out[k] = float(np.max(np.abs(num - dp[k])))
    numX = np.zeros_like(X)
    for i in np.ndindex(X.shape):
        old = X[i]
        X[i] = old + eps
        fp = loss(L.BLOCKS[name][1](X, P)[0])
        X[i] = old - eps
        fm = loss(L.BLOCKS[name][1](X, P)[0])
        X[i] = old
        numX[i] = (fp - fm) / (2 * eps)
    out["X"] = float(np.max(np.abs(numX - dx)))
    scale = max(1.0, float(np.max(np.abs(num))))
    return {k: v / scale for k, v in out.items()}


def chunk_parity(name, P, X, cuts=(3, 4, 2)):
    fwd = L.BLOCKS[name][1]
    Y, _ = fwd(X, P)
    y = np.zeros_like(Y)
    init, a = None, 0
    sizes = list(cuts)
    while sum(sizes[: len(sizes) - 1]) < X.shape[1]:      # хвост добить остатком
        if sum(sizes) < X.shape[1]:
            sizes.append(X.shape[1] - sum(sizes))
        break
    for n in sizes:
        c = a + n
        Yc, ctx = fwd(X[:, a:c], P, init)
        y[:, a:c] = Yc
        init = ctx["last"]
        a = c
    return float(np.max(np.abs(Y - y)))


ok = True
for name in ("dddecay", "rot", "delta"):
    P = {k: v.copy() for k, v in L.BLOCKS[name][0](D, rng).items()}
    err = fd_grad(name, P, X)
    if name == "delta" and False:          # сходимость шага: ошибка должна падать как O(ε²)
        e1 = max(fd_grad(name, {k: v.copy() for k, v in P.items()}, X.copy(), eps=1e-4).values())
        e2 = max(fd_grad(name, {k: v.copy() for k, v in P.items()}, X.copy(), eps=1e-6).values())
        print(f"         ε-сходимость: err(1e-4)={e1:.2e} → err(1e-6)={e2:.2e} "
              f"(×{e1/max(e2,1e-300):.0f}, ожидание ≫1 ⇒ баг не метод)")
    worst = max(err.values())
    par = chunk_parity(name, P, X)
    fin = bool(np.isfinite(L.BLOCKS[name][1](X, P)[0]).all())
    print(f"{name:8s} FD  max|аналитика−FD| = {worst:.2e}   "
          + "  ".join(f"{k}:{v:.1e}" for k, v in err.items()))
    print(f"{'':8s} чанк-vs-полный max|Δ| = {par:.2e}   finite={fin}")
    if name in L.CHUNKED:          # float32: относительная сходимость чанк-формы
        Xf = X.astype(np.float32)
        Pf = {k: v.astype(np.float32) for k, v in P.items()}
        Yf = L.BLOCKS[name][1](Xf, Pf)[0]
        Ycf = L.CHUNKED[name](Xf, Pf, C=4)[0]
        relf = float(np.max(np.abs(Yf - Ycf)) / max(np.max(np.abs(Yf)), 1e-30))
        print(f"{'':8s} float32 чанк-vs-рекурр относит. |Δ| = {relf:.2e}"
              + ("  ✗" if relf > 2e-4 else "  (OK: разная ассоциативность сумм)"))
        if relf > 2e-4:
            ok = False
        Yr = L.BLOCKS[name][1](X, P)[0]
        for CC in (1, 3, 4, 128):
            Yc = L.CHUNKED[name](X, P, C=CC)[0]
            dp = float(np.max(np.abs(Yr - Yc)))
            print(f"{'':8s} chunked(C={CC:3d}) vs рекуррентно: max|Δ|={dp:.2e}"
                  + ("  ✗" if dp > 1e-9 else ""))
            if dp > 1e-9:
                ok = False
    if worst > 5e-6 or par > 1e-9 or not fin:
        ok = False
        print(f"{'':8s} ✗ ПОРОГ ПРЕВЫШЕН (5e-6 / 1e-9)")

print("\niso-cost таблица (D=%d, T=%d) — A/B обязан равнять MAC, а не 'кто красивее'" % (D, T))
hdr = f"  {'блок':9s} {'MAC/токен':>10s} {'сост-я/ток':>10s} {'пар/блок':>9s} {'к attn':>7s}"
print(hdr)
a_mac = L.cost_per_token("attn", D, T)["mac"]
for nm in ("ema", "dddecay", "rot", "delta", "attn"):
    c = L.cost_per_token(nm, D, T)
    pp = L.params_per_block(nm, D)
    print(f"  {nm:9s} {c['mac']:10d} {c['state']:10d} {pp:9d} {c['mac']/a_mac:6.2f}×")

# длинная последовательность: устойчивость (δ-правка не должна взрываться)
P = {k: v.copy() for k, v in L.BLOCKS["delta"][0](16, rng).items()}
Y, ctxd = L.BLOCKS["delta"][1](rng.standard_normal((2, 2048, 16)), P)
snorm = np.abs(ctxd["Sall"]).max(-1).max(-1)
print(f"\n  delta @T=2048: finite={np.isfinite(Y).all()}, |Y|max={np.abs(Y).max():.3g}, "
      f"|S|: t=1 {snorm[0,0]:.2f} → t=2048 {snorm[0,-1]:.2f} (рост ≫1 = расходимость)")
print("\n" + ("✓ все три блока: градиенты сходятся, потоковый инференс точен"
             if ok else "✗ есть расхождения — не запускать A/B, пока не исправлено"))
sys.exit(0 if ok else 1)
