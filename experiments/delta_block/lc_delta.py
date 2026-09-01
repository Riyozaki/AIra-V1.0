"""lc_delta.py — три апгрейда миксера LeanCore (вместо time-invariant EMA).

Контакт: `fwd(X, P, init=None) -> (Y, ctx)` и `bwd(dY, ctx, P) -> (dX, dP)`;
X: (B,T,D); P — dict массивов numpy (float64 в тестах, float32 в бою); `init` —
начальное состояние (для потокового/чанкового прогона, см. test_delta.py).

Зачем эти три. В `nano_lc.py` (~стр. 268) `a = sigmoid(th)`, где `th` — *параметр*,
а не функция входа: диагональная рекуррентность без управления данными и без
коррективающего правила = самый слабый член семейства (S4-diag / RetNet без гейта).
Три оси модернизации, каждая дешевле attention:

  A `dddecay`  a_t = σ(W_a x_t + b): data-dependent decay (GLA-класс).
               +D² MAC/токен, состояние O(D). Ожидание: почти бесплатный выигрыш.
  B `rot`      λ = ρ·e^{iω} — комплексная диагональ в вещественной арифметике 2×2
               (Mamba-3-класс: фаза вместо чистого затухания). +~8D MAC/токен,
               состояние O(D). Лечит state-tracking/чётность — то, на чём
               диагональные модели ломаются первыми.
  C `delta`    (gated) delta rule с матричным состоянием:
                   S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ,  o_t = q_tᵀ S_t
               «запиши, но сначала вытрешь старое значение по этому ключу» — механизм
               DeltaNet/GDN/KDA. ~4D² MAC/токен, состояние O(D²): платишь памятью за память.

Рекуррентные форварды намеренно «медленные» (цикл по T): это ЭТАЛОН. Бой обязан считать
чанк-за-чанком и совпадать с эталоном в пределах плавающей точки — ровно так, как
проверен потоковый EMA в MATH.md §11. Backward выведен аналитически, сверен
конечными разностями (float64, ε=1e-6).
"""
from __future__ import annotations

import numpy as np

sig = lambda x: 1.0 / (1.0 + np.exp(-x))


# ============================================================ A: dddecay ====
def dddecay_init(D, rng):
    return dict(Wa=rng.standard_normal((D, D)) * (D ** -0.5), ba=np.full(D, -1.0),
                sc=np.ones(D))


def dddecay_fwd(X, P, init=None):
    """h_t = a_t h_{t−1} + (1−a_t) x_t,  a_t = σ(W_a x_t + b),  y_t = sc ⊙ h_t"""
    B, T, D = X.shape
    a = sig(X @ P["Wa"] + P["ba"])
    H = np.empty_like(X)
    h = np.zeros((B, D), X.dtype) if init is None else np.array(init, dtype=X.dtype)
    for t in range(T):
        h = a[:, t] * h + (1.0 - a[:, t]) * X[:, t]
        H[:, t] = h
    return H * P["sc"], dict(a=a, H=H, X=X, last=h)


def dddecay_bwd(dY, ctx, P):
    a, H, X = ctx["a"], ctx["H"], ctx["X"]
    B, T, D = H.shape
    sc = P["sc"]
    dH = dY * sc
    dsc = np.einsum("btd,btd->d", dY, H)
    dX = np.zeros_like(H)
    da = np.zeros_like(a)
    # сопряжённое: λ_t = dL/dh_t = dH_t + a_{t+1} λ_{t+1}
    lam = np.zeros((B, D), dY.dtype)
    for t in range(T - 1, -1, -1):
        lam = dH[:, t] + (a[:, t + 1] * lam if t + 1 < T else 0.0)
        hprev = H[:, t - 1] if t > 0 else np.zeros((B, D), H.dtype)
        da[:, t] = lam * (hprev - X[:, t])          # ∂h_t/∂a_t = h_{t−1} − x_t
        dX[:, t] = lam * (1.0 - a[:, t])             # ∂h_t/∂x_t = 1 − a_t
    ds = da * a * (1.0 - a)                          # σ' через саму σ
    dX = dX + ds @ P["Wa"].T
    dP = dict(Wa=np.einsum("btd,bte->de", X, ds), ba=ds.sum((0, 1)), sc=dsc)
    return dX, dP


# ============================================================== B: rot ======
# h1_t = ρ(c h1_{t−1} − s h2_{t−1}) + g x_t ;  h2_t = ρ(s h1_{t−1} + c h2_{t−1})
# y_t = sc h1_t ;  ρ = exp(logr) < 1,  c = cos w, s = sin w,  g = 1 − ρ
def rot_init(D, rng):
    return dict(logr=np.full(D, -0.5), w=rng.uniform(-0.6, 0.6, D), sc=np.ones(D))


def rot_fwd(X, P, init=None):
    B, T, D = X.shape
    rho, w = np.exp(P["logr"]), P["w"]
    c, s, g = np.cos(w), np.sin(w), 1.0 - np.exp(P["logr"])
    h = np.zeros((B, 2, D), X.dtype) if init is None else np.array(init, dtype=X.dtype)
    H = np.empty((B, T, 2, D), X.dtype)
    for t in range(T):
        h1, h2 = h[:, 0], h[:, 1]
        nh = np.empty((B, 2, D), X.dtype)
        nh[:, 0] = rho * (c * h1 - s * h2) + g * X[:, t]
        nh[:, 1] = rho * (s * h1 + c * h2)
        h = nh
        H[:, t] = h
    return H[:, :, 0] * P["sc"], dict(H=H, X=X, rho=rho, c=c, s=s, g=g, last=h)


def rot_bwd(dY, ctx, P):
    H, X, rho, c, s, g = ctx["H"], ctx["X"], ctx["rho"], ctx["c"], ctx["s"], ctx["g"]
    sc = P["sc"]
    B, T, _, D = H.shape
    dX = np.zeros((B, T, D), dY.dtype)
    dlogr = np.zeros(D, dY.dtype)
    dw = np.zeros(D, dY.dtype)
    dsc = np.einsum("btd,btd->d", dY, H[:, :, 0])
    l1 = np.zeros((B, D), dY.dtype)      # λ_{t+1} по h1
    l2 = np.zeros((B, D), dY.dtype)      # λ_{t+1} по h2
    for t in range(T - 1, -1, -1):
        l1 = l1 + sc * dY[:, t]
        p1, p2 = (H[:, t - 1, 0], H[:, t - 1, 1]) if t > 0 else (
            np.zeros((B, D), H.dtype), np.zeros((B, D), H.dtype))
        # ∂h1_t/∂ρ = c p1 − s p2 ; ∂h2_t/∂ρ = s p1 + c p2 ; ∂h1_t/∂g = x_t ; dρ/dlogr = ρ
        dlogr += rho * np.einsum("bd,bd->d", l1, c * p1 - s * p2) \
            + rho * np.einsum("bd,bd->d", l2, s * p1 + c * p2) \
            - rho * np.einsum("bd,bd->d", l1, X[:, t])
        # ∂h1_t/∂w = ρ(−s p1 − c p2) ; ∂h2_t/∂w = ρ(c p1 − s p2)
        dw += rho * np.einsum("bd,bd->d", l1, -s * p1 - c * p2) \
            + rho * np.einsum("bd,bd->d", l2, c * p1 - s * p2)
        dX[:, t] = g * l1
        n1 = rho * (c * l1 + s * l2)                 # λ_{t−1} = Aᵀ λ_t
        n2 = rho * (-s * l1 + c * l2)
        l1, l2 = n1, n2
    return dX, dict(logr=dlogr, w=dw, sc=dsc)


# ============================================================ C: delta =====
# K = l2norm(X Wk), V = X Wv, Q = X Wq ;  β = σ(X Wb + bb), α = σ(X Wa + ba)  (скаляр/токен)
# u_t = S_{t−1}ᵀ k_t ;  S_t = α_t (S_{t−1} − β_t k_t u_tᵀ) + β_t k_t v_tᵀ ;
# o_t = q_tᵀ S_t ;  y_t = sc ⊙ (o_t Wo)
# l2-нормализация k — НЕ косметика, а условие устойчивости: без неё собственный вектор
# (I − β k kᵀ) равен 1 − β|k|² ≈ 1 − 6 = −5 и состояние расходило на T≈2000 (поймано
# тестом). С |k|=1 и β∈(0,1] матрица правки — сжатие (собств. числа ∈ [0,1]).
def delta_init(D, rng):
    f = D ** -0.5
    return dict(Wk=rng.standard_normal((D, D)) * f, Wv=rng.standard_normal((D, D)) * f,
                Wq=rng.standard_normal((D, D)) * f, Wo=rng.standard_normal((D, D)) * f,
                Wb=rng.standard_normal((D, 1)) * f, bb=np.zeros((1,)),
                Wa=rng.standard_normal((D, 1)) * f, ba=np.full((1,), 2.0), sc=np.ones(D))


def delta_fwd(X, P, init=None):
    B, T, D = X.shape
    Kn, V, Q = X @ P["Wk"], X @ P["Wv"], X @ P["Wq"]
    # np.maximum(...,1e-12), а не «норма + eps»: сдвиг меняет саму функцию (FD видит
    # расхождение ~1e-6), а клип совпадает с аналитикой всюду, где норма ≠ 0.
    nrm = np.maximum(np.sqrt(np.einsum("btd,btd->bt", Kn, Kn)), 1e-12)
    K = Kn / nrm[:, :, None]
    beta = sig(X @ P["Wb"] + P["bb"])
    alpha = sig(X @ P["Wa"] + P["ba"])
    S = np.zeros((B, D, D), X.dtype) if init is None else np.array(init, dtype=X.dtype)
    Sall = np.empty((B, T + 1, D, D), X.dtype)
    Uall = np.empty((B, T, D), X.dtype)
    Oall = np.empty((B, T, D), X.dtype)
    Sall[:, 0] = S
    for t in range(T):
        k, v, q = K[:, t], V[:, t], Q[:, t]
        b_, a_ = beta[:, t, 0], alpha[:, t, 0]
        u = np.einsum("bkv,bk->bv", S, k)                     # u = Sᵀ k  (v-индекс)
        Uall[:, t] = u
        S = a_[:, None, None] * (S - b_[:, None, None] * k[:, :, None] * u[:, None, :]) \
            + b_[:, None, None] * k[:, :, None] * v[:, None, :]
        Sall[:, t + 1] = S
        Oall[:, t] = np.einsum("bk,bkv->bv", q, S)            # o = qᵀ S
    Y = Oall @ P["Wo"] * P["sc"]
    return Y, dict(K=K, Kn=Kn, nrm=nrm, V=V, Q=Q, beta=beta, alpha=alpha, Sall=Sall,
                   Uall=Uall, Oall=Oall, last=S, X=X)


def delta_bwd(dY, ctx, P):
    K, V, Q, beta, alpha = ctx["K"], ctx["V"], ctx["Q"], ctx["beta"], ctx["alpha"]
    Sall, Uall, Oall, X = ctx["Sall"], ctx["Uall"], ctx["Oall"], ctx["X"]
    nrm = ctx["nrm"]
    B, T, D = K.shape
    sc, Wo = P["sc"], P["Wo"]
    dYs = dY * sc
    dO = dYs @ Wo.T                                   # градиент на o_t
    dWo = np.einsum("btd,bte->de", Oall, dYs)
    dsc = np.einsum("btd,btd->d", Oall @ Wo, dY)
    dK, dV, dQ = np.zeros_like(K), np.zeros_like(V), np.zeros_like(Q)
    dbeta, dalpha = np.zeros_like(beta), np.zeros_like(alpha)
    # Λ_t = dL/dS_t; Sall[:, t] = S_{t−1}
    dS = np.zeros((B, D, D), dY.dtype)
    for t in range(T - 1, -1, -1):
        Sprev, S = Sall[:, t], Sall[:, t + 1]
        k, v, q, u = K[:, t], V[:, t], Q[:, t], Uall[:, t]
        b_, a_ = beta[:, t, 0], alpha[:, t, 0]
        dS += np.einsum("bk,bv->bkv", q, dO[:, t])            # ∂o_v/∂S_kv = q_k
        dQ[:, t] = np.einsum("bv,bkv->bk", dO[:, t], S)
        kL = np.einsum("bk,bkv->bv", k, dS)                   # kᵀΛ   (v-индекс)
        Lu = np.einsum("bkv,bv->bk", dS, u)                    # Λ u   (k-индекс)
        Lv = np.einsum("bkv,bv->bk", dS, v)                    # Λ v
        dalpha[:, t, 0] = np.einsum("bkv,bkv->b", dS, Sprev) \
            - b_ * np.einsum("bk,bk->b", k, Lu)
        dbeta[:, t, 0] = np.einsum("bv,bv->b", kL, v - a_[:, None] * u)
        dK[:, t] = b_[:, None] * Lv - (a_ * b_)[:, None] * Lu \
            - (a_ * b_)[:, None] * np.einsum("bkv,bv->bk", Sprev, kL)
        dV[:, t] = b_[:, None] * kL
        dS = a_[:, None, None] * dS - (a_ * b_)[:, None, None] * np.einsum("bk,bv->bkv", k, kL)
    # backward для l2-нормализации: dK_pre = (dK − (dK·k̂)k̂)/n
    dKn = (dK - np.einsum("btd,btd->bt", dK, K)[:, :, None] * K) / nrm[:, :, None]
    dsb = dbeta * beta * (1.0 - beta)
    dsa = dalpha * alpha * (1.0 - alpha)
    dX = dKn @ P["Wk"].T + dV @ P["Wv"].T + dQ @ P["Wq"].T \
        + dsb @ P["Wb"].T + dsa @ P["Wa"].T
    Xf = X.reshape(-1, D)
    dP = dict(Wk=Xf.T @ dKn.reshape(-1, D), Wv=Xf.T @ dV.reshape(-1, D),
              Wq=Xf.T @ dQ.reshape(-1, D), Wo=dWo, sc=dsc,
              Wb=Xf.T @ dsb.reshape(-1, 1), bb=dsb.sum((0, 1)).reshape(1),
              Wa=Xf.T @ dsa.reshape(-1, 1), ba=dsa.sum((0, 1)).reshape(1))
    return dX, dP


# ══════════════════════ ЧАНК-ФОРМЫ (производственная ветка) ══════════════════════
# Рекуррентный цикл выше — ЭТАЛОН. Чанк-форма распараллеливает T через BLAS/einsum и
# обязана совпадать с эталоном бит-в-бит до плавающей точки (test_delta::chunked_parity).
# ВАЖНО для учёта MAC: чанк-форма платит C× по MAC/токен ради параллелизма
# (C = размер чанка), т.е. обучение на GPU в выигрыше, CPU-инференс — нет (там O(D)
# на токен в рекуррентной форме; ровно как в их §11 потоковый EMA ≡ оконному).

def _mask_from_loga(loga, C):
    """W[t,i] = exp(L_t − L_i) для i ≤ t, иначе 0;  L = cumsum(loga) внутри чанка."""
    L = np.cumsum(loga, axis=1)
    # clip сверху: в верхней треугольной части (i>t) аргумент положителен и в float32
    # даёт overflow→inf до того, как его сотрёт маска; clip убирает предупреждение и
    # риск NaN в BLAS-путях, не меняя ни одного используемого значения.
    Wm = np.exp(np.minimum(L[:, :, None, :] - L[:, None, :, :], 0.0))   # (B,c,c,D)
    tri = np.tril(np.ones((C, C), dtype=bool))
    return np.where(tri[None, :, :, None], Wm, 0.0)


def dddecay_fwd_chunked(X, P, C=64, init=None):
    """То же, что dddecay_fwd, но без цикла по T: h_t = Σ_{i≤t} W[t,i](1−a_i)x_i + W[t,−1] h_prev."""
    B, T, D = X.shape
    a = sig(X @ P["Wa"] + P["ba"])
    loga = np.log(np.maximum(a, 1e-30))
    H = np.empty_like(X)
    h = np.zeros((B, D), X.dtype) if init is None else np.array(init, dtype=X.dtype)
    for j in range(0, T, C):
        c = min(C, T - j)
        aj = a[:, j:j + c]
        Wm = _mask_from_loga(loga[:, j:j + c], c)           # (B,c,c,D)
        u = (1.0 - aj) * X[:, j:j + c]                      # (B,c,D)
        Hin = np.einsum("btid,bid->btd", Wm, u)              # вклад внутри чанка
        hcarry = np.exp(np.cumsum(loga[:, j:j + c], axis=1))  # (B,c,D)
        H[:, j:j + c] = Hin + hcarry * h[:, None, :]
        h = Hin[:, c - 1] + hcarry[:, c - 1] * h
    return H * P["sc"], dict(a=a, H=H, X=X, last=h)


def rot_fwd_chunked(X, P, C=64, init=None):
    """h1/h2 в замкнутой форме: ядро чанка = ρ^m cos/sin(m w), m = t−i ≥ 0."""
    B, T, D = X.shape
    rho, w = np.exp(P["logr"]), P["w"]
    g = 1.0 - rho
    H = np.empty((B, T, 2, D), X.dtype)
    h = np.zeros((B, 2, D), X.dtype) if init is None else np.array(init, dtype=X.dtype)
    for j in range(0, T, C):
        c = min(C, T - j)
        m = np.arange(c)
        kern_r = (rho[None, :] ** m[:, None]) * np.cos(np.outer(m, w) * 1.0)
        kern_i = (rho[None, :] ** m[:, None]) * np.sin(np.outer(m, w) * 1.0)
        u = g * X[:, j:j + c]                                 # (B,c,D) вход только в h1
        # W1[t,i] = ρ^{t−i}cos((t−i)w) при i ≤ t, иначе 0 (иначе отрицательный индекс
        # уходит в хвост массива и добавляет мусор — ловил это паритетом на C≥3)
        msk = m[:, None] - m[None, :]
        k = np.where(msk >= 0, msk, 0)[..., None]                              # (c,c,1)
        lowp = (msk >= 0)[..., None]
        W1 = np.where(lowp, kern_r[k[..., 0]], 0.0)                            # (c,c,D)
        W2 = np.where(lowp, kern_i[k[..., 0]], 0.0)
        Hin1 = np.einsum("tid,bid->btd", W1, u)
        Hin2 = np.einsum("tid,bid->btd", W2, u)
        # carry: A^{t+1} = ρ^{t+1}[cos,sin] от предыдущего состояния чанка
        kp = m + 1
        cr = (rho[None, :] ** kp[:, None]) * np.cos(np.outer(kp, w))
        ci = (rho[None, :] ** kp[:, None]) * np.sin(np.outer(kp, w))
        H[:, j:j + c, 0] = Hin1 + cr[None] * h[:, None, 0] - ci[None] * h[:, None, 1]
        H[:, j:j + c, 1] = Hin2 + ci[None] * h[:, None, 0] + cr[None] * h[:, None, 1]
        h = np.stack([H[:, j + c - 1, 0], H[:, j + c - 1, 1]], axis=1)
    return H[:, :, 0] * P["sc"], dict(H=H, X=X, rho=rho, c=np.cos(w), s=np.sin(w),
                                      g=g, last=h)


CHUNKED = {"dddecay": dddecay_fwd_chunked, "rot": rot_fwd_chunked}


# ══════════════════ CTRL: его же time-invariant EMA, но с проверенным grad ══════════════════
def ema_init(D, rng):
    return dict(th=np.zeros(D), sc=np.ones(D))


def ema_fwd(X, P, init=None):
    """h_t = σ(th) h_{t-1} + (1-σ(th)) x_t  — дословно nano_lc.ema_mix (обучаемый th)."""
    B, T, D = X.shape
    a = sig(P["th"])
    H = np.empty_like(X)
    h = np.zeros((B, D), X.dtype) if init is None else np.array(init, dtype=X.dtype)
    for t in range(T):
        h = a * h + (1.0 - a) * X[:, t]
        H[:, t] = h
    return H * P["sc"], dict(a=a, H=H, X=X, last=h)


def ema_bwd(dY, ctx, P):
    """a = σ(th) — time-invariant, поэтому dth накапливается по всем t и всем строкам."""
    a, H, X = ctx["a"], ctx["H"], ctx["X"]
    B, T, D = H.shape
    sc = P["sc"]
    dH = dY * sc
    dsc = np.einsum("btd,btd->d", dY, H)
    dX = np.zeros_like(H)
    da = np.zeros((B, T, D), dY.dtype)              # a — (D,), da — по всем токеням
    lam = np.zeros((B, D), dY.dtype)
    for t in range(T - 1, -1, -1):
        lam = dH[:, t] + a * lam                  # ∂h_{t+1}/∂h_t = a (одинаков для всех t)
        hprev = H[:, t - 1] if t > 0 else np.zeros((B, D), H.dtype)
        da[:, t] = lam * (hprev - X[:, t])         # ∂h_t/∂a = h_{t-1} - x_t
        dX[:, t] = lam * (1.0 - a)                 # ∂h_t/∂x_t = 1 - a
    dth = (da * a * (1.0 - a)).sum((0, 1))          # цепочка через σ(th)
    return dX, dict(th=dth, sc=dsc)


BLOCKS = {"ema": (ema_init, ema_fwd, ema_bwd),
          "dddecay": (dddecay_init, dddecay_fwd, dddecay_bwd),
          "rot": (rot_init, rot_fwd, rot_bwd),
          "delta": (delta_init, delta_fwd, delta_bwd)}


def ema_fwd_ref(X, th, sc, init=None):
    """Точная семантика `nano_lc.ema_mix` (time-invariant, без управления данными) —
    эталон A/B: любой новый блок обязан не просто «быть лучше», а быть лучше при том же MAC."""
    B, T, D = X.shape
    a = sig(th)
    H = np.empty_like(X)
    h = np.zeros(D, X.dtype) if init is None else np.array(init, dtype=X.dtype)
    for t in range(T):
        h = a * h + (1 - a) * X[:, t]
        H[:, t] = h
    return H * sc


def cost_per_token(name, D, T):
    """MAC/токен и состояние — чтобы A/B равнял стоимость, а не «у кого блок жирнее»."""
    return {"ema": dict(mac=2 * D, state=D),
            "dddecay": dict(mac=D * D + 3 * D, state=D),
            "rot": dict(mac=8 * D, state=2 * D),
            "delta": dict(mac=8 * D * D, state=D * D),
            "attn": dict(mac=4 * D * D + 4 * T * D, state=T * D)}[name]


def params_per_block(name, D):
    return {"ema": 2 * D, "dddecay": D * D + 2 * D, "rot": 3 * D,
            "delta": 4 * D * D + 2 * D + D, "attn": 4 * D * D}[name]
