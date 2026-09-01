"""one_write / lc_rls.py — ставка «обучение одним проходом» (one-write learning).

Ядро. Delta-rule (и, тем более, time-invariant EMA) — это ОДИН шаг спуска по MSE состояния
за токен. RLS (recursive least squares) — ТОЧНЫЙ минимум той же цели за один шаг:

    P = A^{-1},  A = rho^{-1} I + K^T K
    v_hat = W^T k ;  g = P k / (rho^{-1} + k^T P k)
    W <- W + g (v - v_hat)^T ;  P <- rho^{-1} (P - g k^T P)          # ранг-1, O(Dk^2)

Следствия, которые здесь проверяются числами (не обещаниями):

  1) `rls_write_seq` за 1 проход == `rls_solve_batch` (точное ridge-решение) == batch-решатель
     с точностью плавающей точки. Знание, на которое у SGD уходят ЭПОХИ, получается за O(1)
     шагов на пример. Это и есть множитель «эпохи → 1», а не «-0.5% MAC».
  2) surprise-gate: писать только если ||v - v_hat||^2 > tau. Меньше записей — меньше
     энергии и меньше трафика; цена по качеству измеряется в том же прогоне.
  3) no-BPTT: градиент по адресации считается через замкнутую форму одного решателя
     (`rls_grad_batch`), а не развёрткой обратного прохода по T. Проверено конечными
     разностями (test_rls.py).
     Вывод: обучаемой остаётся только адресация (мелкая), память — решается.

Формула backward (для dL/dW = P, A = KᵀK + λI, S = A⁻¹P):
    dV = K S ;   dK = (V − 2 K W) Sᵀ
(Выведена из dW = A⁻¹(dBm − dA W), dBm = dKᵀV + KᵀdV, dA = dKᵀK + KᵀdK; два слагаемых
от dA НЕ схлопываются в «2×» — contracting-индексы разные. Именно это и поймал FD:
вариант с «2» давал ошибку 1.6 по dK при корректных dV/read (1e-9).)
"""
from __future__ import annotations

import numpy as np


# ═════════════════════════ потоковая запись (инференс / стриминг) ═════════════════════════
def rls_write_seq(K, V, rho=1.0, lam=1e-2, tau=0.0):
    """Один проход (K,V) -> (W, P, n_writes). tau>0: писать только при остатке > tau."""
    n, Dk = K.shape
    Dv = V.shape[1]
    P = np.eye(Dk, dtype=K.dtype) / lam
    W = np.zeros((Dk, Dv), dtype=K.dtype)
    nw = 0
    for t in range(n):
        k, v = K[t], V[t]
        Pk = P @ k
        s = (1.0 / rho) + k @ Pk
        if s <= 1e-12:
            continue
        err = v - W.T @ k
        if tau > 0.0 and float(err @ err) <= tau:
            continue
        g = Pk / s
        W = W + np.outer(g, err)
        Pn = (P - np.outer(g, Pk)) / rho
        P = 0.5 * (Pn + Pn.T)
        nw += 1
    return W, P, nw


def rls_stream(k, v, st, rho=1.0, tau=0.0):
    """Один шаг потока: st = (W, P) -> (v_hat, st). Память живёт вне модели."""
    W, P = st
    v_hat = W.T @ k
    if v is None:
        return v_hat, st
    Pk = P @ k
    s = (1.0 / rho) + k @ Pk
    if s > 1e-12:
        err = v - v_hat
        if not (tau > 0.0 and float(err @ err) <= tau):
            g = Pk / s
            W = W + np.outer(g, err)
            Pn = (P - np.outer(g, Pk)) / rho
            P = 0.5 * (Pn + Pn.T)
    return v_hat, (W, P)


# ═════════════════════════════════ батч-решатель и его grad ═════════════════════════════════
def rls_solve_batch(K, V, lam=1e-2):
    """W[b] = argmin ||K W - V||² + lam||W||²  — замкнуто, один solve на последовательность."""
    Dk = K.shape[-1]
    A = np.einsum("bki,bkj->bij", K, K) + lam * np.eye(Dk, dtype=K.dtype)
    Bm = np.einsum("bni,bnj->bij", K, V)
    return np.linalg.solve(A, Bm)


def rls_grad_batch(K, V, W, dW, lam=1e-2):
    """Точные dK, dV через замкнутую форму (без развёртки по T).

    dV = K S ;  dK = (V − K W) Sᵀ − K (S Wᵀ),  S = A⁻¹ dL/dW,  A = KᵀK + lam I.
    """
    Dk = K.shape[-1]
    A = np.einsum("bki,bkj->bij", K, K) + lam * np.eye(Dk, dtype=K.dtype)
    S = np.linalg.solve(A, dW)                                  # (B,Dk,Dv)
    dV = np.einsum("bni,bij->bnj", K, S)
    U = V - np.einsum("bni,bij->bnj", K, W)                     # (B,n,Dv) = V − K W
    dK = np.einsum("bnj,bij->bni", U, S)         - np.einsum("bna,bad->bnd", K, np.einsum("bac,bdc->bad", S, W))
    return dK, dV


# ═════════════════════════════════════ зонды ═══════════════════════════════════════════════
def make_recall(rng, B, M, Dk, Dv, rounds=1, query_every=None):
    """Ключи ортогонализуются, чтобы ёмкость не маскировала метод (M <= Dk)."""
    keys = rng.standard_normal((B, M, Dk))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    vals = rng.standard_normal((B, M, Dv))
    return keys, vals


def probe_parity(rng, n=512, Dk=24, Dv=16, lam=1e-2):
    """1 проход RLS == точное решатель (то же знание без эпох) + цена surprise-gate."""
    K = rng.standard_normal((n, Dk))
    Wtrue = rng.standard_normal((Dk, Dv))
    V = K @ Wtrue + 0.05 * rng.standard_normal((n, Dv))
    W1, _, nw = rls_write_seq(K, V, lam=lam)
    W2 = rls_solve_batch(K[None], V[None], lam=lam)[0]
    rel = float(np.linalg.norm(W1 - W2) / np.linalg.norm(W2))
    out = {}
    for tau in (0.0, 1e-2, 1e-1, 0.5):
        Wg, _, ng = rls_write_seq(K, V, lam=lam, tau=tau)
        e = float(np.linalg.norm(K @ Wg - V) / np.linalg.norm(K @ W2 - V))
        out[tau] = (ng, e)
    return dict(rel=rel, nw=nw, gate=out)
