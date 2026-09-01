"""Зонд ставки «обучение одним проходом» (one-write learning).

Задача: in-context associative recall. Отображение key→value задаётся ВНУТРИ последовательности
и меняется от последовательности к последовательности, поэтому его нельзя зашить в веса —
его можно только записать в состояние. Ровно то, что делает RLS одним шагом, а градиентному
миксеру нужны эпохи.

  study-раунды: x_t = [key_i | val_i]   (val виден входе, никаких утечек цели)
  query-шаги:   x_t = [key_i | 0]       цель y_t = val_i
Метрика: ||pred-y|| / ||y|| по query-позициям. Сравниваем число ПРОХОДОВ ПО ДАННЫМ.

  python3 probe_recall.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "delta_block"))
import lc_delta as DL  # noqa: E402
import lc_rls as R  # noqa: E402

DK, DV, M, ROUNDS, B = 16, 16, 24, 3, 8
D = DK + DV
rng = np.random.default_rng(0)


def make_data(rng, B=B, M=M, rounds=ROUNDS, shift_at=None, keys=None, vals=None):
    """x: (B,T,D); Y: (B,T,DV); qmask: 1 на query-шагах; vp: 1 на study-шагах."""
    if keys is None:
        keys = rng.standard_normal((B, M, DK))
        keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    if vals is None:
        vals = rng.standard_normal((B, M, DV))
    if shift_at is not None:                       # концепт-сдвиг: новые значения
        vals = vals.copy()
        vals[:, shift_at:] *= -1.0
    T = rounds * M * 2
    x = np.zeros((B, T, D)); y = np.zeros((B, T, DV))
    qm = np.zeros((B, T)); vp = np.zeros((B, T))
    for b in range(B):
        order = rng.permutation(M)
        t = 0
        for r in range(rounds):
            for j in range(M):
                i = order[j]
                x[b, t, :DK] = keys[b, i]; x[b, t, DK:] = vals[b, i]
                y[b, t] = vals[b, i]; vp[b, t] = 1.0
                t += 1
            for j in range(M):
                i = order[j]
                x[b, t, :DK] = keys[b, i]; x[b, t, DK:] = 0.0
                y[b, t] = vals[b, i]; qm[b, t] = 1.0
                t += 1
    return x, y, qm[:, :, None], vp[:, :, None]


X, Y, QM, VP = make_data(rng)
T = X.shape[1]
DEN = float(np.linalg.norm(Y * QM))


def rel_err(P):
    return float(np.linalg.norm((P - Y) * QM) / DEN)


# ─────────────────────────── arm 1: RLS, ноль шагов обучения ───────────────────────────
def rls_arm(X, VP, rho=1.0, lam=1e-2, tau=0.0):
    Bn, Tn, _ = X.shape
    out = np.zeros((Bn, Tn, DV))
    nw = 0
    for b in range(Bn):
        W = np.zeros((DK, DV)); P = np.eye(DK) / lam
        for t in range(Tn):
            k = X[b, t, :DK]
            out[b, t] = W.T @ k
            if VP[b, t, 0] < 0.5:
                continue
            v = X[b, t, DK:]
            Pk = P @ k
            s = 1.0 / rho + k @ Pk
            if s <= 1e-12:
                continue
            g = Pk / s
            err = v - W.T @ k
            if tau > 0.0 and float(err @ err) <= tau:
                continue
            W = W + np.outer(g, err)
            P = 0.5 * ((P - np.outer(g, Pk)) / rho + (P - np.outer(g, Pk)).T / rho)
            nw += 1
    return out, nw


# ─────────────────────────── arms 2: градиентные миксеры ───────────────────────────
def sgd_arm(name, steps, seed=0, lr=3e-3, data=None):
    Xd, Yd, QMd, _ = data if data is not None else (X, Y, QM, VP)
    r = np.random.default_rng(seed)
    init, fwd, bwd = DL.BLOCKS[name]
    Dd = Xd.shape[-1]
    base = init(Dd, r)
    P = {k: v.astype(np.float64) * (1.0 if k.endswith(("sc", "ba", "bb", "th")) else 0.3)
         for k, v in base.items()}
    mom = {k: np.zeros_like(v) for k, v in P.items()}
    var = {k: np.zeros_like(v) for k, v in P.items()}
    t0 = time.time()
    for s in range(max(1, steps)):
        Yh, ctx = fwd(Xd, P)
        dY = np.zeros_like(Yh)
        dY[..., :DV] = (Yh[..., :DV] - Yd) * QMd     # градиент только по DV выходам, остальное 0
        dX, dP = bwd(dY, ctx, P)
        for k in P:
            mom[k] = 0.9 * mom[k] + 0.1 * dP[k]
            var[k] = 0.999 * var[k] + 0.001 * dP[k] ** 2
            P[k] -= lr * mom[k] / (np.sqrt(var[k]) + 1e-8)
    Yh, _ = fwd(Xd, P)
    return Yh[..., :DV], time.time() - t0


print(f"задача: M={M} пар, rounds={ROUNDS}, T={T}, D={D}, B={B}   (float64, 1 последовательность = 1 проход)")
print(f"  {'arm':38s} {'шагов/эпох':>10s} {'сек':>7s} {'rel-MSE':>9s} {'writes':>7s}")
rows = []


def add(nm, st, dt, o, nw=None):
    rows.append((nm, st, dt, rel_err(o), nw))
    print(f"  {nm:38s} {st:10d} {dt:7.2f} {rows[-1][3]:9.4f} {str(nw) if nw else '-':>7s}")


t0 = time.time(); o, nw = rls_arm(X, VP)
add("RLS — 0 эпох, 1 проход, без обучения", 0, time.time() - t0, o, nw)
t0 = time.time(); o, nw = rls_arm(X, VP, tau=1e-3)
add("RLS + surprise-gate (tau=1e-3)", 0, time.time() - t0, o, nw)
t0 = time.time(); o, nw = rls_arm(X, VP, rho=0.999)
add("RLS rho=0.999 (закатывание старого)", 0, time.time() - t0, o, nw)
for nm in ("ema", "dddecay", "delta"):
    for st in (1, 10, 100, 1000):
        o, dt = sgd_arm(nm, st)
        add(f"{nm} — SGD {st} шаг(ов) по 1 проходу", st, dt, o)


# ───────────────── концепт-сдвиг: кто быстрее перестроится без переобучения ─────────
print("\nconcept shift (значения меняются с середины раундов) — та же схема, 0 эпох против SGD:")
Xs, Ys, QMs, VPs = make_data(rng, shift_at=M // 2)
DENs = float(np.linalg.norm(Ys * QMs))
o, nw = rls_arm(Xs, VPs, rho=0.98)
print(f"  RLS rho=0.98: rel-MSE = {np.linalg.norm((o - Ys) * QMs) / DENs:.4f}  writes={nw}")
o, nw = rls_arm(Xs, VPs, rho=1.0)
print(f"  RLS rho=1.00: rel-MSE = {np.linalg.norm((o - Ys) * QMs) / DENs:.4f}  writes={nw}  (без забывания — старое мешает)")
for st in (1, 100):
    d = (Xs, Ys, QMs, VPs)
    o, dt = sgd_arm("delta", st, data=d)
    print(f"  delta SGD {st:4d} шагов на том же потоке: rel-MSE = {np.linalg.norm((o[..., :DV] - Ys) * QMs) / DENs:.4f}")

# ───────────────── стоимость: то, за что платим вместо эпох ─────────────────
print("\nцена на токен (D=%d):" % D)
for nm in ("ema", "dddecay", "delta", "rot"):
    c = DL.cost_per_token(nm, D, T)
    print(f"  {nm:8s} {c['mac']:7d} MAC  состояние {c['state']:6d}")
print(f"  {'rls':8s} {6 * DK * DK + 4 * DK * DV:7d} MAC  состояние {DK * DK + DK * DV:6d}"
      "   (W и P; Dk=Dv=16)")
p = R.probe_parity(rng, n=M * ROUNDS * B)
print(f"\nпаритет «1 проход RLS == точное ridge-решение»: {p['rel']:.1e} (float64)")
