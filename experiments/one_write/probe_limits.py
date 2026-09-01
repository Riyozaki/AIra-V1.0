"""Пределы ставки: (1) численная устойчивость RLS в float32 на длинном потоке,
(2) учит ли что-то обучение адресации, когда памяти НЕ хватает."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lc_rls as R

def write_loop(K, V, dt, rho=1.0, lam=1e-2, reset_at=None):
    n, Dk = K.shape; Dv = V.shape[1]
    K = K.astype(dt); V = V.astype(dt)
    P = np.eye(Dk, dtype=dt)/lam; W = np.zeros((Dk, Dv), dtype=dt); t0 = time.time()
    for t in range(n):
        k = K[t]; Pk = P @ k; s = dt(1.0)/rho + k @ Pk
        if s <= 1e-12: continue
        g = Pk/s; W = W + np.outer(g, V[t] - W.T @ k)
        Pn = (P - np.outer(g, Pk))/rho; P = dt(0.5)*(Pn + Pn.T)
        if reset_at and (t+1) % reset_at == 0:            # CovarianceResetting-подобный фикс
            P = np.eye(Dk, dtype=dt)/lam
    return W, time.time()-t0

print("1) численная устойчивость: 1 проход RLS против точного решателя, разное n и dtype")
print(f"  {'n':>6s} {'Dk':>4s} {'float64':>11s} {'float32':>11s} {'f32+reset(256)':>15s} {'сек f32':>9s}")
for Dk in (16, 64):
    for n in (1024, 8192, 65536):
        rng = np.random.default_rng(0)
        K = rng.standard_normal((n, Dk)); Wt = rng.standard_normal((Dk, 16))*0.3
        V = K @ Wt + 0.01*rng.standard_normal((n, 16))
        ref = np.linalg.solve(K.T@K + 1e-2*np.eye(Dk), K.T@V)
        den = np.linalg.norm(ref)
        W64, _ = write_loop(K, V, np.float64)
        W32, t32 = write_loop(K, V, np.float32)
        W32r, _ = write_loop(K, V, np.float32, reset_at=256)
        e = lambda A: float(np.linalg.norm(A-ref)/den)
        print(f"  {n:6d} {Dk:4d} {e(W64):11.2e} {e(W32):11.2e} {e(W32r):15.2e} {t32:9.2f}")

print("\n2) float32 при том же n, но с нормализацией потока (k/=||k||) — как в delta-rule")
for n in (8192, 65536):
    rng = np.random.default_rng(1); Dk = 64
    K = rng.standard_normal((n, Dk)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    Wt = rng.standard_normal((Dk, 16))*0.3; V = K @ Wt + 0.01*rng.standard_normal((n, 16))
    # ВНИМАНИЕ: ref обязан использовать тот же lam, что и поток, иначе сравниваем
    # разные задачи (первая версия этой строки содержала другой lam и давала
    # фиктивное расхождение 8e-3; правильный контроль — probe_fair.py).
    ref = np.linalg.solve(K.T@K + 1e-2*np.eye(Dk), K.T@V)
    W32, t32 = write_loop(K, V, np.float32)
    print(f"  n={n:6d}  |W_rls-W_solve|/|W| = {np.linalg.norm(W32-ref)/np.linalg.norm(ref):.2e}   ({t32:.1f}s)")

print("\n3) адресация: учится ли что-то полезное, когда ключей больше, чем память (M>Dk)")
sys.path.insert(0, os.path.join(HERE, "..", "delta_block"))
import probe_learn as PL
tr = np.random.default_rng(1); ev = np.random.default_rng(2)
M, Dk = 32, PL.DK
Xt, Yt, Xq, Yq = PL.make(tr, B=16, M=M); Xe_t, Ye_t, Xe_q, Ye_q = PL.make(ev, B=16, M=M)
I = np.eye(Dk)
print(f"  {'0 эпох (тождественная)':34s} rel={PL.relerr(PL.rls_pred(Xe_t,Ye_t,Xe_q,I,I,I), Ye_q):.4f}")
for lr in (3e-3, 1e-2, 3e-2):
    for st in (50, 500):
        (Ek, Ev, Eq), dt, h = PL.train_rls(Xt, Yt, Xq, Yq, st, lr=lr)
        print(f"  {'адресация, lr=%.0e, %d шаг.' % (lr, st):34s} rel={PL.relerr(PL.rls_pred(Xe_t,Ye_t,Xe_q,Ek,Ev,Eq), Ye_q):.4f}  train-rel={h[-1]:.4f}")
print("  -> если не падает ниже 0.71, вывод: при нехватке ёмкости обучение адресации не чинит,"
      "\n     чинит размер состояния (Dk) — байты надо класть в память, а не в параметры.")
