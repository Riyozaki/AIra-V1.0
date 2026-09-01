"""Проверки one_write: паритет «1 проход == решатель», замкнутый grad (FD), gate-цена."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_rls as R

rng = np.random.default_rng(0)
ok = True

print("1) паритет RLS-проход == ridge-решатель, и цена surprise-gate")
for n in (64, 256, 2048):
    p = R.probe_parity(rng, n=n, Dk=32, Dv=16)
    g = p['gate']
    print(f"   n={n:5d}  |W_rls - W_solve|/|W| = {p['rel']:.2e}   writes {p['nw']}"
          + "".join(f" | tau={t}: {v[0]} записей, ×{v[1]:.2f} ошибка" for t, v in g.items() if t > 0))
    if p['rel'] > 1e-10:
        ok = False; print("   ✗ паритет не сошёлся")

print("\n2) потоковый rls_stream == пакетный rls_write_seq (важно для инференса)")
K = rng.standard_normal((128, 24)); V = K @ rng.standard_normal((24, 8))
W1, _, _ = R.rls_write_seq(K, V)
st = (np.zeros((24, 8)), np.eye(24) / 1e-2)
for t in range(128):
    _, st = R.rls_stream(K[t], V[t], st)
print(f"   |W_stream - W_batch|/|W| = {np.abs(st[0]-W1).max()/np.abs(W1).max():.2e}")
if np.abs(st[0] - W1).max() / np.abs(W1).max() > 1e-10:
    ok = False; print("   ✗ поток != пакет")

print("\n3) замкнутый grad через решатель: FD float64, eps=1e-6 (батч, n>Dk)")
B, n, Dk, Dv = 2, 11, 8, 5
K = rng.standard_normal((B, n, Dk)); V = rng.standard_normal((B, n, Dv))
E = rng.standard_normal((Dk, Dk)) * 0.5
Q = rng.standard_normal((B, 6, Dk)); Wq = rng.standard_normal((Dk, Dk)) * 0.3
lam = 1e-2
def loss(Kx, Vx, Wqx):
    W = R.rls_solve_batch(Kx, Vx, lam=lam)
    O = np.einsum("btd,bdv->btv", Q @ Wqx, W)
    return float(np.einsum("btv,btv->", O, O) * 1e-3)
def analytic(Kx, Vx, Wqx):
    W = R.rls_solve_batch(Kx, Vx, lam=lam)
    O = np.einsum("btd,bdv->btv", Q @ Wqx, W)
    dO = 2e-3 * O
    dW = np.einsum("btd,btv->bdv", Q @ Wqx, dO)
    dQh = np.einsum("btv,bdv->btd", dO, W)
    dWq = Q.reshape(-1, Dk).T @ dQh.reshape(-1, Dk)
    dK, dV = R.rls_grad_batch(Kx, Vx, W, dW, lam=lam)
    return dK, dV, dWq
dK, dV, dWq = analytic(K, V, Wq)
eps = 1e-6
def relerr(num, ana):
    return float(np.max(np.abs(num - ana)) / max(np.max(np.abs(num)), 1e-30))
nK = np.zeros_like(K)
for b in range(B):
    for i in range(n):
        for d in range(Dk):
            K[b, i, d] += eps; fp = loss(K, V, Wq)
            K[b, i, d] -= 2*eps; fm = loss(K, V, Wq)
            K[b, i, d] += eps
            nK[b, i, d] = (fp - fm)/(2*eps)
nV = np.zeros_like(V)
for b in range(B):
    for i in range(n):
        for d in range(Dv):
            V[b, i, d] += eps; fp = loss(K, V, Wq)
            V[b, i, d] -= 2*eps; fm = loss(K, V, Wq)
            V[b, i, d] += eps
            nV[b, i, d] = (fp - fm)/(2*eps)
nQ = np.zeros_like(Wq)
for i in range(Dk):
    for j in range(Dk):
        Wq[i, j] += eps; fp = loss(K, V, Wq)
        Wq[i, j] -= 2*eps; fm = loss(K, V, Wq)
        Wq[i, j] += eps
        nQ[i, j] = (fp - fm)/(2*eps)
r1, r2, r3 = relerr(nK, dK), relerr(nV, dV), relerr(nQ, dWq)
print(f"   max относит. |аналитика-FD|: dK={r1:.2e}  dV={r2:.2e}  dWq(read)={r3:.2e}")
if max(r1, r2, r3) > 1e-6:
    ok = False; print("   ✗ замкнутый grad не сходится — no-BPTT-обучение недоступно")
print("\n" + ("✓ one_write: паритет, поток и замкнутый grad подтверждены" if ok else "✗ есть расхождения"))
sys.exit(0 if ok else 1)
