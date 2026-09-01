"""Гейт: моя обвязка обязана дать его же число (111.21 PPL на val), иначе все дальнейшие
Δ приписывать нельзя."""
import os
import sys

LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore"))
sys.path.insert(0, LC)

import json, sys, time
import numpy as np
import nano_lc as NL

LC = os.environ.get("LEANCORE", os.path.expanduser("~/AIra/leancore"))
tr = np.load(f"{LC}/data/prep/train.npy").astype(np.int64)
va = np.load(f"{LC}/data/prep/val.npy").astype(np.int64)
V = json.load(open(f"{LC}/data/prep/meta.json"))["vocab"]
print(f"train {tr.size:,}  val {va.size:,}  V={V}", flush=True)

m = NL.NanoGPT(V, D=192, L=4, ff=576, kind="ema")
d = np.load(f"{LC}/results/ckpt_L_mussa1500s1.npz")
missing = [k for k in m.p.d if k not in d.files]
print("ключей в чкп:", len(d.files), " нет в чкп:", missing, flush=True)
for k in m.p.d:
    if k in d.files: m.p.d[k][...] = d[k]

def ppl_full(model, ids, ctx=96, stride=None, destroy=False):
    """Скользящие окна по всему val, без сэмплирования (у него vloss = 6 случайных окон)."""
    stride = stride or ctx
    tot = 0.0; n = 0
    for s in range(0, len(ids) - ctx - 1, stride):
        x = ids[s:s+ctx][None]; y = ids[s+1:s+ctx+1][None]
        H, _ = model.forward(x)
        l, _ = NL.softmax_ce(model.logits(H), y, destroy=True)
        tot += float(l) * (ctx - 0); n += ctx - 0
    return tot / n

t0 = time.time(); p = ppl_full(m, va); print(f"val PPL (полный свип, stride=96): {np.exp(p):.2f}   [{time.time()-t0:.0f} с]", flush=True)

# его же метрика: 6 случайных окон, seed 1234
rr = np.random.default_rng(1234); tot = 0.0
for _ in range(6):
    st = rr.integers(0, len(va) - 96 - 1, size=24)
    x = np.stack([va[s:s+96] for s in st]); y = np.stack([va[s+1:s+97] for s in st])
    H, _ = m.forward(x); l, _ = NL.softmax_ce(m.logits(H), y, destroy=True); tot += l
print(f"val PPL (его метрика, 6 окон): {np.exp(tot/6):.2f}   [ожидание 111.21]", flush=True)
