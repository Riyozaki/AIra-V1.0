#!/usr/bin/env python3
"""aira_calc.py — калькулятор «обхода» для стандартного железа.

Всё держится на одном равенстве (decode при batch=1 — задача транспорта байт, а не
арифметики), и на следствии из него (энергия):

    tok/s  ≈  eff · BW / B_tok              B_tok = N_act·b + L·c_kv + a   [байт на токен]
    E_tok  =  P / tok/s  =  ρ_sys · B_tok    ρ_sys = цена одного байта трафика, Дж/байт

ρ_sys — не константа железа «вообще», а метрика вашей конфигурации: batching, idle-налог,
CPU-сэмплер и порядок промпта меняют её в разы сильнее, чем смена модели. Поэтому сначала
E0 (замер), потом всё остальное.

Команды:
  budget   сколько байт/токен мне разрешено при 20 Вт и целевых ток/с
  fit      влезет ли модель, с какой скоростью, за сколько джоулей (+offload/MTP/гибрид/PLE)
  train    цена обучения/дистилляции: FLOPs, часы, $, кВт·ч, пик памяти
  payback  окупается ли «большая студент-модель» по энергии жизненного цикла
  cascade  стоит ли двухступенчатый роутинг своих пунктов качества (учитывает recall гейта)
  distill  SFT vs on-policy distillation vs RLVR — в GPU-часах, $ и кВт·ч
  kv       состояние vs веса: кто именно жрёт полосу
  route    что делать прямо сейчас на моём железе (по out/hw.json)

Только stdlib. Оценки ±30–50%: годятся для порядка решений, не для статей.
"""
from __future__ import annotations

import argparse
import json
import sys

# --------------------------------------------------------------- пресеты ---
# bw = паспортная полоса памяти, GB/s | fp16 = dense TFLOPS | vram = GiB
# w = типичное потребление под decode-нагрузкой (не TDP) | usd = аренда $/ч
PRESET = {
    "3090":      dict(bw=936,  fp16=71,   vram=24,  w=280, usd=0.18),
    "4090":      dict(bw=1008, fp16=165,  vram=24,  w=330, usd=0.35),
    "4090-48":   dict(bw=1008, fp16=165,  vram=48,  w=330, usd=0.50),
    "5090":      dict(bw=1792, fp16=210,  vram=32,  w=420, usd=0.45),
    "a6000":     dict(bw=768,  fp16=77,   vram=48,  w=250, usd=0.40),
    "l4":        dict(bw=300,  fp16=121,  vram=24,  w=60,  usd=0.35),
    "a100":      dict(bw=1935, fp16=312,  vram=80,  w=300, usd=1.20),
    "h100":      dict(bw=3350, fp16=989,  vram=80,  w=500, usd=2.00),
    "h200":      dict(bw=4800, fp16=989,  vram=141, w=520, usd=2.50),
    "b200":      dict(bw=8000, fp16=2250, vram=192, w=750, usd=3.00),
    "m3ultra":   dict(bw=819,  fp16=30,   vram=512, w=180, usd=0.00),
    "strixhalo": dict(bw=256,  fp16=60,   vram=128, w=105, usd=0.00),
    "spark":     dict(bw=273,  fp16=100,  vram=128, w=120, usd=0.00),
    "ddr5-2ch":  dict(bw=80,   fp16=6,    vram=128, w=65,  usd=0.00),
    "ddr5-4ch":  dict(bw=160,  fp16=12,   vram=256, w=110, usd=0.00),
    "ddr5-8ch":  dict(bw=320,  fp16=24,   vram=512, w=180, usd=0.00),
}
# цена байта трафика, Дж/байт — СИСТЕМНАЯ метрика (не физика DRAM)
RHO = [(20e-12, "специализированное MatMul-free ядро/FPGA (оценочно; NSLLM ~0.09 Дж/токен)"),
       (100e-12, "NPU/Mobile-SoC при полной загрузке, идеальный режим"),
       (400e-12, "x86-CPU под нагрузкой, AVX512/AMX, тернарные веса"),
       (800e-12, "дискретный GPU, batch=1, кэши включены"),
       (1600e-12, "дискретный GPU, batch=1, CPU-sampler bound / нет кэшей"),
       (4000e-12, "плохая конфигурация: idle-налог, перекомпиляция, переписанный префикс")]
BIT = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "q8": 1.0,
       "q6": 0.75, "q5": 0.66, "q4": 0.5625, "q3": 0.36, "iq3": 0.44, "q2": 0.30,
       "iq2": 0.27, "bitnet": 1.585 / 8.0, "1.58": 1.585 / 8.0, "ternary": 1.585 / 8.0}


def n(x) -> float:
    """'35B' | '3.5e9' | '512M' | 512  ->  float единиц."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower().replace(",", "").replace(" ", "")
    mult = 1.0
    for suf, m in (("t", 1e12), ("b", 1e9), ("m", 1e6), ("k", 1e3)):
        if s.endswith(suf):
            mult, s = m, s[:-1]
            break
    return float(s) * mult


def bytes_per_param(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    k = str(s).strip().lower()
    if k in BIT:
        return BIT[k]
    if k.endswith("bit"):
        return float(k[:-3]) / 8.0
    return float(k) / 8.0


def kv_bytes_per_token(layers, kv_heads, head_dim, kv_b, attn_ratio=1.0, heads=None):
    """(байт KV на токен у full-attn слоёв, байт O(1)-состояния линейных слоёв)."""
    heads = heads or kv_heads
    la = layers * attn_ratio
    per_tok = 2.0 * la * kv_heads * head_dim * kv_b
    lin = layers * (1.0 - attn_ratio)
    state = lin * heads * head_dim * head_dim * kv_b * 2.0
    return per_tok, state


def fit(params, active, b, ctx, layers, kv_heads, head_dim, kv_b, bw, eff, watts, rho,
        fp16_tf=0.0, attn_ratio=1.0, gather_penalty=True, mtp_acc=0.0, mtp_steps=2,
        offload=0.0, offload_bw=None, ple=0.0, embed_share=0.18, heads=None, prefill_mfu=0.3):
    """Главная функция. Единицы: BW и rho задают либо (bw,eff,watts), либо rho напрямую."""
    total_gb = params * b / 1e9 * (1.0 - ple * embed_share)
    w_act_gb = active * b / 1e9 * (1.0 - ple * embed_share)
    kv_tok, kv_state = kv_bytes_per_token(layers, kv_heads, head_dim, kv_b, attn_ratio, heads)
    ctx_gb = kv_tok * ctx / 1e9
    gather = 0.12 * total_gb if (gather_penalty and params > 10e9) else 0.0
    aux = 0.15 * (w_act_gb + ctx_gb)
    B = w_act_gb + ctx_gb + gather + aux                     # GB чтения на токен
    if rho is not None:
        j_tok = rho * B * 1e9
        tok_s = (watts / j_tok) if (watts and j_tok > 0) else 0.0
    elif offload > 0:
        f = min(0.95, offload)
        obw = offload_bw or max(40.0, bw * 0.12)
        t = (1 - f) * B / (eff * bw) + f * B / (0.55 * obw)
        tok_s = 1.0 / t if t > 0 else 0.0
        j_tok = watts / tok_s if tok_s else float("inf")
    else:
        tok_s = eff * bw / B if B > 0 else 0.0
        j_tok = watts / tok_s if tok_s else float("inf")
    if mtp_acc > 0:
        tok_s *= 1.0 + mtp_acc * mtp_steps * 0.55
        j_tok /= 1.0 + mtp_acc * mtp_steps * 0.55
    prefill = (prefill_mfu * fp16_tf * 1e12 / (2.0 * max(active, params * 0.35))) if fp16_tf else 0.0
    kv_cache_gb = ctx_gb + kv_state / 1e9
    return dict(B_gb=B, weights_gb=total_gb, active_gb=w_act_gb, ctx_gb=ctx_gb,
                state_mb=kv_state / 1e6, gather_gb=gather, aux_gb=aux, kv_cache_gb=kv_cache_gb,
                tok_s=tok_s, j_tok=j_tok, rho=j_tok / (B * 1e9) if B else 0.0,
                prefill=prefill)


def block(d, vram=None, label=""):
    print(f"  B_tok = {d['B_gb']:.2f} GB/токен  (веса {d['active_gb']:.2f}"
          f" + контекст {d['ctx_gb']:.2f} + gather {d['gather_gb']:.2f}"
          f" + служебное {d['aux_gb']:.2f}){label}")
    print(f"  tok/s ≈ {d['tok_s']:.1f}  ·  prefill ≈ {d['prefill']:.0f} ток/с"
          f"  ·  E_tok ≈ {d['j_tok']:.3f} Дж/токен  (ρ_sys = {d['rho']*1e12:.0f} пДж/байт)")
    need = d["weights_gb"] + d["kv_cache_gb"] + 1.5
    if vram:
        print(f"  память: веса {d['weights_gb']:.1f} GB + KV/состояние {d['kv_cache_gb']:.2f} GB"
              f" ≈ {need:.1f} GB -> {'влезает' if need <= vram else 'НЕ влезает'} в {vram:.0f} GB"
              + ("" if need <= vram else "  (решение: offload / квант ниже / контекст меньше)"))


# ------------------------------------------------------------- команды -----


def cmd_budget(a):
    e_max = a.watts / a.tps
    print(f"# {a.watts:.0f} Вт при {a.tps:.0f} ток/с  =>  E_tok ≤ {e_max:.3f} Дж/токен")
    print(f"  = {a.watts*24/1000:.2f} кВт·ч/сутки · {a.watts*24*365/1000:.0f} кВт·ч/год"
          f" · ~{a.watts*24*365*0.4/1000:.0f} кг CO₂/год · {a.watts/e_max*3600*24/1e6:.1f}"
          f" млн ток/сутки непрерывно")
    print(f"\n## Допустимый B_tok при разной цене байта трафика (ρ_sys — метрика КОНФИГУРАЦИИ)")
    for rho, note in RHO:
        B = e_max / rho / 1e9
        print(f"  {rho*1e12:6.0f} пДж/байт -> B_tok ≤ {B:6.2f} GB/токен   {note}")
    print("\n## Перевод в размер (70% байтового бюджета на веса, 30% на состояние/служебное)")
    for rho in (100e-12, 400e-12, 800e-12, 1600e-12):
        B = e_max / rho / 1e9
        line = f"  ρ={rho*1e12:5.0f} пДж/Б:"
        for name in ("bitnet", "q4", "iq2"):
            qb = bytes_per_param(name)
            line += f"  {name}≤{0.7*B/qb:5.2f}B"
        print(line + f"   | {B:5.2f} GB/токен")
    print("\n## Обратная задача: какой ρ_sys вам нужен, чтобы мечта сошлась")
    for tgt in ("3B", "8B"):
        B = n(tgt) * bytes_per_param("bitnet") / 1e9 / 0.7
        print(f"  {tgt} активных в 1.58-bit при {a.tps:.0f} ток/с и {a.watts:.0f} Вт"
              f" требует ρ_sys ≤ {e_max/(B*1e9)*1e12:.0f} пДж/байт"
              f"  (B_tok={B:.2f} GB)")
    print("\n  Честный вывод: 20 Вт — это НЕ «70B дома», а «1–3B активных + дисциплина».")
    print("  Публичные якоря: BitNet 2B4T на CPU −82% энергии (0.2–0.5 Дж/токен достижимо);")
    print("  NSLLM-1.5B на FPGA: 13.85 Вт, 161.8 ток/с = 0.086 Дж/токен (51× против")
    print("  RWKV-4 на A800). Датацентр же берёт не мощью, а батчем: 70B при батче 128")
    print("  даёт ~0.39 Дж/токен. Ваша экономика = ρ_sys × B_tok, и оба множителя управляемы.")


def cmd_fit(a):
    p = PRESET[a.preset]
    bw, watts = (a.bw or p["bw"]), (a.watts if a.watts is not None else p["w"])
    rho = a.rho * 1e-12 if a.rho else None
    params, active = n(a.params), n(a.active or a.params)
    b, kv_b = bytes_per_param(a.bits), (a.kv_bits / 8.0 if a.kv_bits > 1 else a.kv_bits)
    base = dict(params=params, active=active, b=b, ctx=a.ctx, layers=a.layers,
                kv_heads=a.kv_heads, head_dim=a.head_dim, kv_b=kv_b, bw=bw, eff=a.eff,
                watts=watts, rho=rho, fp16_tf=p["fp16"], attn_ratio=a.attn_ratio,
                gather_penalty=not a.no_gather, mtp_acc=a.spec_accept, mtp_steps=a.spec_steps,
                offload=a.offload, offload_bw=a.offload_bw, ple=a.ple, heads=a.heads)
    d = fit(**base)
    print(f"# fit: {a.params} total / {a.active or a.params} active · {a.bits} · ctx {a.ctx}"
          f" · {a.preset} · full-attn-слоёв {a.attn_ratio*100:.0f}%")
    block(d, p["vram"])
    share = d["ctx_gb"] / max(1e-9, d["B_gb"])
    print(f"  доля контекста в B_tok: {share*100:.0f}%"
          + ("   <-- состояние дороже весов: сначала KV-квант/линейные слои/кэш"
             if share > 0.5 else ""))
    print("\n  Чувствительность (одно изменение за раз):")
    variants = [("контекст ×4", dict(ctx=a.ctx * 4)), ("контекст ÷8", dict(ctx=max(1024, a.ctx // 8))),
                ("q8", dict(bits="q8")), ("iq2", dict(bits="iq2")),
                ("fp16 (без кванта)", dict(bits="f16")),
                ("гибрид 3:1", dict(attn_ratio=0.25)), ("гибрид 7:1", dict(attn_ratio=1 / 8)),
                ("MLA-подобно: h_kv=1,d=64", dict(kv_heads=1, head_dim=64))]
    if a.spec_accept > 0:
        variants.append(("MTP выключить", dict(spec=0.0)))
    else:
        variants.append(("MTP acc=0.6×2", dict(spec=0.6)))
    variants.append(("offload 40% на CPU" if a.offload == 0 else "offload выключить",
                     dict(off=0.0 if a.offload else 0.4)))
    for lbl, over in variants:
        o = dict(base)
        if "bits" in over:
            o["b"] = bytes_per_param(over["bits"])
        if "ctx" in over:
            o["ctx"] = over["ctx"]
        if "attn_ratio" in over:
            o["attn_ratio"] = over["attn_ratio"]
        if "spec" in over:
            o["mtp_acc"] = over["spec"]
        if "off" in over:
            o["offload"] = over["off"]
        if "kv_heads" in over:
            o["kv_heads"], o["head_dim"] = over["kv_heads"], over["head_dim"]
        d2 = fit(**o)
        print(f"    {lbl:24s} tok/s={d2['tok_s']:8.1f} ({d2['tok_s']/max(1e-9,d['tok_s']):5.2f}×)"
              f"  Дж/токен={d2['j_tok']:7.3f}  B_tok={d2['B_gb']:6.2f} GB")
    print("\n  Самопроверка стенда: 8B в Q4_K на 4090 при ctx 4k — ожидаем 80–120 ток/с.")
    print("   Расхождение с вашим замером >2× означает сломанный замер (батч≠1, cold cache,")
    print("   power-cap, CPU-sampler), а не «уникальную оптимизацию».")


def cmd_train(a):
    N, D, ctx = n(a.n), n(a.d), a.ctx
    gpu, ngpu = a.gpu.split(":")[0], (int(a.gpu.split(":")[1]) if ":" in a.gpu else 1)
    p = PRESET[gpu]
    L_attn = a.layers * a.attn_ratio
    attn = 12.0 * L_attn * a.heads * a.head_dim * (ctx / 2.0)
    flops = 6.0 * N * D + D * attn
    peak = p["fp16"] * 1e12 * ngpu * a.mfu
    hours = flops / peak / 3600 if peak else float("inf")
    kwh = p["w"] * ngpu * hours / 1000.0
    usd = (a.usd_pph if a.usd_pph is not None else p["usd"]) * ngpu * hours
    opt_state = {"adamw": 12.0, "muon": 8.0, "lora": 0.5}[a.opt]
    per_param = (2.0 + 2.0 + 4.0) + opt_state if a.opt != "lora" else 2.0 + 0.5
    act_tok = (34.0 if not a.recompute else 8.0) * a.layers * a.d_model + 4.0 * a.vocab
    mem = (N * per_param / ngpu + a.batch * ctx * act_tok) / 1e9
    print(f"# train: {a.n} non-emb параметров · {a.d} токенов · ctx {ctx} · opt={a.opt}"
          f"{' · recompute' if a.recompute else ''}")
    print(f"  FLOPs = 6ND + attn = {flops/1e18:.1f} EFLOPs   (attn {D*attn/flops*100:.0f}%,"
          f" растёт линейно с ctx, т.к. каузальное среднее = ctx/2)")
    print(f"  {gpu}×{ngpu} @ MFU {a.mfu:.2f}  ->  {hours:.1f} ч · {kwh:.0f} кВт·ч · ~${usd:.0f}")
    print(f"  токенов/параметр: {D/N:.1f}:1"
          + ("   <- сверх Chinchilla (20:1): для малых моделей это правильно"
             if D / N > 25 else "   <- мало: у 0.1–2B оптимум 30–300:1"))
    print(f"  память на ранг ≈ {mem:.1f} GB  ->  "
          + ("не влезет в 24 GB: ctx/batch вниз, recompute, FSDP, или LoRA"
             if mem > 22 else "влезает в 24 GB" if mem < 18 else "впритык к 24 GB"))
    jt = kwh * 3.6e6 / D
    print(f"  {jt:.3f} Дж на токен предобучения  vs {a.inf_j:.2f} Дж на токен инференса"
          f"  -> инференс дороже в ×{a.inf_j/max(1e-12,jt):.0f}")
    print("  Следствие: сдвигайте бюджет из «параметров» в «данные» (это и есть ваш тезис")
    print("   «информация ничего не весит»): +30% токенов стоит дешевле, чем +30% размеров,")
    print("   и окупается на инференсе. НО: чужое предобучение вы не покупаете вовсе —")
    print("   вы берёте готовые веса и платите только за дистилляцию (см. distill).")
    print("\n  MFU 0.45+ на consumer-GPU — редкость; 0.20–0.30 реалистично. При ctx≥32k")
    print("   доминируют активации и logits (4·V байт/токен!), а не оптимизатор.")


def cmd_payback(a):
    """Выбор размера СТУДЕНТА: доплачиваете ли вы энергией за большую модель."""
    p = PRESET[a.preset]
    small, big, D = n(a.small), n(a.big), n(a.d)
    rho = a.rho * 1e-12
    qb = bytes_per_param(a.bits)
    B_s, B_b = small * qb * 1.15 / 1e9, big * qb * 1.15 / 1e9
    jt_s, jt_b = rho * B_s * 1e9, rho * B_b * 1e9
    dtr = 6.0 * (big - small) * D / (p["fp16"] * 1e12 * a.mfu * a.gpus) * p["w"] * a.gpus / 3.6e6
    hours = dtr * 1000 / max(1.0, p["w"] * a.gpus)
    saved = jt_b - jt_s
    print(f"# payback студента {a.small} -> {a.big} (q {a.bits}, ρ_sys={a.rho:.0f} пДж/байт)")
    print(f"  инференс: {jt_s:.3f} -> {jt_b:.3f} Дж/токен  (дороже на {saved:.3f} Дж/токен)")
    print(f"  доп. энергия на ваш тренинг ({D:.1e} токенов): {dtr:.1f} кВт·ч"
          f" · {hours:.1f} ч на {a.gpus}×{a.preset} · ~${hours*p['usd']*a.gpus:.0f}")
    if saved > 0 and dtr > 0:
        tokens = dtr * 3.6e6 / saved
        print(f"  окупится после {tokens:.2e} сгенерированных токенов")
        if a.tpy:
            print(f"  при вашей нагрузке {a.tpy:.1e} ток/год -> через {tokens/a.tpy*12:.1f} мес."
                  + ("   <- НЕ окупается за срок жизни модели: берите меньшего студента"
                     if tokens / a.tpy > 1.0 else "   <- окупается быстро: большая оправдана"))
    print("\n  Рамка решения: берите меньшую модель, если (а) домен узкий и данные лучше")
    print("   спасают, (б) годовой объём генерации мал, (в) у вас жёсткий ρ_sys (CPU-ветка).")
    print("   Иначе платите параметрами — но тогда и эскалация (см. cascade) обязательна.")


def cmd_cascade(a):
    j1, j2, t1, t2, p, r = a.j1, a.j2, a.tok1, a.tok2, a.p, a.gate_recall
    e_only_big = t2 * j2
    e_s1_only = t1 * j1
    esc = p * (1 - a.q1) * r + (1 - p)
    e_casc = t1 * j1 + esc * (t2 * j2 + 0.3 * t1 * j1)
    q_casc = p * a.q1 + esc * a.q2
    print(f"# каскад S1={a.s1} ({j1} Дж/ток) -> S2={a.s2} ({j2} Дж/ток),"
          f" p={p:.2f}, recall гейта r={r:.2f}")
    print(f"  Э_задача: только S2 {e_only_big:7.0f} Дж  ·  только S1 {e_s1_only:7.0f} Дж"
          f"  ·  каскад {e_casc:7.0f} Дж  ({e_only_big/e_casc:.2f}×)")
    print(f"  Качество: S2 {a.q2:.3f}  ·  S1 {a.q1:.3f}  ·  каскад {q_casc:.3f}"
          f" ({100*(q_casc-a.q2):+.1f} п.п.)")
    print(f"  Доля эскалаций: {esc*100:.1f}%   (обратите внимание: она линейна по 1-r)")
    print("\n  скан по p при разных r (главная ось — надёжность детекции провала, не 'сложность'):")
    for rr in (0.4, 0.7, 0.9, 0.99):
        line = f"    r={rr:.2f}: "
        for pp in (0.5, 0.7, 0.85, 0.95):
            e2 = pp * (1 - a.q1) * rr + (1 - pp)
            ee = t1 * j1 + e2 * (t2 * j2 + 0.3 * t1 * j1)
            qq = pp * a.q1 + e2 * a.q2
            line += f" p={pp:.2f} {e_only_big/ee:4.1f}×/{qq:.3f}  "
        print(line)
    print("\n  Вывод: дешёвый гейт (калибровка + «я не уверен» + валидность tool-JSON) даёт")
    print("   больше, чем +2B параметров. Тренируйте S1 отказываться — это один день работы")
    print("   и ~0 энергии, против месяца на E1.")


def cmd_distill(a):
    Ns, Nt, D = n(a.student), n(a.teacher), n(a.tokens)
    nsa, nta = n(a.student_act or a.student), n(a.teacher_act or a.teacher)
    p = PRESET[a.preset]
    D_rl = a.prompts * a.k * a.olen if a.prompts else D
    modes = {
        "SFT off-policy (учитель писал данные)": (2 * Nt * D + 6 * Ns * D, 1.0),
        "on-policy distillation (reverse-KL)": (2 * nsa * D + 2 * Nt * D + 6 * Ns * D, 1.0),
        f"RLVR на студенте (k={a.k})": (2 * nsa * D_rl * a.k + 6 * Ns * D_rl * a.k, 0.55),
        f"RLVR на большой модели (контраст)": (2 * nta * D_rl * a.k + 6 * Nt * D_rl * a.k, 0.55),
    }
    base = min(f for f, _ in modes.values())
    print(f"# пост-тренинг {a.student} ← учитель {a.teacher} (act {a.teacher_act});"
          f" {a.tokens} токенов обучения; RL-объём {D_rl:.1e} токенов (промпты×k×длина)")
    for name, (f, mf) in modes.items():
        h = f / (p["fp16"] * 1e12 * a.gpus * a.mfu * mf) / 3600
        print(f"  {name:42s} {f/1e18:8.2f} EF {h:8.1f} ч  ~${h*p['usd']*a.gpus:6.0f}"
              f"  {h*p['w']*a.gpus/1000:7.0f} кВт·ч  {f/base:6.2f}×")
    print("\n  Чего отсюда следует (и это ядро стратегии проекта):")
    print("   · SFT на 1e9 отобранных токенов — день на одной 4090-класс карте. Не неделя.")
    print("   · OPD дороже SFT-инференса, но дешевле RL: сравните свои строки; в литературе")
    print("     OPD = 1/10 GPU-часов RL (Qwen3/AIME'24: 74.4 балла за 1/10 стоимости).")
    print("   · RL-объём токенов обычно на 1–2 порядка больше SFT-объёма: именно поэтому RL")
    print("     «дорогой», а не потому что у него хитрая градиентная математика.")
    print("   · Юридически: дистилляция с закрытых API нарушает ToS у ряда провайдеров.")
    print("     Открытые веса учителя — тот же результат, чисто и воспроизводимо.")


def cmd_kv(a):
    per_tok, state = kv_bytes_per_token(a.layers, a.kv_heads, a.head_dim, a.kv_bits / 8.0,
                                        a.attn_ratio, a.heads)
    print(f"# состояние: full-attn слоёв {a.layers*a.attn_ratio:.0f}/{a.layers},"
          f" h_kv={a.kv_heads}, d={a.head_dim}, {a.kv_bits} бит/элемент")
    print(f"  {per_tok:.0f} Б на токен  +  {state/1e6:.1f} MB фиксированного O(1)-состояния")
    for L in sorted({1024, 8192, 32768, 131072, 1048576, a.ctx}):
        print(f"    ctx={L:8d}: KV = {per_tok*L/1e9:6.2f} GB на последовательность")
    print(f"  1M токенов: {per_tok*1e6/1e9:.1f} GB. При 1008 GB/s это {per_tok*1e6/1e9:.1f} с"
          " чтения, если бы кэш перечитывался целиком на каждый токен — вот за что платят")
    print("  dense-модели на длинном контексте, и вот что покупают гибриды/кэш.")
    print("  Множители: GQA (h_kv↓), MLA (латент d_c≈512), KV-q8 (2×), KV-q4 (4×),")
    print("  KDA/GDN-слои (O(1) вместо O(L)), prefix-cache (не перечитывать), PLE (−эмбеддинги).")


def cmd_route(a):
    try:
        d = json.load(open(a.probe, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"нет {a.probe} ({exc}); сначала: python3 tools/hwprobe.py --json {a.probe}")
        return
    bw = d.get("bandwidth") or {}
    agg = max([v for k, v in bw.items() if k.startswith("threads_") and isinstance(v, (int, float))]
              or [bw.get("single_thread_GBs", 60)])
    ram = (d.get("ram_GiB") or {}).get("MemAvailable", 8)
    gpu = ((d.get("gpu") or {}).get("nvidia") or [])
    print(f"# маршрут по фактическому железу: DRAM ≈{agg:.0f} GB/s, доступно {ram:.0f} GiB,"
          f" GPU: {gpu[0]['name'] if gpu else 'нет CUDA/ROCm'}")
    steps = [("E0: сетка измерений {4 кванта} × {4 контекста}, Дж/токен по RAPL/NVML,"
              " лог в out/runs.jsonl (docs/04-BENCH.md B1)",
              "без этого ни одно «улучшение» неопровергаемо")]
    if ram < 16:
        steps.append((f"RAM = {ram:.0f} GiB:MoE-offload и контекст >16k невозможны."
                      " Первый апгрейд — память, не GPU",
                      "+64 GB DDR5 стоит как 2 недели аренды H100 и работает вечно"))
    if gpu:
        v = gpu[0].get("vram_GB", 24)
        steps.append((f"Скан `-ncmoe` 0..{max(1,int(v/1.5))} + KV q8_0 + MTP на 30–40B-A3B Q4",
                      "×1.5–×6 на агентской нагрузке; всё уже есть в llama.cpp"))
        if v < 24:
            steps.append(("Основная ветка — домен-студент 2.5–4B в QAT/Q4",
                          f"{v:.0f} GB VRAM не оставляют выбора"))
    else:
        steps.append(("BitNet b1.58 2B4T через bitnet.cpp как всегда-он-ветка (S0/S1)",
                      "−82% энергии, 2.4–6.2× против fp16 на x86; учить тернарно, а не квантовать потом"))
    steps += [
        ("Промпт в инвариантный префикс (система+скиллы вверху) + --cache-reuse/--cache-ram",
         "5–30× эффективного ускорения: самый дешёвый рычаг из существующих"),
        ("E2: on-policy distillation студента 3–4B с ОТКРЫТОГО учителя; корпус 5–20K,"
         " отбор по s1-принципам (difficulty/diversity/quality)",
         "×6–×15 дешевле RL; и это тот случай, где «данные бесплатны» работает буквально"),
        ("Каскад S1(3B) → S2(35B-A3B), гейт по провалу + тренировка отказа",
         "×2–×5 Дж/задача при −1…2 п.п.; гейт даёт больше, чем +2B параметров"),
        ("Политика записи: скилл-файл vs LoRA-«сон» vs сброс, с контрфакт-оценкой каждого"
         " артефакта (см. docs/02-ARCHITECTURE.md §4)",
         "ниша свободна: авто-скиллы 30% против 74% у написанных человеком (2026)"),
        ("E1: архитектурная ставка (KDA-гибрид × MoR-рекурсия × тернарный FFN) на стенде"
         " 300M–1.6B, 3 сида, равный ЭНЕРГЕТИЧЕСКИЙ бюджет",
         "высокая дисперсия; единственный путь к «своей формуле» — и её надо публиковать обеими руками"),
    ]
    for i, (s, why) in enumerate(steps, 1):
        print(f"  {i}. {s}\n       └ {why}")
    print("\n  Не делать: предобучение >10B токенов с нуля; свой токенизатор; свой"
          " оптимизатор «на глаз»; чистый SNN/KAN-LM как основную линию; ещё один RNN"
          " без attention-гибрида; «ещё один бенчмарк» вместо трёх сидов.")


# ------------------------------------------------------------------- cli ---


def build() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AIra efficiency calculator (stdlib)",
                                 epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget", help="ватты -> Дж/токен -> байты/токен")
    b.add_argument("--watts", type=float, default=20.0)
    b.add_argument("--tps", type=float, default=40.0, help="целевых ток/с")
    b.set_defaults(fn=cmd_budget)

    f = sub.add_parser("fit", help="влезет ли / как быстро / сколько джоулей")
    f.add_argument("--preset", default="4090", choices=sorted(PRESET))
    f.add_argument("--params", default="35B", help="общих параметров")
    f.add_argument("--active", default=None, help="активных (для MoE)")
    f.add_argument("--bits", default="q4", help="q4|iq2|q8|bitnet|0.56|'4.5bit'")
    f.add_argument("--ctx", type=int, default=32768)
    f.add_argument("--layers", type=int, default=48)
    f.add_argument("--heads", type=int, default=None)
    f.add_argument("--kv-heads", type=int, default=4)
    f.add_argument("--head-dim", type=int, default=128)
    f.add_argument("--kv-bits", type=float, default=16.0)
    f.add_argument("--attn-ratio", type=float, default=1.0, help="доля full-attn слоёв (3:1 -> 0.25)")
    f.add_argument("--bw", type=float, default=None)
    f.add_argument("--eff", type=float, default=0.45, help="КПД полосы при batch=1")
    f.add_argument("--watts", type=float, default=None)
    f.add_argument("--rho", type=float, default=None, help="Дж/байт трафика ×1e12 (из замера) —"
                                                            " перебивает bw/eff/watts")
    f.add_argument("--offload", type=float, default=0.0, help="доля байтов с CPU/NVMe")
    f.add_argument("--offload-bw", type=float, default=None)
    f.add_argument("--spec-accept", type=float, default=0.0, help="acceptance rate спекуляции")
    f.add_argument("--spec-steps", type=int, default=2)
    f.add_argument("--ple", type=float, default=0.0, help="per-layer embeddings: доля экономии таблиц")
    f.add_argument("--no-gather", action="store_true")
    f.set_defaults(fn=cmd_fit)

    t = sub.add_parser("train", help="цена обучения")
    t.add_argument("--n", default="1.6B", help="non-embedding параметров")
    t.add_argument("--d", default="100B", help="токенов")
    t.add_argument("--gpu", default="4090:1")
    t.add_argument("--mfu", type=float, default=0.25)
    t.add_argument("--ctx", type=int, default=4096)
    t.add_argument("--layers", type=int, default=24)
    t.add_argument("--attn-ratio", type=float, default=0.25)
    t.add_argument("--heads", type=int, default=16)
    t.add_argument("--head-dim", type=int, default=128)
    t.add_argument("--opt", choices=["adamw", "muon", "lora"], default="muon")
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--d-model", type=int, default=2048)
    t.add_argument("--vocab", type=int, default=65536)
    t.add_argument("--recompute", action="store_true")
    t.add_argument("--usd-pph", type=float, default=None)
    t.add_argument("--inf-j", type=float, default=0.5, help="Дж/токен инференса для контраста")
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("payback", help="окупается ли больший студент")
    p.add_argument("--small", default="3B")
    p.add_argument("--big", default="8B")
    p.add_argument("--bits", default="q4")
    p.add_argument("--d", default="2B", help="токенов вашего тренинга (дистилляция/SFT)")
    p.add_argument("--preset", default="4090")
    p.add_argument("--rho", type=float, default=800.0, help="пДж/байт (из замера!)")
    p.add_argument("--mfu", type=float, default=0.3)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--tpy", type=float, default=2e8, help="своих токенов в год")
    p.set_defaults(fn=cmd_payback)

    c = sub.add_parser("cascade", help="каскад: энергия vs качество vs надёжность гейта")
    c.add_argument("--s1", default="3B"); c.add_argument("--s2", default="35B-A3B")
    c.add_argument("--p", type=float, default=0.75, help="доля задач, принимаемых S1")
    c.add_argument("--gate-recall", type=float, default=0.7, help="r: доля провалов S1, пойманных гейтом")
    c.add_argument("--j1", type=float, default=0.3); c.add_argument("--j2", type=float, default=2.0)
    c.add_argument("--tok1", type=float, default=900); c.add_argument("--tok2", type=float, default=3500)
    c.add_argument("--q1", type=float, default=0.78); c.add_argument("--q2", type=float, default=0.90)
    c.set_defaults(fn=cmd_cascade)

    x = sub.add_parser("distill", help="SFT vs OPD vs RLVR по стоимости")
    x.add_argument("--student", default="3B"); x.add_argument("--teacher", default="35B")
    x.add_argument("--student-act", default=None); x.add_argument("--teacher-act", default="3B")
    x.add_argument("--tokens", default="1B"); x.add_argument("--k", type=int, default=8)
    x.add_argument("--prompts", type=int, default=0, help="число RL-промптов (включает честный объём)")
    x.add_argument("--olen", type=int, default=8000, help="токенов на рулоут")
    x.add_argument("--preset", default="4090"); x.add_argument("--gpus", type=int, default=1)
    x.add_argument("--mfu", type=float, default=0.3)
    x.set_defaults(fn=cmd_distill)

    k = sub.add_parser("kv", help="состояние vs веса")
    k.add_argument("--layers", type=int, default=48); k.add_argument("--kv-heads", type=int, default=8)
    k.add_argument("--heads", type=int, default=None)
    k.add_argument("--head-dim", type=int, default=128); k.add_argument("--kv-bits", type=float, default=16)
    k.add_argument("--attn-ratio", type=float, default=1.0); k.add_argument("--ctx", type=int, default=131072)
    k.set_defaults(fn=cmd_kv)

    r = sub.add_parser("route", help="что делать на этом железе")
    r.add_argument("--probe", default="out/hw.json")
    r.set_defaults(fn=cmd_route)
    return ap


def main(argv=None) -> int:
    a = build().parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
