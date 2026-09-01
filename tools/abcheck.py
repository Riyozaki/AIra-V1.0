#!/usr/bin/env python3
"""abcheck.py — охранник честного сравнения (см. docs/03-EXPERIMENTS.md, трек E1).

Соло-исследователь умирает не от нехватки GPU, а от «улучшил и взлетело», потому что
по дороге поменяли три вещи и сида. Скрипт проверяет configs/lab/arms.json ДО того,
как вы сожжёте десятки кВт·ч:

  1. каждый арм меняет ключи только ОДНОГО механизма (swipe по той же оси разрешён);
  2. комбо-арм обязан объявить interaction_with и быть ровно их объединением;
  3. ни один арм не переопределяет поля бюджета (иначе сравнение нечестное);
  4. >= 3 сида на тире confirm; >=1 на screen;
  5. есть baseline-корень, к которому всё приравнивается;
  6. план по тиерам не превышает energy_cap_kwh, и печатает часы/кВт·ч/$ заранее.

    python3 tools/abcheck.py configs/lab/arms.json --plan
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aira_calc import PRESET  # noqa: E402

GUARD = ("depth", "params_non_embedding", "tokens", "seqlen", "global_batch_tokens",
         "precision", "vocab", "corpus", "corpus_hash", "tokenizer", "lr_schedule",
         "seeds", "energy_cap_kwh", "eval", "hardware", "mfu_assumed")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def resolve(name, arms, cache):
    if name in cache:
        return cache[name]
    chain, cur = [], name
    while cur:
        chain.append(cur)
        cur = arms[cur].get("parent")
    cfg = {}
    for node in reversed(chain):
        cfg.update(flatten(arms[node].get("config", {})))
    cache[name] = (cfg, chain)
    return cache[name]


def diff(x, y):
    keys = set(x) | set(y)
    return {k for k in keys if x.get(k) != y.get(k)}


def run_cost(tokens, params, gname, mfu):
    p = PRESET[gname]
    flops = 6.0 * params * tokens
    hours = flops / (max(1.0, p["fp16"] * 1e12 * mfu)) / 3600
    return hours, hours * p["w"] / 1000.0, hours * p["usd"]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    spec = json.load(open(path, encoding="utf-8"))
    budget, arms = spec["budget"], spec["arms"]
    mech = spec.get("mechanisms", {})
    tiers = spec.get("tiers", {"all": dict(tokens=budget.get("tokens", 0),
                                          seeds=budget.get("seeds", [1]),
                                          energy_cap_kwh=budget.get("energy_cap_kwh", 1e9))})
    errors, warnings, cache = [], [], {}
    resolved = {k: resolve(k, arms, cache)[0] for k in arms}

    if not any(a.get("mechanism") in (None, "") for a in arms.values()):
        errors.append("нет baseline-корня (арм с mechanism=null) — не с чем сравнивать")
    if len(set(budget.get("corpus_hash", "") ) ) == 1:
        warnings.append("corpus_hash не заполнен — корпус нельзя воспроизвести, а значит и результат")

    # (1)(2) механизмы и комбо: отклонение арма считается ОТ КОРНЯ (baseline)
    root = next((k for k, a in arms.items() if not a.get("mechanism")), None)
    for name, a in arms.items():
        m, p = a.get("mechanism"), a.get("parent")
        bad = sorted(k for k in flatten(a.get("config", {})) if k in GUARD)
        if bad:
            errors.append(f"{name}: лезет в бюджет: {bad}")
        if name == root:
            continue
        if not m:
            errors.append(f"{name}: у не-базового арма нет mechanism")
            continue
        allowed = set(mech.get(m, {}).get("keys", []))
        if not allowed:
            errors.append(f"{name}: механизм '{m}' не описан в mechanisms")
            allowed = set()
        inter = a.get("interaction_with", [])
        if root not in (p, None) and not inter:
            errors.append(f"{name}: parent={p}; не-комбо арм обязан висеть на baseline,"
                          " иначе в сравнение попадает чужое изменение")
        dev = diff(resolved[root], resolved[name])
        if inter:
            expect = set()
            for other in inter:
                expect |= diff(resolved[root], resolved[other])
            if dev != expect:
                errors.append(f"{name}: комбо-арм = {sorted(dev)}, а объединение родителей ="
                              f" {sorted(expect)} — сравнение с обоими родителями потеряется")
        elif dev - allowed:
            errors.append(f"{name}: изменены чужие ключи {sorted(dev - allowed)}"
                          f" (механизм '{m}' = {sorted(allowed)})")
        if dev and not (dev <= allowed):
            continue
    # циклы
    for name in arms:
        _, chain = resolve(name, arms, cache)
        if len(set(chain)) != len(chain):
            errors.append(f"{name}: цикл в parent-цепочке {chain}")

    # (4)(6) тиреры и план
    plan_rows, tier_tot = [], {}
    gpu = str(budget.get("hardware", "4090")).split()
    gname = next((k for k in PRESET if any(k in x.lower() for x in gpu)), "4090")
    mfu = budget.get("mfu_assumed", 0.25)
    for tname, t in tiers.items():
        if t.get("arms") == "survivors_only":
            names = list(arms)[: max(1, int(t.get("n_survivors", 3)))]
            note_t = f"(считаем по {len(names)} выжившим из {len(arms)})"
        else:
            names, note_t = list(arms), ""
        if len(t.get("seeds", [])) < 3 and tname != "screen":
            errors.append(f"тир '{tname}': seeds={t.get('seeds')} — нужно >=3")
        params = float(str(t.get("params_non_embedding", budget.get("params_non_embedding", "250M")))
                       .upper().replace("M", "e6").replace("B", "e9"))
        h, kwh, usd = run_cost(t.get("tokens", 1e9), params, gname, mfu)
        runs = len(names) * len(t.get("seeds", [1]))
        tot = (h * runs, kwh * runs, usd * runs)
        tier_tot[tname] = (runs, tot, note_t)
        if t.get("energy_cap_kwh") and tot[1] > t["energy_cap_kwh"]:
            errors.append(f"тир '{tname}': {tot[1]:.0f} кВт·ч > cap {t['energy_cap_kwh']} —"
                          " сокращайте число армов или tokens (одинаково у всех -> вывод не страдает)")
        if tot[0] / 24 > 45:
            warnings.append(f"тир '{tname}': {tot[0]/24:.0f} суток на одной карте — это больше месяца;"
                            " аренда 8×H100 на ночь стоит тех же денег, что ваши кВт·ч дома")

    print(f"# {spec.get('study', path)}")
    w = 1 + max(len(k) for k in arms)
    print(f"\n  {'арм':<{w}} {'механизм':<18} {'Δ-ключи':<44} гипотеза")
    for name, a in arms.items():
        p = a.get("parent")
        d = "—" if not p else ", ".join(sorted(diff(resolved[p], resolved[name])))
        star = " *" if a.get("interaction_with") else ""
        print(f"  {name+star:<{w}} {str(a.get('mechanism')):<18} {d:<44}"
              f" {a.get('hypothesis','')[:58]}")
    print("\n  тиеры:")
    for tname, (runs, (h, kwh, usd), note_t) in tier_tot.items():
        t = tiers[tname]
        hpp, kpp, upp = (h / runs if runs else 0), (kwh / runs if runs else 0), (usd / runs if runs else 0)
        print(f"    {tname:8s} {runs:3d} прогонов × {hpp:6.1f} ч = {h:6.0f} ч"
              f" · всего {kwh:5.0f} кВт·ч (~${usd:.0f}) · cap {t.get('energy_cap_kwh')}"
              f" · tokens {t.get('tokens'):,.0e} · seeds {t.get('seeds')} {note_t}"
              + (f" · {1000*hpp*PRESET[gname]['w']/1e6:.0f} кДж/прогон" if runs else ""))
    if spec.get("note"):
        print(f"\n  заметка плана: {spec['note']}")
    print("  правило принятия: арм живёт, если при равном ЭНЕРГЕТИЧЕСКОМ бюджете ppl ниже"
          " на >=3σ повторов,\n  либо ppl равен при >=1.15x меньших Дж/шаг. Иначе —"
          " negative result в 05-RESULTS.md.")

    if warnings:
        print("\n  ⚠ предупреждения:")
        for x in warnings:
            print("    - " + x)
    if errors:
        print("\n  ✗ блокирующие проблемы:")
        for x in errors:
            print("    - " + x)
        print("\n  Не запускать.")
        return 1
    print("\n  ✓ план честный: один механизм на арм, свипы по одной оси разрешены,"
          " комбо обоснованы, бюджет общий, тиры в энергетическом потолке.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
