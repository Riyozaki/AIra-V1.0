# Эксперименты: как это реально запускается

Здесь намеренно **нет своего трейнера**. Свой трейнер — это -3 недели и +1 источник
необъяснимых расхождений. План: взять чужой tuned-стенд, добавить один механизм,
получить честное сравнение.

## E1 — стенд nanochat/modded-nanogpt

```bash
git clone https://github.com/karpathy/nanochat            # или modded-nanogpt
cd nanochat && uv sync                                     # или pip install -e .
python3 -m nanochat.site ...                               # см. runs/speedrun.sh
```

Как маппить `configs/lab/arms.json` на реальный код (делаем один diff-коммит на арм):

| Поле арма | Куда в nanochat/modded-nanogpt | Что проверять |
|---|---|---|
| `depth` | `--depth` (d14 ≈ 250M non-emb) | что Chinchilla-вывод не ломается при уменьшении |
| `optimizer: muon / adamw_only` | `MuonAdamW` флаг opt в optimizer-builder | +5–10% к скорости сходимости — воспроизводится или нет |
| `attention.layers_ratio_full` | паттерн слоёв (FA3 `window_size` / чередование) | что 0.25 ≠ «меньше внимания»: сравниваем и `arm04` |
| `attention.recurrent: gated_delta` | вставить GDN/KDA-слой (FLA-библиотека: `flash-linear-attention`) | наличие кернеля, numerics (bf16), loss-curve |
| `recursion.blocks: 2` | цикл по общему блоку + роутер глубины (MoR, есть неофициальные реализации) | −25% FLOPs/шаг при равном ppl |
| `ternary_ffn_layers_ratio` | BitLinear (репо microsoft/BitNet, свой BitLinear, не post-hoc) | STE + absmean; следить за `activation` 8-bit |
| `activation: relu2/exp3` | выбор нелинейности в FFN | при равном байт-бюджете, не при равном FLOPs |
| `value_embeddings` | есть в nanochat (`--value-embeddings`) | на 24GB-режиме тоже полезно? |
| `energy_cap_kwh` | лимит мощности + лог RAPL/NVML на шаг | считаем не шаги, а кВт·ч |

Обязательный минимум прогона (иначе не сравнимо):
1. `python3 tools/hwprobe.py --json out/hw-<host>.json` и зафиксировать `-pl`.
2. 3 сида на confirm-тире; на screen — 1 сид и право отбрасывать по ≥2σ.
3. Метрика принятия — **ppl при равных кВт·ч**, а не «ppl при равных шагах»
   (иначе арма с recompute всегда проигрывает несправедливо).
4. Один прогон = один файл: `out/<arm>-<seed>.jsonl` с полной конфигурацией и git hash.

## E2 — дистилляция и «сон» (пост-тренинг)

```
стенд:  llama-factory | unsloth | TRL | verl   (что уже стоит на вашем железе)
учитель: открытый MoE 30-40B (A3-A4B), локально, не API   ← юридически и воспроизводимо
корпус:  5-20K промптов из реальных сессий + синтетика с верификацией
отбор:   difficulty p∈[0.05,0.9] по самой модели · diversity по таксономии · quality тестом
дедуп:   8-gram против eval (скрипт обязателен)
этапы:   SFT 1-2 эры → on-policy distillation (reverse-KL) → RLVR только на проверяемом
квант:   QAT/тернарный — под целевой носитель, а не «потом поквантуем»
откат:   регресс-набор ≥200 задач после каждого вшивания адаптера
```

Метрики: доля решённых домен-задач · Дж/задача (с учётом энергии тренинга!) ·
пик RAM · доля задач, решённых **без** длинного контекста (признак «выучено, а не подсмотрено»).

Бейзлайны, которых вы обязаны коснуться, иначе результат ничего не стоит:
`тот же студент без дистилляции`, `RAG по тому же корпусу`, `учитель напрямую`,
`человеческий скилл-файл` (см. 74.5% против 30.4% на SkillLearnBench — вот ваша планка).

## Порядок дня (пока нет железа)

```bash
python3 tools/hwprobe.py --mb 256 --threads 4 --disk-mb 2048 --json out/hw.json
python3 tools/aira_calc.py route --probe out/hw.json
python3 tools/aira_calc.py budget --watts 20 --tps 40
python3 tools/aira_calc.py fit --preset <ваше> --params 35B --active 3B --bits q4 --ctx 32768 \
        --attn-ratio 0.25 --spec-accept 0.6
python3 tools/abcheck.py configs/lab/arms.json --plan
```
