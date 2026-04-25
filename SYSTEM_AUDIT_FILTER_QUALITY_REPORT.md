# SYSTEM AUDIT + FILTER QUALITY RESTRUCTURE (A/B/C)

## Scope / What was audited

Проведён аудит полного decision pipeline в live-ветке:

1. **Signal generation layer**: `MarketScanner` + `BacktestStrategyAdapter` + `LightModeStrategy`.
2. **Filter layer**: TREND / STRUCTURE / RSI / MACD / VOLUME / ADX и related checks.
3. **Scoring layer**: weighted breakdown (`build_breakdown`) + mode thresholds + adaptive score alignment (`raw_score`, `adjusted_score`, `execution_score`).
4. **Adaptive execution layer**: regime classification, execution confidence, microstructure, risk multiplier.
5. **Final gate logic**: score-alignment gate + decision-engine thresholds.
6. **Execution decision engine**: sizing, portfolio caps, constraints, margin validation.

---

## 3.1 Architectural map (as-is, real flow)

```text
Universe scan
  -> scanner.volume_scanner.get_top_usdt_pairs()
  -> scanner.market_scanner._filter_active_symbols() [MIN_VOLUME_24H + MIN_CHANGE_24H]
  -> candles fetch (OHLCV)
  -> strategy.generate_signal()
       MAIN/SCALPING: services.strategy_adapter.BacktestStrategyAdapter
         -> _build_filter_diagnostics(): TREND/STRUCTURE/RSI/ADX/VOLUME/MACD
         -> _log_score_breakdown(): weighted total_score
         -> hard/soft filter fail gate
         -> adaptive_score_threshold gate
         -> strict entry gate (RR, confidence, structure weak risk checks)
         -> output signal{score, passed_filters, failed_filters, threshold}
       LIGHT: services.light_mode_strategy.LightModeStrategy
         -> indicator checks + build_breakdown + score threshold
         -> output signal_only (no execution)
  -> execution.order_manager.OrderManager._can_execute()
       -> emergency / mode / trading enabled prechecks
       -> adaptive_execution.adapt()
            regime + stress + microstructure + timing + risk_multiplier
       -> scoring_contract.evaluate_score_alignment()
            raw_score -> adjusted_score -> execution_score(final_score)
       -> final score gate (REJECT if final_score < threshold)
       -> execution.decision_engine.evaluate_order()
            second min_score gate + portfolio/risk/constraints/margin validation
       -> execution.compiler.open_order()
```

---

## 1. FULL SYSTEM AUDIT

## 1.1 Logic duplication

### A) Filter duplication across generation and scoring

- `BacktestStrategyAdapter` одновременно:
  - вычисляет `filter_checks` (TREND/STRUCTURE/RSI/ADX/VOLUME/MACD),
  - отдельно считает `score_breakdown` вручную (`trend_score`, `structure_score`, ...),
  - и ещё хранит `passed_filters/failed_filters` в diagnostics.
- Это дублирует идею единого truth-source, т.к. одинаковые логические факторы живут в двух формах: boolean gate + weighted score.

### B) Threshold duplication across layers

- Threshold проверяется минимум в **трёх местах**:
  1. strategy layer (`total_score < adaptive_score_threshold`),
  2. scoring-contract final gate (`final_score < threshold`),
  3. decision engine (`score < min_score` после подмены score).
- При этом источники threshold похожие, но не полностью синхронизированные (mode + overrides + explicit fields).

### C) Volume logic duplication

- Volume фильтруется на уровне universe (`MIN_VOLUME_24H`) и снова в signal-layer (`VOLUME` vs rolling mean).
- Для LIGHT ещё есть `volume_threshold` + `volume_ratio`, которые потом агрегируются обратно в единый `volume` score.

### D) Regime/risk duplication

- Regime влияет на outcome (DEFER/SCALE_DOWN), risk multiplier, mode и далее через adjusted score.
- После этого DecisionEngine снова делает policy reject/scale по портфельным ограничениям.
- В результате риск-контур имеет пересечение функций, где одинаковые «качества» сигнала наказываются многократно.

---

## 1.2 Decision conflicts

### Conflict type 1: filters PASS, execution REJECT

Наблюдается системно возможный сценарий:

- Strategy пропускает сигнал (hard filters pass, score >= adaptive threshold),
- но execution слой отклоняет на одном из downstream gate:
  - `DEFER_EXECUTION` / `REDUCE_RISK_ONLY` из adaptive,
  - `ADAPTIVE_SCORE_BELOW_THRESHOLD` из scoring contract,
  - `SCORE_BELOW_THRESHOLD` / `PORTFOLIO_EXPOSURE_BLOCK` / margin reject в decision engine.

Итого: **signal quality “pass” не означает executable quality “pass”**.

### Conflict type 2: score vs filter contradiction

- Soft-failed filters (например `STRUCTURE` при weak-state) могут присутствовать одновременно с проходом по score threshold.
- В strict payload возможна ситуация `failed_filters` содержит soft-fail, но trade всё равно проходит.
- В scoring-contract дальше score может быть повышен/понижен независимо от semantic failed_filters.

### Conflict type 3: raw_score vs final execution_score inversion

- `evaluate_score_alignment` может:
  - повысить weak raw_score в good regime/liquidity,
  - или «сломать» хороший raw_score в CHOPPY/high-noise через regime/micro/risk adjustments.
- Это создаёт неочевидный поворот решений для одинаковых filter sets.

---

## 1.3 Empty / inefficient filters

### Clear inefficiency

1. **LIGHT_VOLUME_THRESHOLD = 0.0** + `LIGHT_VOLUME_FILTER_ENABLED = True`:
   - `volume_threshold` почти всегда PASS (если объём не нулевой),
   - фильтр фактически шумовой/декоративный.

2. **Optional overlap RSI/MACD/ADX/DI/momentum-like checks**:
   - RSI, MACD histogram sign, DI alignment и trend-direction проверяют схожий directional momentum context.
   - В weak structure они часто дают коррелированные (не независимые) сигналы.

3. **Relaxed branch scoring includes structure as optional in fallback_mode**:
   - при `required_filters = ['trend']` структура может fail, но signal still pass при score.
   - это снижает ценность STRUCTURE как guardrail.

4. **Volume double representation in LIGHT**:
   - `volume_threshold` + `volume_ratio` затем склеиваются в `volume` для scoring.
   - Возможна потеря интерпретируемости: какой именно volume-check дал вклад/провал.

---

## 1.4 Risk system overriding filters

1. **Adaptive risk multiplier rewrites economic impact**:
   - Даже при хорошем filter pass итоговый размер позиции может быть резко сжат (CHOPPY / stress / micro noise), что фактически переопределяет “quality pass”.

2. **Adaptive outcomes can hard-stop passed signals**:
   - `DEFER_EXECUTION` и `REDUCE_RISK_ONLY` блокируют сделку до decision engine.

3. **Regime enters both score and sizing paths**:
   - Regime влияет и на `adjusted_score`, и на `risk_multiplier`, и на execution mode.
   - Это тройной канал влияния одной и той же информации.

---

## 1.5 Score system inconsistencies

1. **Multiple score semantics**:
   - `total_score` (strategy),
   - `raw_score` (scoring contract, обычно из signal score),
   - `adjusted_score` / `final_score` (execution quality adjusted).
- Нет единого naming contract между генерацией и исполнением, хотя в логах выглядит как единая шкала.

2. **Threshold reuse with mutable score**:
   - После score alignment в `OrderManager` signal score заменяется на final_score, затем DecisionEngine снова сравнивает с threshold.
   - Формально это повторная проверка того же условия над уже модифицированным score.

3. **Structure-aware adaptive threshold in strategy vs mode threshold elsewhere**:
   - strategy может адаптировать threshold по `structure_state`, execution слои обычно используют mode-based threshold.
   - Потенциальная рассинхронизация требований между «signal pass» и «execution pass».

---

## 2. FILTER QUALITY CLASSIFICATION (A/B/C)

| Filter | Tier | Reason | Overlaps | Impact |
|---|---|---|---|---|
| TREND (EMA50/EMA200 direction + tolerance) | 🟢 A | Базовый рыночный контекст, снижает контртренд-входы | ADX, MACD, DI | High |
| STRUCTURE (strong/weak/invalid) | 🟢 A | Ключевой quality gate для BOS/continuation валидности | TREND, breakout/momentum checks | High |
| ADX | 🟡 B | Измеряет силу тренда, полезен как подтверждение | TREND, DI, momentum | Medium |
| VOLUME ratio vs MA | 🟡 B | Улучшает вероятность follow-through и снижает thin-market noise | scanner MIN_VOLUME_24H, micro liquidity | Medium |
| MACD histogram/direction | 🟡 B | Подтверждает импульс, особенно после structure/trend pass | RSI, DI, trend slope | Medium |
| RSI bands (40/60 etc.) | 🟡 B | Ограничивает экстремумы/перекупленность, но не standalone predictor | MACD, momentum consistency | Medium-Low |
| DI alignment (+DI/-DI) | 🔴 C | Высоко коррелирует с ADX+trend direction; в текущем design избыточен | ADX, TREND, MACD | Low |
| LIGHT volume absolute threshold (0.0) | 🔴 C | Практически always-pass при ненулевом объёме | volume ratio, universe volume filter | Very Low |
| Probability gate (LIGHT_SIGNAL_PROBABILITY) | 🔴 C | Не quality filter, а stochastic throttling | N/A | Noise / control-only |
| Soft-fail STRUCTURE in weak mode | 🔴 C (как реализация) | Семантика ослаблена: фильтр может fail и не блокировать | score threshold logic | Ambiguous |

---

## 3.3 Problem list

## Critical architecture issues

1. **No single source of truth for quality**: quality проверяется и как booleans, и как weighted score, и как adaptive execution score.
2. **Three-stage gate chain** (strategy -> score contract -> decision engine) создаёт сложные reject-paths и difficult explainability.
3. **Regime/micro/risk triple influence** приводит к пере-наказанию/пере-бонусу одних и тех же market conditions.
4. **Filter-to-score mapping partially implicit** (часть filter names normalizes/grouped), ухудшая прозрачность analytics.

## Duplication points

- TREND/STRUCTURE участвуют и в hard gate, и в score.
- Volume проверяется на universe layer + signal layer (+ micro liquidity later).
- Threshold checks дублируются на strategy, final score gate и decision engine.

## Logic conflicts

- PASS at filter layer != PASS at executable layer (часто deferred/rejected downstream).
- Soft failed filters могут coexist с accepted signals.
- Raw vs adjusted score может поменять исход без изменения passed_filters.

---

## 3.4 Recommendations (analysis only, no refactor yet)

## Что убрать

1. Убрать/деактивировать **LIGHT absolute volume threshold** при значении `0.0` (или не считать его как quality filter).
2. Убрать из quality-решения фильтры, которые служат только operational throttling (например probability gate) — держать отдельно как control policy.
3. Сократить redundant momentum checks (DI как минимум в C-tier-кандидаты на removal).

## Что объединить

1. Объединить все quality filters в единый **Filter Quality Contract**:
   - standardized filter registry,
   - единая нормализация имен,
   - единый `passed/failed/weight/tier` payload.
2. Свести threshold policy в один master-source, чтобы strategy/score-contract/decision-engine читали одинаковое правило.
3. Объединить volume quality в один semantic filter (не 2-3 разных представления).

## Что перенести в scoring layer

1. Soft-support filters (RSI, MACD, часть volume) — оставить в score-only path без hard blocking.
2. Структурировать tiers:
   - A-tier = block-capable,
   - B-tier = score modifiers,
   - C-tier = analytics-only (не участвуют в gate).
3. Risk/regime/micro adjustments держать в execution-quality score, но не дублировать их эффект в policy gates.

---

## Target state for next refactor step

- **Single source of truth for signal quality**: один quality object на весь pipeline.
- Явное разделение:
  - **Signal validity** (A-tier hard filters),
  - **Signal quality score** (A+B weighted),
  - **Execution feasibility** (risk/margin/exchange constraints).
- Устранение “illusion of quality filters”: каждый фильтр либо реально влияет на решение, либо уходит в telemetry.