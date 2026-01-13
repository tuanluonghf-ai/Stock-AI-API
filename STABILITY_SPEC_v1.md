# STABILITY LAYER — SPEC & FREEZE v1 (INCEPTION)

## 0) Mục tiêu & Scope (khóa cứng)

### Mục tiêu
- Giảm “giật cục” hành vi khi input dao động nhỏ.
- Giữ Trade Plan **không biến mất**: chỉ đổi trạng thái có lý do.
- Narrative **giữ mạch tư duy** theo DNA private banker.
- Mọi can thiệp phải **minh bạch** (audit) và **testable** (Golden).

### Scope khóa cứng
✅ Can thiệp **chỉ** vào:
- Decision (stable_action)
- Trade Plan (plan_state)
- Narrative (anchor / regime)
- Diagnostics (audit + explain line)

❌ Không được can thiệp:
- Indicator computations (MA/RSI/MACD/Volume/Scenario/Score…)
- Logic core của DecisionPack/TradePlanPack/NarrativeDraftPack (raw)
- Grouping/portion/portfolio/Future Optimize

---

## 1) Kiến trúc chuẩn (Single Source of Truth)

### Core pipeline
- Core packs (raw) được tạo trước:
  - `DecisionPack` (raw_action)
  - `TradePlanPack` (raw plan + gates)
  - `NarrativeDraftPack` (raw narrative)
- Stability layer chạy **sau raw packs**, xuất:
  - `DecisionStabilityPack`
  - `PlanStabilityPack`
  - `NarrativeAnchorPack`
  - `StabilityDiagnosticsPack`

### Render & Regression policy
- UI **chỉ render stable outputs**:
  - `DecisionStabilityPack.stable_action`
  - `PlanStabilityPack.plan_state`
  - `NarrativeAnchorPack.regime` + `anchor_phrase`
- Golden regression **snapshot stable**, không snapshot raw (trừ khi debug chuyên biệt).

---

## 2) DECISION STABILITY — Hysteresis Governor (Freeze)

### Input
- `prev_action` (từ persistence; nếu disabled → dùng raw_action)
- `raw_action` từ `DecisionPack.action`
- Gates/flags từ `TradePlanPack` + distress flags + risk shock (nếu có)

### Output (DecisionStabilityPack.v1)
- `prev_action`
- `raw_action`
- `stable_action`
- `reason` ∈ {`raw`, `hysteresis_hold`, `hysteresis_wait`, `confirmed_shift`, `fast_deescalate`, `hard_exit`}
- `confidence_delta` (âm khi governor giữ lại)
- `gates`: chứa negatives_count, required, buy_confirmation, flags/gates

### Rule Freeze
1) **EXIT luôn pass-through**  
   - raw EXIT → stable EXIT (`hard_exit`)

2) **BUY → HOLD/WAIT/TRIM** cho phép nhanh (risk-off)
   - prev BUY và raw downgrade → stable = raw (`fast_deescalate`)

3) **HOLD → TRIM** chỉ khi “xấu đủ nghĩa”
   - `negatives_count ≥ NEG_REQUIRED` (default 2) **hoặc** structure FAIL  
   - Nếu chưa đủ → giữ HOLD (`hysteresis_hold`, confidence_delta < 0)

4) **WAIT/HOLD → BUY** chỉ khi setup + confirmation
   - raw BUY nhưng `buy_confirmation=False` → giữ prev (`hysteresis_wait`, confidence_delta < 0)
   - đủ xác nhận → BUY (`confirmed_shift`)

5) **TRIM → HOLD** chỉ khi negatives cleared
   - negatives_count == 0 và structure PASS → HOLD
   - ngược lại giữ TRIM (`hysteresis_hold`)

### ENV knobs (Freeze)
- `INCEPTION_HYST_NEG_REQUIRED` (default 2)
- `INCEPTION_BUY_CONFIRM_STRICT` (default 1)
- `INCEPTION_FAST_DEESCALATE` (default 1)

---

## 3) TRADE PLAN STABILITY — Plan Persistence (Freeze)

### Input
- `prev_plan_state` (persistence; nếu disabled → infer từ raw)
- `raw_plan_present` (heuristic từ TradePlanPack primary name/triggers)
- `structure_breakdown` / `roe_broken` (labels) nếu có
- `gates` (structure/trigger/volume/rr/plan)

### Output (PlanStabilityPack.v1)
- `prev_plan_state`
- `raw_plan_present`
- `plan_state` (stable_plan_state) ∈ {ACTIVE, PAUSED, INVALIDATED}
- `reason`
- `notes` (ngắn, audit)
- `carryover_plan_id`

### Rule Freeze
1) **Plan không bao giờ “mất”**: luôn có `plan_state`.
2) ACTIVE → PAUSED:
   - fail nhẹ / thiếu 1 confirm / volume divergence / trigger WAIT
3) PAUSED → ACTIVE:
   - gates phục hồi (revalidated)
4) INVALIDATED:
   - structure breakdown / ROE broken (đứt kỷ luật lõi)

---

## 4) NARRATIVE STABILITY — Semantic Anchoring (Freeze)

### Input
- `prev_regime` (persistence; nếu disabled → infer)
- `raw narrative` (không dùng để quyết định anchor)
- `dominant regime` (Trend/Range/Defensive) từ labels/packs

### Output (NarrativeAnchorPack.v1)
- `prev_regime`
- `regime`
- `anchor_phrase`
- `anchor_changed`
- `reason`

### Rule Freeze
1) Narrative phải kế thừa `anchor_phrase` khi regime không đổi.
2) Regime change có hysteresis (không đổi do 1 phiên nhiễu).

---

## 5) STABILITY DIAGNOSTICS — Audit & Explainability (Freeze)

### Output (StabilityDiagnosticsPack.v1)
- `decision_changed` (raw != stable)
- `plan_state_changed` (prev != current)
- `narrative_anchor_changed` (prev_regime != regime)
- `raw_vs_stable` {raw_action, stable_action, raw_plan_present, plan_state, regime}
- `notes` (audit)
- `ui_explain_line` (short)
- `anchor_phrase` (for display if needed)

### UI explain line policy (Freeze)
- **Chỉ hiển thị khi:**
  1) `stable_action != raw_action`, hoặc
  2) `plan_state ∈ {PAUSED, INVALIDATED}`
- **Không hiển thị** nếu chỉ đổi regime/anchor (tránh spam).
- Tone mapping:
  - distress MEDIUM/SEVERE → câu mềm, phòng thủ (warning)
  - bình thường → kỷ luật (info)
- Phrase rotation:
  - deterministic hash theo (ticker + tình huống + state/reason)

---

## 6) GOLDEN REGRESSION — Freeze

### Deterministic by default
- Golden tự set `INCEPTION_DISABLE_STABILITY_STATE=1`
- Override nếu cần test stateful:
  - `INCEPTION_GOLDEN_ALLOW_STATE=1`

### Snapshot rules (Freeze)
- Snapshot **stable fields**:
  - `stability.decision.stable_action`
  - `stability.trade_plan.plan_state`
  - `stability.narrative.regime` (+ anchor_phrase nếu muốn)
  - `stability.diagnostics.*` (booleans only)
- Không snapshot:
  - raw action
  - full narrative text
  - timestamps (`generated_at_utc` bị static)

**PASS/FAIL**
- Raw dao động nhưng stable không đổi → PASS
- Stable đổi → FAIL (drift thật)

---

## 7) Definition of Done (Freeze)
- UI stable-first, không lộ raw trừ debug.
- Trade plan không “–” vô cớ, luôn có state.
- Explain line đúng lúc, đúng mức.
- Golden PASS ổn định, không phụ thuộc state/history.
