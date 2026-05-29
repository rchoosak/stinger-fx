# Plan: XauLiquiditySweepReversal Strategy

## Objective

สร้าง strategy สำหรับ trade XAU/USD แบบเข้าเร็วออกเร็วในตลาด sideway โดยใช้แนวคิด liquidity sweep:

- รอราคาทะลุกรอบบนหรือล่างแบบหลอก
- ยืนยันด้วยการปิดกลับเข้ากรอบ
- เข้าแบบ reversal กลับไปหา midpoint, VWAP, หรืออีกฝั่งของ range
- จำกัดความเสี่ยงด้วย stop หลัง wick sweep และ time-based exit

Strategy นี้ควรเหมาะกับ session ที่ XAU/USD แกว่งในกรอบ เช่น Asia range หรือช่วงก่อนข่าวที่ volatility ยังไม่ขยาย แต่ต้องมี filter เพื่อหลีกเลี่ยงตอน range แตกจริง

## Recommended Timeframes

ใช้ multi-timeframe:

- `M15`: regime filter ว่าตลาดไม่ trend แรง
- `M5`: สร้าง sideway range และหา liquidity levels
- `M1`: trigger เข้า order หลัง sweep และ close กลับเข้ากรอบ

M5-only mode ควรทำเป็น fallback สำหรับ backtest ง่ายขึ้น แต่เป้าหมายหลักคือ `M1 entry + M5 structure + M15 regime`

## Strategy Class

เพิ่มไฟล์:

`src/stinger_fx/strategies/examples/xau_liquidity_sweep_reversal.py`

Class ที่จะเพิ่ม:

```python
class XauLiquiditySweepReversalParams(StrategyParams):
    ...

class XauLiquiditySweepReversal(BaseStrategy):
    name = "xau_liquidity_sweep_reversal"
    Params = XauLiquiditySweepReversalParams
```

Class path สำหรับ config:

```text
stinger_fx.strategies.examples.xau_liquidity_sweep_reversal:XauLiquiditySweepReversal
```

## Parameters

เสนอ params เริ่มต้น:

```python
symbol: str = "XAUUSD"
entry_timeframe: Timeframe = Timeframe.M1
structure_timeframe: Timeframe = Timeframe.M5
regime_timeframe: Timeframe = Timeframe.M15

range_lookback_bars: int = 36
min_range_atr: float = 0.8
max_range_atr: float = 4.0

adx_period: int = 14
max_adx: float = 22.0

atr_period: int = 14
sweep_buffer_atr: float = 0.08
reentry_buffer_atr: float = 0.03
stop_buffer_atr: float = 0.12

tp_mode: Literal["mid", "vwap", "fixed_r"] = "mid"
take_profit_r: float = 1.2
min_rr: float = 0.8

volume: float = 0.01
max_hold_bars: int = 8
cooldown_bars: int = 5
max_trades_per_session: int = 4

use_session_filter: bool = True
session_start_hour_utc: int = 0
session_end_hour_utc: int = 16

avoid_news: bool = False
```

หมายเหตุ: `avoid_news` วางเป็น future hook ก่อน เพราะตอนนี้ยังไม่เห็น calendar/news service ใน framework

## Subscriptions

Strategy ต้อง subscribe ทั้ง 3 timeframe:

```python
return [
    Subscription(symbol=params.symbol, timeframe=params.entry_timeframe),
    Subscription(symbol=params.symbol, timeframe=params.structure_timeframe),
    Subscription(symbol=params.symbol, timeframe=params.regime_timeframe),
]
```

ใน `on_bar()` ให้ทำงานหลักเฉพาะเมื่อ bar เป็น `entry_timeframe` เพื่อไม่ให้ส่ง signal ซ้ำจาก M5/M15

## Market Regime Filter

ใช้ `M15` เพื่อยืนยันว่าเหมาะกับ sideway:

1. ดึง bars จาก `ctx.history_for(symbol, regime_timeframe)`
2. คำนวณ `adx(bars, adx_period)`
3. อนุญาต trade เมื่อ `adx <= max_adx`
4. ถ้า ADX สูงกว่า threshold ให้ไม่สวน เพราะอาจเป็น trend breakout จริง

เพิ่มเติม:

- ใช้ ATR บน M5 เพื่อ reject ตลาดนิ่งเกินไปหรือผันผวนเกินไป
- ถ้า M5 range กว้างกว่า `max_range_atr * atr` ให้ถือว่าไม่ใช่ clean sideway
- ถ้า range แคบกว่า `min_range_atr * atr` ให้ skip เพราะ spread/slippage กิน edge ได้ง่าย

## Structure Range

ใช้ M5 bars เพื่อหา range:

```text
range_high = max(high ของ M5 bars ย้อนหลัง range_lookback_bars)
range_low = min(low ของ M5 bars ย้อนหลัง range_lookback_bars)
range_mid = (range_high + range_low) / 2
```

ควร exclude bar ล่าสุดได้ถ้าต้องการป้องกัน lookahead ใน backtest:

- ใช้ closed M5 bars เท่านั้น
- ถ้า current M1 bar เป็น trigger ให้ range มาจาก M5 bars ที่ปิดไปแล้ว

## Entry Logic

### Short Setup: Sweep Range High

เงื่อนไข:

1. M15 ADX ต่ำกว่า `max_adx`
2. M5 range valid
3. M1 bar high ทะลุ `range_high + sweep_buffer`
4. M1 bar close กลับต่ำกว่า `range_high - reentry_buffer`
5. ไม่มี position เปิดอยู่ใน symbol เดียวกัน
6. cooldown ผ่านแล้ว

Action:

```text
SELL market
SL = sweep_wick_high + stop_buffer
TP = range_mid หรือ fixed R
```

### Long Setup: Sweep Range Low

เงื่อนไข:

1. M15 ADX ต่ำกว่า `max_adx`
2. M5 range valid
3. M1 bar low หลุด `range_low - sweep_buffer`
4. M1 bar close กลับสูงกว่า `range_low + reentry_buffer`
5. ไม่มี position เปิดอยู่ใน symbol เดียวกัน
6. cooldown ผ่านแล้ว

Action:

```text
BUY market
SL = sweep_wick_low - stop_buffer
TP = range_mid หรือ fixed R
```

## Exit Logic

เริ่มด้วย exit แบบ simple:

1. Initial SL/TP ส่งไปพร้อม `ctx.buy(..., sl=..., tp=...)` หรือ `ctx.sell(..., sl=..., tp=...)`
2. ถ้า position ยังไม่ปิดภายใน `max_hold_bars` ของ M1 ให้ `ctx.close(ticket, reason="time_exit")`
3. ถ้าราคา close นอกกรอบ M5 ไปทางตรงข้ามกับ reversal ให้ปิดก่อน
4. ถ้ามีกำไรถึง 0.8R ถึง 1.0R อาจย้าย SL ไป break-even ใน phase ถัดไปด้วย `BreakEvenManager`

Phase แรกไม่ควรซับซ้อนเกินไป: ใช้ fixed SL/TP + time exit ก่อน แล้วค่อยเพิ่ม partial close หรือ trailing หลัง backtest เห็น expectancy

## Risk Rules

Rules ที่ควรบังคับใน strategy:

- เปิดได้ครั้งละ 1 position ต่อ strategy ต่อ symbol
- จำกัด `max_trades_per_session`
- cooldown หลังขาดทุนหรือหลังปิด position
- ไม่เข้า trade ถ้า SL distance น้อยเกินไปหรือมากเกินไปเมื่อเทียบกับ ATR
- reject trade ถ้า calculated R:R ต่ำกว่า `min_rr`

Rules ที่ควรปล่อยให้ `RiskMonitor` หรือ config กลางจัดการ:

- daily max loss
- max open positions
- max risk per symbol
- account-level drawdown guard

## Position State

Strategy ต้องเก็บ state อย่างน้อย:

```python
self._last_trade_bar_index: int | None
self._trades_this_session: int
self._session_key: str | None
self._entry_bar_by_ticket: dict[int, int]
```

ใช้ hooks:

- `on_order_filled`: เก็บ ticket กับ entry bar index
- `on_position_closed`: ลบ ticket state
- `on_partial_closed`: update state ถ้าภายหลังเพิ่ม partial close

## Implementation Steps

1. เพิ่มไฟล์ strategy ใหม่ใต้ `src/stinger_fx/strategies/examples/`
2. เพิ่ม params class พร้อม validation ด้วย `pydantic.Field`
3. Implement `subscriptions()` สำหรับ M1/M5/M15
4. Implement helper:
   - `_get_histories(ctx, params)`
   - `_compute_range(structure_bars, params)`
   - `_sideway_regime(regime_bars, structure_bars, params)`
   - `_build_long_setup(...)`
   - `_build_short_setup(...)`
   - `_rr_ok(entry, sl, tp, params)`
5. Implement `on_bar()` ให้ process เฉพาะ M1 closed bar
6. Implement `on_order_filled()` และ `on_position_closed()` สำหรับ state/cooldown
7. เพิ่ม unit tests
8. เพิ่ม integration/backtest test ด้วย synthetic XAUUSD bars
9. Run:

```bash
uv run pytest
uv run mypy src tests
uv run ruff check .
```

## Tests

เพิ่มไฟล์:

`tests/unit/strategies/test_xau_liquidity_sweep_reversal.py`

Cases:

1. ไม่ส่ง signal ถ้า bars ไม่พอ
2. ไม่ส่ง signal ถ้า ADX สูงกว่า threshold
3. short signal เมื่อ M1 sweep range high แล้ว close กลับเข้ากรอบ
4. long signal เมื่อ M1 sweep range low แล้ว close กลับเข้ากรอบ
5. ไม่เข้าเมื่อ close นอกกรอบจริง
6. ไม่เข้าเมื่อมี position เปิดอยู่แล้ว
7. cooldown ป้องกัน entry ซ้ำ
8. R:R ต่ำกว่า `min_rr` ต้อง skip
9. max trades per session ต้องหยุดส่ง signal
10. SL/TP ถูกคำนวณจาก wick + ATR buffer ถูกต้อง

Integration test:

- สร้าง synthetic M1/M5/M15 bars สำหรับ sideway XAUUSD
- run ผ่าน `FileBacktester`
- assert ว่ามี order ฝั่ง reversal และไม่ trade ตอน trend regime

## Backtest Plan

Dataset ที่ควรใช้:

- XAU/USD M1 อย่างน้อย 6 ถึง 12 เดือน
- spread realistic
- commission/slippage realistic
- แยก session Asia, London, NY

Metrics หลัก:

- expectancy ต่อ trade
- profit factor
- max drawdown
- win rate และ average win/loss
- average holding time
- trades per day
- performance แยกตาม session
- MAE/MFE เพื่อปรับ SL/TP

Optimization ranges:

```text
range_lookback_bars: 24, 36, 48, 72
max_adx: 18, 20, 22, 25
sweep_buffer_atr: 0.05, 0.08, 0.12
stop_buffer_atr: 0.08, 0.12, 0.18
take_profit_r: 0.8, 1.0, 1.2, 1.5
max_hold_bars: 5, 8, 12
```

ต้องทำ walk-forward หลัง optimize เพื่อกัน overfit

## Rollout Plan

1. Implement M5-only prototype ก่อนเพื่อ validate range/sweep logic
2. เพิ่ม M1 entry trigger และ M15 ADX filter
3. Backtest แบบ fixed spread/slippage
4. Backtest แบบ pessimistic spread/slippage
5. ทำ walk-forward
6. Paper trade live feed อย่างน้อย 2 สัปดาห์
7. เปิด live ด้วย lot ต่ำสุด และ cap max trades/day

## Acceptance Criteria

Strategy ถือว่าพร้อม merge เมื่อ:

- unit/integration tests ผ่านครบ
- `pytest`, `mypy`, `ruff` ผ่าน
- ไม่มี lookahead bias ใน range calculation
- มี SL/TP ทุก entry
- ไม่เปิด position ซ้ำใน symbol เดียวกัน
- backtest มี positive expectancy หลังหัก spread/slippage
- performance ไม่มาจาก trade แค่ session เดียวหรือช่วงข่าวเฉพาะ

## Future Enhancements

- เพิ่ม VWAP target แทน midpoint
- เพิ่ม news blackout calendar
- เพิ่ม partial close ที่ midpoint แล้ว runner ไป opposite range
- เพิ่ม break-even manager หลังถึง 0.8R
- เพิ่ม spread filter จาก live tick
- เพิ่ม session-specific params เช่น Asia tighter TP, London stricter ADX
