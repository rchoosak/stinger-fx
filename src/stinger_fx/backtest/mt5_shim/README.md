# Stinger-Fx MT5 Shim EA

`stinger_fx_shim.mq5` is the MetaTrader 5 side of the MT5 Strategy Tester
integration. It forwards every tick to a Python `MT5StrategyTester` over
ZeroMQ and applies the order action returned in the reply.

## Why a shim is needed

MT5 Strategy Tester only runs MQL5 code. To run a Python strategy through the
tester, this thin EA acts as the strategy in MQL5 and delegates every decision
to Python via JSON-over-ZeroMQ REQ/REP. The actual strategy logic still lives
in Python (`stinger_fx/strategies/`), so the same code runs identically in
file backtest, MT5 tester, and live MT5.

## Install (Windows MT5 only)

1. Install ZeroMQ for MT5 — copy `libzmq-mt-4_3_5.dll` (or the build you have)
   into `<MT5>/MQL5/Libraries/`.
2. Create folder `<MT5>/MQL5/Experts/StingerFx/` and drop
   `stinger_fx_shim.mq5` in it.
3. Open in MetaEditor → **Compile**. You should see `stinger_fx_shim.ex5`.
4. In **Tools → Options → Expert Advisors**, enable *Allow DLL imports*.

## Wire-format

The EA sends:
```json
{"type":"tick","symbol":"EURUSD","time":1704067200,"bid":1.0823,"ask":1.0825}
```

Python replies with one of:
```json
{"action":"NONE"}
{"action":"BUY","volume":0.01,"sl":1.0800,"tp":1.0850}
{"action":"SELL","volume":0.01}
{"action":"CLOSE","ticket":0}
```

`ticket: 0` means *close every position on the EA's symbol*.

## Run a backtest

```bash
stinger-fx backtest run --run-id <mt5_tester_run_id>
```

The Python side (`MT5StrategyTester`) generates `tester.ini`, spawns
`terminal64.exe /config:tester.ini /portable`, hosts the REP socket on
`tcp://127.0.0.1:5555`, and waits for the tester to finish.
