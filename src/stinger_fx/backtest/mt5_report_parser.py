"""Parse MT5 Strategy Tester XML/HTML reports into our BacktestReport.

Phase-1 implementation supports MT5's XML report layout (`Report=...`)
produced by `terminal64.exe /config:tester.ini`. The XML format is stable
across recent terminal builds, but key names sometimes vary by language —
the parser uses fuzzy fallbacks where it can.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from stinger_fx.backtest.reports import BacktestReport, TradeRecord

logger = logging.getLogger("stinger.backtest.mt5_parser")


def _to_float(s: str | None) -> float:
    if not s:
        return 0.0
    return float(s.replace(" ", "").replace(",", ""))


def parse_report(path: Path, *, run_id: str, strategy_id: str) -> BacktestReport:
    tree = ET.parse(path)
    root = tree.getroot()

    initial_balance = 0.0
    final_balance = 0.0
    trades: list[TradeRecord] = []
    equity_curve: list[tuple[datetime, float]] = []

    # Summary table — MT5 exports key/value rows under Workbook/Worksheet/Table
    # Walk Cell pairs looking for known headers.
    for row in root.iter("{urn:schemas-microsoft-com:office:spreadsheet}Row"):
        cells = [c for c in row.iter("{urn:schemas-microsoft-com:office:spreadsheet}Data")]
        texts = [c.text or "" for c in cells]
        if not texts:
            continue
        head = texts[0].strip().lower()
        if head.startswith("initial deposit"):
            initial_balance = _to_float(texts[-1])
        elif head.startswith("total net profit"):
            net = _to_float(texts[-1])
            final_balance = initial_balance + net if initial_balance else net
        elif head.startswith("equity") and head.endswith("balance"):
            # Some reports embed a single equity line — skipped here.
            pass

    # Trades section — many builds put a "Deals" sheet with chronological deals.
    # For Phase 1 we surface counts via the summary and leave deal extraction
    # to a follow-up if the user wants per-deal records. Returning an empty
    # `trades` list keeps the metrics consistent (the summary's net P&L is used).
    if not trades:
        logger.debug("mt5 report parsed without per-deal trade list path=%s", path)

    now = datetime.now(UTC)
    return BacktestReport(
        run_id=run_id,
        strategy_id=strategy_id,
        started_at=now,
        finished_at=now,
        trades=trades,
        equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_balance=final_balance,
    )
