#!/usr/bin/env python3
"""Download Dukascopy tick history → stinger-fx Parquet store.

Dukascopy publishes free FX / metals / index tick data as per-hour
LZMA-compressed binary files (``.bi5``) at:

    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

Quirks worth knowing:
  * Month is **0-indexed** (Jan = 00, May = 04, Dec = 11) — Dukascopy convention.
  * Each record is 20 bytes big-endian: ``>iIIff``
        time_offset_ms (int32, ms since start of hour)
        ask_raw         (uint32, divide by 10**digits)
        bid_raw         (uint32, divide by 10**digits)
        ask_volume      (float32, millions of base units)
        bid_volume      (float32, millions of base units)
  * Weekends / market holidays return an empty body or 404 — skip silently.
  * Raw LZMA1 stream (no ``.lzma`` container). Decompress with FORMAT_AUTO;
    fall back to FORMAT_RAW + manual LZMA1 filter if that fails.

Lands in ``data/parquet/<SYMBOL>/TICK/<YYYY-MM-DD>.parquet`` via
``ParquetStore.append_ticks``.

Usage:
    python scripts/download_dukascopy_ticks.py \\
        --symbol XAUUSD --start 2026-05-01 --end 2026-05-30
"""
from __future__ import annotations

import asyncio
import lzma
import struct
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import typer

# Make src/ importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stinger_fx.data import ParquetStore  # noqa: E402
from stinger_fx.domain import Tick  # noqa: E402


# Decimal places per instrument — used to scale Dukascopy's integer
# price back to a real float. Extend as you need more symbols.
SYMBOL_DIGITS: dict[str, int] = {
    "XAUUSD": 3,
    "XAGUSD": 3,
    "EURUSD": 5,
    "GBPUSD": 5,
    "USDJPY": 3,
    "AUDUSD": 5,
    "USDCAD": 5,
    "NZDUSD": 5,
}

DUKASCOPY_URL = (
    "https://datafeed.dukascopy.com/datafeed/{symbol}/"
    "{year:04d}/{month0:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
)
RECORD_SIZE = 20
RECORD_FORMAT = ">iIIff"  # big-endian: int32, uint32, uint32, float32, float32

USER_AGENT = "stinger-fx-dukascopy-downloader/1.0"


def _decompress_bi5(data: bytes) -> bytes:
    """Decompress a Dukascopy .bi5 payload to the raw 20-byte records.

    Dukascopy serves raw LZMA1 streams without a container header. Most
    modern files decompress fine via FORMAT_AUTO; older ones occasionally
    need explicit FILTER_LZMA1 with a generous dict_size.
    """
    if not data:
        return b""
    try:
        return lzma.decompress(data, format=lzma.FORMAT_AUTO)
    except lzma.LZMAError:
        filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 23}]
        return lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW, filters=filters,
        ).decompress(data)


def _parse_hour(
    symbol: str, raw: bytes, hour_start: datetime, digits: int,
) -> list[Tick]:
    """Turn a decompressed hour buffer into a list of Tick objects."""
    if not raw or len(raw) % RECORD_SIZE != 0:
        return []
    divisor = float(10**digits)
    ticks: list[Tick] = []
    for offset in range(0, len(raw), RECORD_SIZE):
        ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack_from(
            RECORD_FORMAT, raw, offset,
        )
        # Dukascopy quotes volumes in millions of base units; round to an
        # integer "lots * 1e6" so we keep some signal without floats.
        volume = int(round((ask_vol + bid_vol) * 1_000_000))
        ticks.append(
            Tick(
                symbol=symbol,
                time=hour_start + timedelta(milliseconds=ms),
                bid=bid_raw / divisor,
                ask=ask_raw / divisor,
                last=0.0,
                volume=volume,
                flags=0,
            )
        )
    return ticks


def _hours_between(start: datetime, end: datetime) -> list[datetime]:
    """All UTC hour-boundary timestamps in [start, end). Inclusive of start
    hour, exclusive of end hour."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    hours: list[datetime] = []
    while cur < end:
        hours.append(cur)
        cur += timedelta(hours=1)
    return hours


async def _fetch_hour(
    client: httpx.AsyncClient,
    symbol: str,
    hour_start: datetime,
    digits: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> tuple[datetime, list[Tick]]:
    """Fetch one hour bucket. Returns ``(hour_start, [])`` on permanent
    404 / unrecoverable decode errors. Retries transient 5xx / network
    errors with exponential backoff (1s, 2s, 4s, 8s, 16s)."""
    url = DUKASCOPY_URL.format(
        symbol=symbol,
        year=hour_start.year,
        # Dukascopy month is 0-indexed.
        month0=hour_start.month - 1,
        day=hour_start.day,
        hour=hour_start.hour,
    )
    backoff = 1.0
    last_err: str = ""
    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                resp = await client.get(url, timeout=30.0)
            except httpx.HTTPError as e:
                last_err = f"http error: {e!r}"
                resp = None  # type: ignore[assignment]
        if resp is not None:
            if resp.status_code == 404:
                return hour_start, []  # weekend / no data — permanent
            if resp.status_code == 200:
                try:
                    raw = _decompress_bi5(resp.content)
                except lzma.LZMAError as e:
                    typer.echo(
                        f"  ! decompress fail {hour_start.isoformat()}: {e}",
                        err=True,
                    )
                    return hour_start, []
                return hour_start, _parse_hour(symbol, raw, hour_start, digits)
            last_err = f"HTTP {resp.status_code}"
        # Retryable (5xx, network error). Backoff if attempts remain.
        if attempt < max_retries:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 16.0)
    typer.echo(
        f"  ! gave up {hour_start.isoformat()} after {max_retries} attempts "
        f"({last_err})",
        err=True,
    )
    return hour_start, []


async def _download_range(
    symbol: str,
    start: datetime,
    end: datetime,
    out_root: Path,
    concurrency: int = 8,
) -> int:
    """Download [start, end) hourly, write into ParquetStore per UTC day.
    Returns total tick count written."""
    digits = SYMBOL_DIGITS.get(symbol)
    if digits is None:
        raise typer.BadParameter(
            f"unknown SYMBOL_DIGITS for {symbol!r}; "
            f"add it to SYMBOL_DIGITS in this script (Dukascopy decimal places).",
        )
    store = ParquetStore(out_root)
    hours = _hours_between(start, end)
    typer.echo(
        f"fetching {len(hours)} hour(s) for {symbol} "
        f"[{start.isoformat()} → {end.isoformat()})  "
        f"concurrency={concurrency}",
    )
    semaphore = asyncio.Semaphore(concurrency)
    total_ticks = 0
    # Buffer ticks by UTC date — write each day once. Keeps memory bounded
    # for month-long downloads (≈5–10M ticks for XAU at < 1 GB RAM).
    by_day: dict[str, list[Tick]] = {}
    last_flush_day: str | None = None

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        tasks = [
            asyncio.create_task(
                _fetch_hour(client, symbol, h, digits, semaphore),
                name=f"dkc.{h.isoformat()}",
            )
            for h in hours
        ]
        done = 0
        # Process in completion order so progress logging is responsive,
        # but buffer per-day to avoid 24× write amplification per day.
        for coro in asyncio.as_completed(tasks):
            hour_start, ticks = await coro
            done += 1
            day_key = hour_start.date().isoformat()
            by_day.setdefault(day_key, []).extend(ticks)
            # Flush a day's buffer once all 24 hours have arrived — but
            # we can't tell here, so flush at the end. For progress we
            # just log every 24 hours processed.
            if done % 24 == 0 or done == len(hours):
                typer.echo(
                    f"  fetched {done}/{len(hours)} hours, "
                    f"buffered {sum(len(v) for v in by_day.values()):,} ticks"
                )

    # Sort ticks by time within each day (as_completed scrambles order)
    # and write.
    for day_key in sorted(by_day):
        day_ticks = sorted(by_day[day_key], key=lambda t: t.time)
        if not day_ticks:
            continue
        n = store.append_ticks(symbol, day_ticks)
        total_ticks += n
        typer.echo(
            f"  wrote {n:,} ticks → "
            f"{out_root / symbol / 'TICK' / (day_key + '.parquet')}",
        )
    return total_ticks


def _parse_date(s: str) -> datetime:
    """Accept YYYY-MM-DD (treated as 00:00 UTC) or full ISO datetime."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    symbol: str = typer.Option(..., "--symbol", help="e.g. XAUUSD, EURUSD"),
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD (UTC)"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD (UTC, exclusive)"),
    out: Path = typer.Option(
        Path("data/parquet"), "--out",
        help="Parquet root; defaults to ./data/parquet",
    ),
    concurrency: int = typer.Option(
        8, "--concurrency", min=1, max=32,
        help="Concurrent HTTP fetches (be polite to Dukascopy)",
    ),
    end_inclusive: bool = typer.Option(
        False, "--end-inclusive/--end-exclusive",
        help="If set, end date includes the full day (end+1d 00:00 UTC).",
    ),
) -> None:
    """Download Dukascopy tick history and write into the stinger-fx Parquet store."""
    s = _parse_date(start)
    e = _parse_date(end)
    if end_inclusive:
        e = e + timedelta(days=1)
    if e <= s:
        raise typer.BadParameter("--end must be after --start")
    n = asyncio.run(_download_range(symbol, s, e, out, concurrency))
    typer.echo(f"\ndone: {n:,} ticks written under {out}/{symbol}/TICK/")


if __name__ == "__main__":
    app()
