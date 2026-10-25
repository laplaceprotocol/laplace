"""Data beyond hourly candles ("senses"), from public, unauthenticated APIs, cached on disk per day.

- **Implied volatility:** Deribit's DVOL index (30-day implied volatility, annualized percent) for BTC and ETH, hourly.
  A row is stamped with the start of its hour; its close is the index at the end of the hour.
- **Intraday realized variance:** the sum of squared 5-minute log returns within each hour, from Coinbase's
  5-minute candles. A 5-minute interval without trades repeats the previous close, so it adds nothing.
- **Perpetual funding:** Binance USDT-margined perpetuals' funding rate, settled every eight hours; a rate is known
  from its settlement time.
- **Open interest:** Bybit USDT perpetuals' open interest in coins, as hourly snapshots stamped with their time
  (Binance keeps only 30 days of history).

Every function returns values keyed by Unix seconds and only includes periods that have fully closed, so nothing
here can carry information from after the moment it describes.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

HOUR = 3600
FIVE_MINUTES = 300
DAY = 86_400
DVOL_CURRENCIES = ("BTC", "ETH")
DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
COINBASE = "https://api.exchange.coinbase.com/products/{product}/candles"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BYBIT_OPEN_INTEREST = "https://api.bybit.com/v5/market/open-interest"
BYBIT_PAGE = 200  # the API's largest page
REQUEST_INTERVAL_S = 0.15


def _days(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    while day < end:
        out.append((day, day + timedelta(days=1)))
        day += timedelta(days=1)
    return out


def _months(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while month < end:
        following = (month + timedelta(days=32)).replace(day=1)
        out.append((month, following))
        month = following
    return out


def _cached(cache: Path, period_end: datetime, fetch: Any) -> Any:
    """A period's rows from the cache, or fetched; only periods that ended over an hour ago are cached."""
    if cache.exists():
        return json.loads(cache.read_text())
    rows = fetch()
    if period_end <= datetime.now(UTC) - timedelta(hours=1):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    return rows


def _get(client: httpx.Client, url: str, params: dict[str, Any]) -> Any:
    for attempt in range(6):
        time.sleep(REQUEST_INTERVAL_S)
        response = client.get(url, params=params)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"{url}: rate-limited on every attempt")


def dvol_hourly(
    currency: str, start: datetime, end: datetime, *, client: httpx.Client, cache_dir: Path
) -> dict[int, float]:
    """``{hour start: DVOL at the end of that hour}`` for the hours in ``[start, end)`` that have closed."""
    out: dict[int, float] = {}
    for first, following in _months(start, end):
        lo, hi = int(first.timestamp()), int(following.timestamp())

        def fetch(lo: int = lo, hi: int = hi) -> list[list[float]]:
            params = {
                "currency": currency,
                "start_timestamp": lo * 1000,
                "end_timestamp": (hi - HOUR) * 1000,
                "resolution": str(HOUR),
            }
            rows: list[list[float]] = _get(client, DERIBIT, params)["result"]["data"]  # at most 744 rows a month
            return [r for r in rows if lo <= r[0] // 1000 < hi]

        for stamp, *_, close in _cached(cache_dir / f"dvol-{currency}-{first:%Y%m}.json", following, fetch):
            hour = int(stamp) // 1000
            if hour + HOUR <= time.time():
                out[hour] = float(close)
    lo, hi = int(start.timestamp()), int(end.timestamp())
    return {h: v for h, v in out.items() if lo <= h < hi}


def realized_hourly(
    product: str, start: datetime, end: datetime, *, client: httpx.Client, cache_dir: Path
) -> dict[int, float]:
    """``{hour start: sum of squared 5-minute log returns within the hour}`` for closed hours in ``[start, end)``.

    The hour's first return runs from the last close before the hour, so twelve returns cover each hour exactly.
    """
    closes: dict[int, float] = {}
    for day, following in _days(start - timedelta(minutes=5), end):
        lo = int(day.timestamp())

        def fetch(lo: int = lo) -> list[list[float]]:
            params = {
                "granularity": FIVE_MINUTES,
                "start": datetime.fromtimestamp(lo, UTC).isoformat(),
                "end": datetime.fromtimestamp(lo + DAY - FIVE_MINUTES, UTC).isoformat(),
            }  # 288 candles, under the API's 300 per request
            rows: list[list[float]] = _get(client, COINBASE.format(product=product), params)
            return [r for r in rows if lo <= r[0] < lo + DAY]

        for row in _cached(cache_dir / f"rv-{product}-{day:%Y%m%d}.json", following, fetch):
            closes[int(row[0])] = float(row[4])  # [time, low, high, open, close, volume]
    lo, hi = int(start.timestamp()), int(end.timestamp())
    closed = [h for h in range(lo - lo % HOUR, hi, HOUR) if h + HOUR <= time.time()]
    return hourly_realized_variance(closes, closed)


def hourly_realized_variance(closes: dict[int, float], hours: list[int]) -> dict[int, float]:
    """For each hour start, the sum of the twelve squared 5-minute log returns within the hour. The first runs from
    the last close before the hour; a 5-minute interval without a candle repeats the previous close. Hours with no
    close in the hour before them are left out."""
    out: dict[int, float] = {}
    for hour in hours:
        previous = _last_at_or_before(closes, hour - FIVE_MINUTES)
        if previous is None:
            continue
        total, last = 0.0, previous
        for k in range(12):
            current = closes.get(hour + k * FIVE_MINUTES, last)
            total += math.log(current / last) ** 2
            last = current
        out[hour] = total
    return out


def _last_at_or_before(closes: dict[int, float], stamp: int, *, lookback: int = 12) -> float | None:
    for k in range(lookback):
        if stamp - k * FIVE_MINUTES in closes:
            return closes[stamp - k * FIVE_MINUTES]
    return None


def funding_rates(
    symbol: str, start: datetime, end: datetime, *, client: httpx.Client, cache_dir: Path
) -> list[tuple[int, float]]:
    """``[(settlement time, rate)]`` of a Binance perpetual in ``[start, end)``, oldest first."""
    out: list[tuple[int, float]] = []
    for first, following in _months(start, end):
        lo, hi = int(first.timestamp()), int(following.timestamp())

        def fetch(lo: int = lo, hi: int = hi) -> list[list[float]]:
            params = {"symbol": symbol, "startTime": lo * 1000, "endTime": hi * 1000 - 1, "limit": 1000}
            return [[r["fundingTime"] // 1000, float(r["fundingRate"])] for r in _get(client, BINANCE_FUNDING, params)]

        rows = _cached(cache_dir / f"funding-{symbol}-{first:%Y%m}.json", following, fetch)
        out += [(int(t), float(rate)) for t, rate in rows]
    lo, hi = int(start.timestamp()), int(end.timestamp())
    return sorted({t: r for t, r in out if lo <= t < hi}.items())


def open_interest_hourly(
    symbol: str, start: datetime, end: datetime, *, client: httpx.Client, cache_dir: Path
) -> dict[int, float]:
    """``{snapshot time: open interest in coins}`` of a Bybit USDT perpetual, hourly snapshots in ``[start, end)``
    taken at least an hour ago."""
    out: dict[int, float] = {}
    for first, following in _months(start, end):
        lo, hi = int(first.timestamp()), int(following.timestamp())

        def fetch(lo: int = lo, hi: int = hi) -> list[list[float]]:
            rows: dict[int, float] = {}
            cursor = ""
            while True:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": "1h",
                    "startTime": lo * 1000,
                    "endTime": hi * 1000 - 1,
                    "limit": BYBIT_PAGE,
                    "cursor": cursor,
                }
                body = _get(client, BYBIT_OPEN_INTEREST, params)
                if body.get("retCode") != 0:
                    raise RuntimeError(f"bybit open interest for {symbol}: {body.get('retMsg')}")
                page = body["result"]["list"]
                rows |= {int(r["timestamp"]) // 1000: float(r["openInterest"]) for r in page}
                cursor = body["result"].get("nextPageCursor") or ""
                if not cursor or not page:
                    break
            return [[t, v] for t, v in sorted(rows.items()) if lo <= t < hi]

        for stamp, value in _cached(cache_dir / f"oi-{symbol}-{first:%Y%m}.json", following, fetch):
            if stamp + HOUR <= time.time():
                out[int(stamp)] = float(value)
    lo, hi = int(start.timestamp()), int(end.timestamp())
    return {t: v for t, v in out.items() if lo <= t < hi}
