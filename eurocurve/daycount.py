"""Day counts and payment schedules.

A day count convention answers one question: given two dates, how much of a year
elapsed? It sounds like bookkeeping, but it moves real money. A 30-year EUR swap
priced ACT/365 instead of ACT/360 is off by roughly 1.4% of every coupon.

The rule for EUR OIS: **ACT/360 on both legs**. The rule for the ECB's published
yield curve: it isn't a day count at all, it's continuous compounding over a
maturity expressed in years. Keep those two worlds separate in your head.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Iterable

# Weekday of the "modified following" business day convention. We deliberately do
# NOT model the TARGET2 holiday calendar here -- for a toy curve the error is a day
# or two on a schedule, well under the noise in the quotes. If you productionise
# this, swap in a real calendar and this is the only file that changes.
_WEEKEND = {5, 6}


def year_fraction(start: dt.date, end: dt.date, convention: str = "ACT/360") -> float:
    """Year fraction between two dates.

    Supported: ACT/360, ACT/365F, 30/360 (bond basis), ACT/ACT (ISDA-lite).
    """
    conv = convention.upper().replace("-", "/")
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")

    if conv == "ACT/360":
        return (end - start).days / 360.0
    if conv in ("ACT/365F", "ACT/365"):
        return (end - start).days / 365.0
    if conv in ("30/360", "30E/360", "BOND"):
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        return (
            360 * (end.year - start.year)
            + 30 * (end.month - start.month)
            + (d2 - d1)
        ) / 360.0
    if conv == "ACT/ACT":
        # Good enough for a toy: actual days over the average calendar year.
        return (end - start).days / 365.25
    raise ValueError(f"unknown day count convention: {convention}")


def parse_tenor(tenor: str) -> float:
    """'3M' -> 0.25, '18M' -> 1.5, '30Y' -> 30.0, '1W' -> 7/365, 'ON' -> 1/365.

    Returns a maturity in years. Used for labelling and for the ECB curve, which
    is quoted directly against a maturity in years rather than against dates.
    """
    t = tenor.strip().upper()
    if t in ("ON", "O/N", "TN", "T/N"):
        return 1.0 / 365.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([DWMY])", t)
    if not m:
        raise ValueError(f"cannot parse tenor {tenor!r}")
    n, unit = float(m.group(1)), m.group(2)
    return {"D": n / 365.0, "W": n * 7 / 365.0, "M": n / 12.0, "Y": n}[unit]


def add_tenor(start: dt.date, tenor: str) -> dt.date:
    """Advance a date by a tenor, then roll to the next business day."""
    t = tenor.strip().upper()
    if t in ("ON", "O/N"):
        return _roll(start + dt.timedelta(days=1))
    m = re.fullmatch(r"(\d+)\s*([DWMY])", t)
    if not m:
        raise ValueError(f"cannot parse tenor {tenor!r}")
    n, unit = int(m.group(1)), m.group(2)
    if unit == "D":
        end = start + dt.timedelta(days=n)
    elif unit == "W":
        end = start + dt.timedelta(weeks=n)
    elif unit == "M":
        end = _add_months(start, n)
    else:
        end = _add_months(start, 12 * n)
    return _roll(end)


def _add_months(d: dt.date, n: int) -> dt.date:
    """Add months, clamping the day of month (31 Jan + 1M = 28/29 Feb)."""
    total = d.month - 1 + n
    year = d.year + total // 12
    month = total % 12 + 1
    last = [31, 29 if _leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return dt.date(year, month, min(d.day, last))


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _roll(d: dt.date) -> dt.date:
    """Modified following: roll forward off a weekend, but never into a new month."""
    orig_month = d.month
    while d.weekday() in _WEEKEND:
        d += dt.timedelta(days=1)
    if d.month != orig_month:  # rolled past month end -> go backwards instead
        while d.weekday() in _WEEKEND:
            d -= dt.timedelta(days=1)
    return d


def annual_schedule(start: dt.date, maturity_tenor: str) -> list[dt.date]:
    """Payment dates for an annual-pay EUR OIS, earliest first, maturity last.

    EUR OIS pays annually beyond 1Y, and a single payment at maturity for anything
    1Y or shorter. That single-payment case is what makes the short end of the
    bootstrap a one-liner.

    Dates are generated **backwards from maturity**, which is the market convention
    and the only way to handle broken tenors like 18M correctly: an 18M swap pays
    at 6M and 18M, not at 12M and 24M. Generating forwards from the start date
    would silently give you the wrong schedule and a curve that looks fine.
    """
    years = parse_tenor(maturity_tenor)
    end = _unrolled_end(start, maturity_tenor)
    if years <= 1.0 + 1e-9:
        return [_roll(end)]

    dates: list[dt.date] = []
    k = 0
    while True:
        d = _add_months(end, -12 * k)
        if d <= start:
            break
        dates.append(d)
        k += 1
    return [_roll(d) for d in reversed(dates)]


def _unrolled_end(start: dt.date, tenor: str) -> dt.date:
    """Maturity date before any business-day rolling (needed for back-generation)."""
    t = tenor.strip().upper()
    m = re.fullmatch(r"(\d+)\s*([DWMY])", t)
    if not m:
        raise ValueError(f"cannot parse tenor {tenor!r}")
    n, unit = int(m.group(1)), m.group(2)
    if unit == "D":
        return start + dt.timedelta(days=n)
    if unit == "W":
        return start + dt.timedelta(weeks=n)
    if unit == "M":
        return _add_months(start, n)
    return _add_months(start, 12 * n)


def accruals(start: dt.date, dates: Iterable[dt.date], convention: str = "ACT/360") -> list[float]:
    """Year fractions for consecutive periods running from `start` through `dates`."""
    out, prev = [], start
    for d in dates:
        out.append(year_fraction(prev, d, convention))
        prev = d
    return out
