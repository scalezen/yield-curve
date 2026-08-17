import datetime as dt

import pytest

from eurocurve.daycount import (
    accruals,
    add_tenor,
    annual_schedule,
    parse_tenor,
    year_fraction,
)


def test_act360_vs_act365():
    a, b = dt.date(2026, 1, 1), dt.date(2027, 1, 1)
    assert year_fraction(a, b, "ACT/360") == pytest.approx(365 / 360)
    assert year_fraction(a, b, "ACT/365F") == pytest.approx(1.0)
    # The gap is why quoting a rate without its day count is meaningless:
    # the same cashflow is 1.39% larger on ACT/360.
    assert year_fraction(a, b, "ACT/360") / year_fraction(a, b, "ACT/365F") > 1.013


def test_thirty_360_ignores_month_length():
    assert year_fraction(dt.date(2026, 1, 31), dt.date(2026, 2, 28), "30/360") == pytest.approx(0.0)
    assert year_fraction(dt.date(2026, 1, 15), dt.date(2026, 7, 15), "30/360") == pytest.approx(0.5)


def test_backwards_dates_rejected():
    with pytest.raises(ValueError):
        year_fraction(dt.date(2026, 6, 1), dt.date(2026, 5, 1))


@pytest.mark.parametrize(
    "tenor,years",
    [("3M", 0.25), ("6M", 0.5), ("18M", 1.5), ("1Y", 1.0), ("30Y", 30.0), ("1W", 7 / 365)],
)
def test_parse_tenor(tenor, years):
    assert parse_tenor(tenor) == pytest.approx(years)


def test_month_addition_clamps_day():
    # 31 Jan + 1M must not overflow into March.
    assert add_tenor(dt.date(2026, 1, 31), "1M").month == 2


def test_rolls_off_weekends():
    sat = dt.date(2026, 8, 15)  # a Saturday
    assert sat.weekday() == 5
    assert add_tenor(sat - dt.timedelta(days=1), "1D").weekday() < 5


def test_annual_schedule_5y_has_five_payments():
    s = dt.date(2026, 8, 13)
    sched = annual_schedule(s, "5Y")
    assert len(sched) == 5
    assert sched == sorted(sched)
    assert (sched[-1] - s).days > 1800


def test_broken_tenor_generates_backwards():
    """An 18M swap pays at 6M and 18M -- NOT at 12M and 24M.

    Forward generation would give the wrong schedule and a curve that still looks
    plausible on a chart. This is the test that catches it.
    """
    s = dt.date(2026, 8, 13)
    sched = annual_schedule(s, "18M")
    assert len(sched) == 2
    first_gap_days = (sched[0] - s).days
    assert 150 < first_gap_days < 200  # ~6 months, not ~12
    assert 520 < (sched[-1] - s).days < 570  # ~18 months


def test_short_tenor_is_single_payment():
    s = dt.date(2026, 8, 13)
    assert len(annual_schedule(s, "3M")) == 1
    assert len(annual_schedule(s, "1Y")) == 1


def test_accruals_sum_to_total_period():
    s = dt.date(2026, 8, 13)
    sched = annual_schedule(s, "10Y")
    a = accruals(s, sched)
    assert sum(a) == pytest.approx(year_fraction(s, sched[-1], "ACT/360"))
