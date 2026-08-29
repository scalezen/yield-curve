import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from eurocurve.bootstrap import (
    OISSwap,
    bootstrap_ois,
    load_quotes,
    par_rate,
    reprice_check,
    spot_date,
)
from eurocurve.daycount import year_fraction

DATA = Path(__file__).resolve().parent.parent / "data" / "eur_ois_sample.csv"
VALUATION = dt.date(2026, 8, 11)
SPOT = spot_date(VALUATION)


@pytest.fixture(scope="module")
def swaps():
    return load_quotes(DATA, spot=SPOT)


@pytest.fixture(scope="module")
def curve(swaps):
    return bootstrap_ois(swaps, overnight_rate=0.0193)


# --------------------------------------------------------------------------- #
# The test that matters
# --------------------------------------------------------------------------- #


def test_bootstrap_reprices_every_input_exactly(swaps, curve):
    """A bootstrapped curve returns its own inputs. Not approximately -- to
    machine precision, because each pillar was solved to make it so.

    This single assertion catches day-count errors, schedule errors, and
    percent-vs-decimal errors. Run it after every change.
    """
    chk = reprice_check(swaps, curve)
    assert chk["error_bp"].abs().max() < 1e-6


def test_par_swaps_have_zero_pv(swaps, curve):
    for s in swaps:
        assert s.pv(curve, notional=1e8) == pytest.approx(
            0.0, abs=1e-3
        )  # <0.1 cent on EUR100m


# --------------------------------------------------------------------------- #
# Checking the algebra by hand
# --------------------------------------------------------------------------- #


def test_one_year_matches_the_closed_form():
    """For a single-payment OIS the bootstrap has an exact solution:

        DF(T) = 1 / (1 + S * alpha)

    If the root-finder disagrees with pencil and paper, the root-finder is wrong.
    """
    s = OISSwap("1Y", 0.02005, SPOT)
    c = bootstrap_ois([s])
    alpha = year_fraction(SPOT, s.pay_dates[-1], "ACT/360")
    expected = 1.0 / (1.0 + s.rate * alpha)
    assert c.df(s.maturity) == pytest.approx(expected, rel=1e-12)


def test_two_year_matches_the_closed_form():
    """DF(T2) = (1 - S*alpha_1*DF(T1)) / (1 + S*alpha_2), solved by hand."""
    s1 = OISSwap("1Y", 0.02005, SPOT)
    s2 = OISSwap("2Y", 0.02170, SPOT)
    c = bootstrap_ois([s1, s2])
    a1, a2 = s2.alphas
    df1 = c.df(s2.times[0])
    expected = (1.0 - s2.rate * a1 * df1) / (1.0 + s2.rate * a2)
    assert c.df(s2.maturity) == pytest.approx(expected, rel=1e-10)


def test_float_leg_telescopes_to_one_minus_df(swaps, curve):
    """PV(float) = 1 - DF(T). Verified by summing the periods explicitly rather
    than using the shortcut, on the 10y swap."""
    s = next(x for x in swaps if x.tenor == "10Y")
    t = np.concatenate([[0.0], s.times])
    dfs = np.asarray(curve.df(t), dtype=float)
    period_pvs = dfs[:-1] - dfs[1:]  # DF(t_{i-1}) - DF(t_i) per period
    assert period_pvs.sum() == pytest.approx(
        1.0 - float(curve.df(s.maturity)), rel=1e-12
    )


def test_flat_curve_gives_recognisable_rates():
    """Sanity anchor: if every OIS quote is 2%, the zero curve should be ~2% too
    (up to the ACT/360-vs-ACT/365 basis, which is a real ~1.4% scaling, not a bug)."""
    quotes = [OISSwap(t, 0.02, SPOT) for t in ("1Y", "2Y", "5Y", "10Y")]
    c = bootstrap_ois(quotes)
    z10 = c.zero(quotes[-1].maturity, "annual")
    assert 0.0200 < z10 < 0.0210


# --------------------------------------------------------------------------- #
# Structural properties
# --------------------------------------------------------------------------- #


def test_discount_factors_strictly_decreasing(curve):
    assert np.all(np.diff(curve.dfs) < 0)


def test_no_negative_forwards_on_this_data(curve):
    """A curve with negative instantaneous forwards implies you can make money by
    holding cash instead of lending -- possible in the euro, but not on THIS data,
    where every quote is positive. If this fails, an interpolation artefact has
    crept in."""
    grid = np.linspace(0.05, 29.5, 400)
    assert np.all(np.asarray(curve.inst_forward(grid)) > 0)


def test_gapped_quotes_still_reprice():
    """The 15Y swap pays at 11y, 13y and 14y where there are no pillars. The
    closed-form bootstrap cannot handle that; the root-solve can. This test is
    the whole reason the implementation uses brentq."""
    quotes = [
        OISSwap("1Y", 0.02005, SPOT),
        OISSwap("10Y", 0.02638, SPOT),
        OISSwap("15Y", 0.02712, SPOT),
        OISSwap("30Y", 0.02605, SPOT),
    ]
    c = bootstrap_ois(quotes)
    chk = reprice_check(quotes, c)
    assert chk["error_bp"].abs().max() < 1e-6


def test_broken_tenor_reprices(swaps, curve):
    s18 = next(x for x in swaps if x.tenor == "18M")
    assert par_rate(s18, curve) == pytest.approx(s18.rate, abs=1e-10)


def test_duplicate_maturities_rejected():
    with pytest.raises(ValueError):
        bootstrap_ois([OISSwap("5Y", 0.024, SPOT), OISSwap("5Y", 0.025, SPOT)])


def test_percent_instead_of_decimal_is_caught_or_absurd():
    """Passing 2.638 instead of 0.02638 is the classic blunder. It should either
    raise, or produce a discount factor so small that nobody could miss it."""
    try:
        c = bootstrap_ois([OISSwap("10Y", 2.638, SPOT)])
    except RuntimeError:
        return
    assert c.df(10.0) < 0.1


def test_bootstrap_is_order_independent(swaps):
    a = bootstrap_ois(swaps, overnight_rate=0.0193)
    b = bootstrap_ois(list(reversed(swaps)), overnight_rate=0.0193)
    np.testing.assert_allclose(a.dfs, b.dfs, rtol=1e-12)


def test_overnight_anchor_shows_up_in_the_short_end(swaps):
    """The €STR anchor reshapes the sub-1-week curve and disturbs nothing else.

    Deliberately anchored well away from the 1W quote (2.50% vs 1.93%) so the
    effect is unambiguous -- at a realistic €STR the interpolated 1-day point is
    already almost exactly right, which is itself reassuring.
    """
    with_on = bootstrap_ois(swaps, overnight_rate=0.0250)
    without = bootstrap_ois(swaps)
    assert abs(with_on.df(1 / 365) - without.df(1 / 365)) > 1e-6
    for s in swaps:
        assert with_on.df(s.maturity) == pytest.approx(
            without.df(s.maturity), rel=1e-12
        )


def test_sample_data_loads_and_is_sane(swaps):
    assert len(swaps) == 21
    assert all(0.0 < s.rate < 0.10 for s in swaps)
    mats = [s.maturity for s in swaps]
    assert mats == sorted(mats)
    assert mats[-1] > 29.0
