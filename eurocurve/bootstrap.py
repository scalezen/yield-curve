r"""Bootstrapping a €STR OIS discount curve from swap par rates.

## What an OIS is

A euro overnight index swap: you pay a fixed rate S, and receive €STR compounded
daily over the same period. Maturities up to 1Y settle in one payment; beyond that,
annually. Both legs ACT/360. Trades start spot (T+2).

## The one equation

At inception the swap is worth zero, so PV(fixed) = PV(float):

    S * sum_i alpha_i * DF(t_i)  =  PV of the floating leg

The left side is the fixed rate times the **annuity** (sum of accruals times discount
factors). The right side is where the magic happens.

## Why the floating leg collapses to 1 - DF(T)

Take one floating period [t_{i-1}, t_i]. The payment is the compounded overnight rate
over that period, times the accrual. In a single-curve world -- where the rate you
*project* is the same rate you *discount* with, which for €STR OIS is exactly true --
the forward compounded rate over that period satisfies

    1 + alpha_i * F_i  =  DF(t_{i-1}) / DF(t_i)

so the present value of that one payment is

    alpha_i * F_i * DF(t_i) = (DF(t_{i-1})/DF(t_i) - 1) * DF(t_i) = DF(t_{i-1}) - DF(t_i)

Sum over all periods and the middle terms cancel -- it telescopes:

    PV(float) = DF(t_0) - DF(t_n) = 1 - DF(T)

That is the whole trick. The floating leg of an OIS is worth "a euro now minus a euro
at maturity", regardless of what the forwards actually do. Which gives the par rate:

    S(T) = (1 - DF(T)) / sum_i alpha_i * DF(t_i)

## Bootstrapping

Now walk up the maturities. The 1Y swap has a single payment, so its equation has one
unknown and you can solve it in your head:

    DF(1Y) = 1 / (1 + S_1Y * alpha_1)

The 2Y swap pays at 1Y and 2Y. You already know DF(1Y). One equation, one unknown
again. Repeat. This is why it is called a bootstrap: each instrument adds exactly one
pillar, and the curve pulls itself up by its own bootstraps.

The catch is gaps. If your quotes jump 10Y -> 12Y -> 15Y, the 15Y swap pays at 11Y,
13Y and 14Y too, and you do not have pillars there. So we do not use the closed form;
we root-solve for DF(T) with the intermediate DFs coming from interpolating between
the last known pillar and the candidate. Slightly slower, always correct, and it makes
the interpolation scheme an explicit part of the curve's definition rather than
something applied afterwards.

## Bootstrap vs. fit -- the thing to actually take away

A bootstrapped curve reprices its inputs **exactly**, by construction. A fitted curve
(Stage 1) does not: it trades exactness for smoothness and for a small, interpretable
parameter set. Use a bootstrap when you must not lose money on the instruments you can
actually trade. Use a fit when you want to compare curves across time or countries, or
when your inputs are noisy and you *want* the noise smoothed away.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from .curve import DiscountCurve
from .daycount import accruals, add_tenor, annual_schedule, parse_tenor, year_fraction

#: Time axis convention for the curve itself. Accruals use ACT/360 (the contract's
#: convention); the curve's t-coordinate uses ACT/365F. Mixing these up is the second
#: most common bug in curve code, after percent-vs-decimal.
CURVE_BASIS = "ACT/365F"
ACCRUAL_BASIS = "ACT/360"


@dataclass
class OISSwap:
    """A EUR €STR overnight index swap quoted at par.

    Attributes
    ----------
    tenor  : '1Y', '18M', '30Y', ...
    rate   : par fixed rate as a DECIMAL (0.0235, not 2.35)
    spot   : the swap's effective (start) date
    """

    tenor: str
    rate: float
    spot: dt.date
    pay_dates: list[dt.date] = field(init=False)
    times: np.ndarray = field(init=False)
    alphas: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pay_dates = annual_schedule(self.spot, self.tenor)
        self.times = np.array(
            [year_fraction(self.spot, d, CURVE_BASIS) for d in self.pay_dates]
        )
        self.alphas = np.array(accruals(self.spot, self.pay_dates, ACCRUAL_BASIS))

    @property
    def maturity(self) -> float:
        return float(self.times[-1])

    def annuity(self, curve: DiscountCurve) -> float:
        """sum_i alpha_i * DF(t_i) -- the PV of receiving 1.0 of fixed rate."""
        return float(np.sum(self.alphas * np.asarray(curve.df(self.times), dtype=float)))

    def pv(self, curve: DiscountCurve, notional: float = 1.0) -> float:
        """PV to the FIXED RECEIVER. Should be ~0 for a swap quoted at par."""
        float_leg = 1.0 - float(curve.df(self.maturity))
        return notional * (self.rate * self.annuity(curve) - float_leg)


def par_rate(swap: OISSwap, curve: DiscountCurve) -> float:
    """The fixed rate that would make this swap worth zero on this curve.

    S = (1 - DF(T)) / annuity. If you feed in the curve you bootstrapped, this
    must return the quote you started from. That is the test in `tests/`.
    """
    return (1.0 - float(curve.df(swap.maturity))) / swap.annuity(curve)


def bootstrap_ois(
    swaps: Sequence[OISSwap],
    overnight_rate: float | None = None,
    as_of: dt.date | None = None,
) -> DiscountCurve:
    """Build a discount curve that exactly reprices every swap given.

    Parameters
    ----------
    swaps
        Par-quoted OIS, any order. Duplicated maturities are an error.
    overnight_rate
        Optional €STR fixing (decimal). Anchors the very short end with a 1-day
        pillar so the curve does not extrapolate blindly below its first swap.
        Without it the first pillar might be 1Y away and everything shorter is a
        straight line from DF(0)=1, which is fine for discounting but useless if
        you want a sensible O/N forward.

    Returns
    -------
    DiscountCurve with one pillar per swap (plus the overnight anchor).
    """
    swaps = sorted(swaps, key=lambda s: s.maturity)
    mats = [round(s.maturity, 9) for s in swaps]
    if len(set(mats)) != len(mats):
        raise ValueError("two swaps share a maturity -- the system is degenerate")

    times: list[float] = [0.0]
    dfs: list[float] = [1.0]

    if overnight_rate is not None:
        t_on = 1.0 / 365.0
        # simple ACT/360 accrual over one day
        times.append(t_on)
        dfs.append(1.0 / (1.0 + overnight_rate * (1.0 / 360.0)))

    for s in swaps:
        if s.maturity <= times[-1]:
            raise ValueError(f"swap {s.tenor} matures inside the curve already built")

        def objective(log_df_T: float, s=s) -> float:
            trial = DiscountCurve(times + [s.maturity], dfs + [float(np.exp(log_df_T))])
            return par_rate(s, trial) - s.rate

        # Bracket generously: DF at T between exp(-0.30*T) (30% rates) and
        # exp(+0.10*T) (deeply negative rates -- the euro has been there).
        lo, hi = -0.30 * s.maturity, 0.10 * s.maturity
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:  # pragma: no cover - only on absurd input
            raise RuntimeError(
                f"cannot bracket DF for {s.tenor} at {s.rate:.4%}; "
                "check the quote is a decimal, not a percentage"
            )
        log_df = brentq(objective, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
        times.append(s.maturity)
        dfs.append(float(np.exp(log_df)))

    return DiscountCurve(times, dfs, as_of=as_of)


def load_quotes(path, spot: dt.date) -> list[OISSwap]:
    """Read a two-column CSV (tenor, rate_pct) into OISSwap objects."""
    import pandas as pd

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "rate_pct" in df.columns:
        rates = df["rate_pct"].astype(float) / 100.0
    elif "rate" in df.columns:
        rates = df["rate"].astype(float)
    else:
        raise ValueError("CSV needs a 'rate_pct' or 'rate' column")
    return [
        OISSwap(tenor=str(t).strip(), rate=float(r), spot=spot)
        for t, r in zip(df["tenor"], rates)
    ]


def spot_date(valuation: dt.date, lag_days: int = 2) -> dt.date:
    """EUR spot is T+2 business days. We treat the curve's origin as the spot date,
    which is a simplification: strictly, discounting from today to spot needs the
    overnight rate too. On a 2-day stub at 2% that is about 1 basis point of PV --
    irrelevant here, not irrelevant on a real book."""
    d = valuation
    for _ in range(lag_days):
        d = add_tenor(d, "1D")
    return d


def reprice_check(swaps: Sequence[OISSwap], curve: DiscountCurve):
    """Every bootstrapped curve should return its inputs to the last basis point.
    Run this every single time. It catches day-count errors, schedule errors and
    percent/decimal errors in one line."""
    import pandas as pd

    rows = []
    for s in sorted(swaps, key=lambda x: x.maturity):
        implied = par_rate(s, curve)
        rows.append(
            {
                "tenor": s.tenor,
                "quoted_pct": s.rate * 100,
                "reprice_pct": implied * 100,
                "error_bp": (implied - s.rate) * 1e4,
                "pv_per_100m": s.pv(curve, notional=1e8),
            }
        )
    return pd.DataFrame(rows)
