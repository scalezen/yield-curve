r"""The one object everything else produces: a discount curve.

A discount curve is a function DF(t): "one euro paid at time t is worth DF(t) euros
now". Everything else on a rates desk is a re-expression of that single function:

    zero rate (continuous)   y(t)  = -ln(DF(t)) / t
    zero rate (annual)       y(t)  = DF(t)^(-1/t) - 1
    forward between t1,t2    F     = (DF(t1)/DF(t2) - 1) / (t2 - t1)
    instantaneous forward    f(t)  = -d/dt ln(DF(t))

If you can move between those four without hesitating, you understand yield curves.

## The interpolation choice matters more than people expect

We store DF at a set of pillar times and interpolate **linearly in log(DF)**. That is
not an aesthetic preference. Since f(t) = -d/dt ln DF(t), a curve that is piecewise
linear in log DF has **piecewise constant instantaneous forwards**. That is the most
defensible thing to assume between two pillars: you learned nothing between them, so
you assume the forward rate is flat there.

Compare the alternatives:
  * linear in DF          -> forwards drift oddly and can imply arbitrage across pillars
  * linear in zero rates  -> forwards get sawtooth jumps; popular, but it is a choice
                             that puts artefacts into exactly the quantity traders look at
  * cubic spline on zeros -> smooth zeros, but forwards can oscillate and go negative
                             in ways nothing in the market justifies

Rule of thumb: judge an interpolation scheme by looking at the **forward** curve it
produces, never the zero curve. Every scheme looks fine on the zero curve.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

    from .nss import NSSParams

ZeroCompounding = Literal["continuous", "annual", "simple"]
RateCompounding = Literal["continuous", "annual"]


class DiscountCurve:
    """Piecewise log-linear discount curve, i.e. piecewise-constant forwards.

    Parameters
    ----------
    times : increasing array of year fractions from the valuation date
    dfs   : discount factors at those times
    """

    def __init__(
        self, times: npt.ArrayLike, dfs: npt.ArrayLike, as_of: dt.date | None = None
    ):
        t = np.asarray(times, dtype=float)
        d = np.asarray(dfs, dtype=float)
        if t.shape != d.shape:
            raise ValueError("times and dfs must be the same length")
        order = np.argsort(t)
        t, d = t[order], d[order]
        if t[0] > 1e-12:  # always anchor DF(0) = 1
            t = np.concatenate([[0.0], t])
            d = np.concatenate([[1.0], d])
        if np.any(d <= 0):
            raise ValueError("discount factors must be strictly positive")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly increasing")
        self.times = t
        self.dfs = d
        self._logdf = np.log(d)
        self.as_of = as_of

    # -- core ---------------------------------------------------------------
    @overload
    def df(self, t: float) -> float: ...
    @overload
    def df(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    def df(self, t):
        """Discount factor. Log-linear between pillars; flat-forward extrapolation
        beyond the last pillar (the only extrapolation that does not eventually
        produce nonsense)."""
        tt = np.asarray(t, dtype=float)
        ld = np.interp(tt, self.times, self._logdf)
        # np.interp clamps outside the range; replace the right-hand clamp with a
        # continuation of the last log-linear slope.
        last_t, prev_t = self.times[-1], self.times[-2]
        slope = (self._logdf[-1] - self._logdf[-2]) / (last_t - prev_t)
        ld = np.where(tt > last_t, self._logdf[-1] + slope * (tt - last_t), ld)
        out = np.exp(ld)
        return out if np.ndim(t) else float(out)

    @overload
    def zero(self, t: float, compounding: ZeroCompounding = "continuous") -> float: ...
    @overload
    def zero(
        self, t: npt.ArrayLike, compounding: ZeroCompounding = "continuous"
    ) -> npt.NDArray[np.float64]: ...
    def zero(self, t, compounding: ZeroCompounding = "continuous"):
        """Zero rate to time t, as a decimal."""
        tt = np.asarray(t, dtype=float)
        safe = np.where(tt <= 1e-12, 1e-12, tt)
        d = np.asarray(self.df(safe), dtype=float)
        if compounding == "continuous":
            out = -np.log(d) / safe
        elif compounding == "annual":
            out = d ** (-1.0 / safe) - 1.0
        elif compounding == "simple":
            out = (1.0 / d - 1.0) / safe
        else:
            raise ValueError(f"unknown compounding {compounding!r}")
        return out if np.ndim(t) else float(out)

    @overload
    def forward(self, t1: float, t2: float, simple: bool = True) -> float: ...
    @overload
    def forward(
        self, t1: npt.ArrayLike, t2: npt.ArrayLike, simple: bool = True
    ) -> npt.NDArray[np.float64]: ...
    def forward(self, t1, t2, simple: bool = True):
        """Forward rate between two future times.

        `simple=True` gives the money-market convention (1 + F*tau); `False` gives
        the continuously-compounded equivalent.
        """
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        tau = t2 - t1
        if np.any(tau <= 0):
            raise ValueError("t2 must exceed t1")
        ratio = np.asarray(self.df(t1), dtype=float) / np.asarray(
            self.df(t2), dtype=float
        )
        out = (ratio - 1.0) / tau if simple else np.log(ratio) / tau
        return out if np.ndim(ratio) else float(out)

    @overload
    def inst_forward(self, t: float, h: float = 1e-4) -> float: ...
    @overload
    def inst_forward(
        self, t: npt.ArrayLike, h: float = 1e-4
    ) -> npt.NDArray[np.float64]: ...
    def inst_forward(self, t, h: float = 1e-4):
        """Instantaneous forward f(t) = -d ln DF / dt, by central difference.

        On a log-linear curve this is piecewise constant, so the derivative is
        genuinely undefined *at* a pillar. Plot it and you will see the steps --
        that is a feature of the interpolation, not a bug in the code.
        """
        tt = np.asarray(t, dtype=float)
        lo = np.maximum(tt - h, 0.0)
        hi = tt + h
        out = -(np.log(self.df(hi)) - np.log(self.df(lo))) / (hi - lo)
        return out if np.ndim(t) else float(out)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_zero_rates(
        cls,
        times: npt.ArrayLike,
        zeros: npt.ArrayLike,
        compounding: RateCompounding = "continuous",
        as_of: dt.date | None = None,
    ) -> DiscountCurve:
        t = np.asarray(times, dtype=float)
        z = np.asarray(zeros, dtype=float)
        if compounding == "continuous":
            d = np.exp(-z * t)
        elif compounding == "annual":
            d = (1.0 + z) ** (-t)
        else:
            raise ValueError(compounding)
        return cls(t, d, as_of=as_of)

    @classmethod
    def from_nss(
        cls,
        params: NSSParams | Sequence[float],
        times: npt.ArrayLike | None = None,
        as_of: dt.date | None = None,
    ) -> DiscountCurve:
        """Sample a fitted NSS curve onto a pillar grid.

        Note this *discretises* a smooth curve, so the resulting object's forwards
        will be stepped rather than smooth. Use a fine grid, or evaluate the NSS
        functions directly, when forwards are what you care about.
        """
        from .nss import nss_discount

        if times is None:
            times = np.concatenate(
                [np.arange(1, 12) / 12.0, np.arange(1.0, 30.25, 0.25)]
            )
        t = np.asarray(times, dtype=float)
        return cls(t, np.asarray(nss_discount(t, params), dtype=float), as_of=as_of)

    # -- misc ---------------------------------------------------------------
    def to_frame(self, times: npt.ArrayLike | None = None) -> pd.DataFrame:
        import pandas as pd

        t = self.times[1:] if times is None else np.asarray(times, dtype=float)
        return pd.DataFrame(
            {
                "maturity": t,
                "df": self.df(t),
                "zero_cc": self.zero(t),
                "zero_annual": self.zero(t, "annual"),
                "inst_fwd": self.inst_forward(t),
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DiscountCurve({len(self.times)} pillars, "
            f"{self.times[1]:.3f}y-{self.times[-1]:.2f}y, as_of={self.as_of})"
        )
