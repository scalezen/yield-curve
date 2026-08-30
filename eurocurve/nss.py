r"""Nelson-Siegel-Svensson.

## The idea

You observe maybe 18 yields. You want a yield at *every* maturity, and you want the
answer to be smooth, because a curve with kinks in it produces forward rates that
oscillate wildly and price nothing sensibly. Interpolation gives you smoothness in
the yields but not in the forwards. So instead you assume a *shape*, with a handful
of parameters, and fit it.

Nelson & Siegel (1987) proposed writing the **instantaneous forward** rate as a sum
of exponential decay terms:

    f(tau) = beta0 + beta1*exp(-tau/lam1) + beta2*(tau/lam1)*exp(-tau/lam1)

Svensson (1994) added a second hump so the curve can bend twice:

    f(tau) = beta0
           + beta1 * exp(-tau/lam1)
           + beta2 * (tau/lam1) * exp(-tau/lam1)
           + beta3 * (tau/lam2) * exp(-tau/lam2)

Read the four terms as four economic stories:

    beta0   the level the curve tends to at very long maturities
    beta1   the short-end deviation from that level; beta0+beta1 = the instantaneous
            short rate, since exp(0)=1 and the hump terms vanish at tau=0
    beta2   the size of a hump (or trough, if negative) in the medium maturities
    beta3   a second, usually longer-dated hump
    lam1    where the first hump sits (peaks at tau = lam1)
    lam2    where the second hump sits

## Getting from forwards to spots

The spot (zero) rate is the *average* forward over the life:

    y(T) = (1/T) * integral_0^T f(u) du

Do that integral term by term and you get the familiar loading functions. Define

    g(tau, lam) = (1 - exp(-tau/lam)) / (tau/lam)

then

    y(tau) = beta0
           + beta1 * g(tau, lam1)
           + beta2 * (g(tau, lam1) - exp(-tau/lam1))
           + beta3 * (g(tau, lam2) - exp(-tau/lam2))

Two limits worth carrying in your head:
    tau -> 0    g -> 1, hump terms -> 0, so y(0) = beta0 + beta1
    tau -> inf  everything decays, so y(inf) = beta0

## Compounding

The ECB publishes this curve **continuously compounded**, so:

    DF(tau) = exp(-y(tau) * tau)

If you ever need annual compounding: y_ann = exp(y_cont) - 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

_EPS = 1e-10


@dataclass(frozen=True)
class NSSParams:
    """The six Svensson parameters. Rates as decimals, taus in years."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2],
            dtype=float,
        )

    @classmethod
    def from_array(cls, x: Sequence[float]) -> NSSParams:
        return cls(*[float(v) for v in x])

    @property
    def short_rate(self) -> float:
        """y(0) = beta0 + beta1 -- the instantaneous overnight rate the fit implies."""
        return self.beta0 + self.beta1

    @property
    def long_rate(self) -> float:
        """y(inf) = beta0."""
        return self.beta0

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NSSParams(beta0={self.beta0:.4%}, beta1={self.beta1:.4%}, beta2={self.beta2:.4%}, beta3={self.beta3:.4%}, "
            f"tau1={self.tau1:.3f}y, tau2={self.tau2:.3f}y)"
        )


def _g(tau: np.ndarray, lam: float) -> np.ndarray:
    """(1 - exp(-x)) / x with x = tau/lam, numerically safe at x -> 0 (limit 1).

    The naive expression is 0/0 at tau=0, and worse, it loses precision for small
    x even before it blows up. We switch to the series 1 - x/2 + x^2/6 below a
    threshold. This kind of care is why the tests check tau=0 explicitly.
    """
    x = np.asarray(tau, dtype=float) / lam
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)  # dummy value to keep the division finite
    out = np.where(small, 1.0 - x / 2.0 + x * x / 6.0, (1.0 - np.exp(-xs)) / xs)
    return out


def nss_spot(tau, p: NSSParams | Sequence[float]) -> np.ndarray:
    """Continuously-compounded zero rate at maturity `tau` (years, scalar or array)."""
    p = p if isinstance(p, NSSParams) else NSSParams.from_array(p)
    t = np.asarray(tau, dtype=float)
    g1, g2 = _g(t, p.tau1), _g(t, p.tau2)
    e1, e2 = np.exp(-t / p.tau1), np.exp(-t / p.tau2)
    return p.beta0 + p.beta1 * g1 + p.beta2 * (g1 - e1) + p.beta3 * (g2 - e2)


def nss_forward(tau, p: NSSParams | Sequence[float]) -> np.ndarray:
    """Instantaneous forward rate f(tau). The thing NSS is really parameterising."""
    p = p if isinstance(p, NSSParams) else NSSParams.from_array(p)
    t = np.asarray(tau, dtype=float)
    e1, e2 = np.exp(-t / p.tau1), np.exp(-t / p.tau2)
    return (
        p.beta0
        + p.beta1 * e1
        + p.beta2 * (t / p.tau1) * e1
        + p.beta3 * (t / p.tau2) * e2
    )


def nss_discount(tau, p: NSSParams | Sequence[float]) -> np.ndarray:
    """DF(tau) = exp(-y(tau) * tau). Continuous compounding throughout."""
    t = np.asarray(tau, dtype=float)
    return np.exp(-nss_spot(t, p) * t)


def nss_par_yield(
    tau, p: NSSParams | Sequence[float], freq: int = 1
) -> np.ndarray | float:
    """Coupon that makes a bond of maturity `tau` price at par.

        c = (1 - DF(T)) / sum_i DF(t_i) * freq

    Useful as a cross-check: the ECB publishes par yields (`PY_10Y`) alongside
    spot rates, and your fitted parameters should reproduce both.

    COMPOUNDING WARNING: the result is compounded at `freq`, while `nss_spot`
    returns continuously-compounded rates. Comparing the two directly makes a
    flat curve look wrong by a few basis points. Convert first:
    `y_annual = exp(y_cc) - 1`.
    """
    p = p if isinstance(p, NSSParams) else NSSParams.from_array(p)
    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    out = np.empty_like(taus)
    for i, T in enumerate(taus):
        n = max(round(T * freq), 1)
        times = np.arange(1, n + 1) / freq
        dfs = nss_discount(times, p)
        out[i] = freq * (1.0 - dfs[-1]) / dfs.sum()
    return out if np.ndim(tau) else float(out[0])


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _residuals(x, maturities, observed, weights):
    # Penalise parameter sets that would make the curve degenerate. least_squares
    # respects bounds so this rarely triggers, but belt and braces.
    if x[4] <= _EPS or x[5] <= _EPS:
        return np.full_like(observed, 1e6)
    return (nss_spot(maturities, x) - observed) * weights


def fit_nss(
    maturities,
    yields,
    weights=None,
    n_starts: int = 40,
    seed: int = 0,
) -> tuple[NSSParams, dict]:
    """Fit Svensson to observed zero rates by weighted least squares.

    This is 'yield error minimisation' -- the `YM` in the ECB's `SV_C_YM` code.
    The alternative, price error minimisation, weights long bonds far more heavily
    because their prices are more rate-sensitive; the ECB chose yields.

    Why `n_starts`: the objective is **not convex in tau1 and tau2**. The betas
    enter linearly, so for any fixed pair of taus the optimal betas are a plain
    linear regression -- but the taus themselves create a landscape with several
    local minima, and one classic failure mode is tau1 and tau2 collapsing onto
    each other so the beta2 and beta3 terms become collinear and the fit becomes
    unidentified. Multi-start from a grid of tau pairs fixes this cheaply. If you
    only remember one practical fact about NSS, make it this one.

    Parameters
    ----------
    maturities : array of years
    yields     : array of DECIMAL continuously-compounded zero rates
    weights    : optional per-point weights (default equal). A common choice is
                 1/sqrt(maturity), which stops the 20y-30y region from dominating.

    Returns
    -------
    (params, info) where info carries rmse, max abs error, and the residuals.
    """
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    if t.shape != y.shape:
        raise ValueError("maturities and yields must have the same shape")
    if t.size < 6:
        raise ValueError("need at least 6 points to identify 6 parameters")
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)

    rng = np.random.default_rng(seed)
    # Sensible seeds: long rate ~ longest observed yield, short deviation ~ the gap.
    b0_0 = float(y[-1])
    b1_0 = float(y[0] - y[-1])

    tau_grid = [
        (0.5, 3.0),
        (1.0, 5.0),
        (2.0, 8.0),
        (1.5, 12.0),
        (3.0, 20.0),
        (0.8, 6.0),
    ]
    starts = [np.array([b0_0, b1_0, 0.0, 0.0, a, b]) for a, b in tau_grid]
    while len(starts) < n_starts:
        starts.append(
            np.array(
                [
                    b0_0 + rng.normal(0, 0.005),
                    b1_0 + rng.normal(0, 0.005),
                    rng.normal(0, 0.02),
                    rng.normal(0, 0.02),
                    float(rng.uniform(0.2, 5.0)),
                    float(rng.uniform(3.0, 25.0)),
                ]
            )
        )

    lo = np.array([-0.20, -0.50, -0.50, -0.50, 0.05, 0.05])
    hi = np.array([0.20, 0.50, 0.50, 0.50, 30.0, 30.0])

    best, best_cost = None, np.inf
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(
                _residuals,
                x0,
                args=(t, y, w),
                bounds=(lo, hi),
                method="trf",
                max_nfev=8000,
                xtol=1e-14,
                ftol=1e-14,
            )
        except Exception:  # pragma: no cover - solver blow-ups are rare
            continue
        if res.cost < best_cost:
            best, best_cost = res, res.cost

    if best is None:  # pragma: no cover
        raise RuntimeError("NSS fit failed from every starting point")

    x = best.x
    # Cosmetic convention: report the shorter decay as tau1. Swapping tau1/tau2
    # together with beta2/beta3 leaves the curve identical, so this is free.
    if x[4] > x[5]:
        x = np.array([x[0], x[1], x[3], x[2], x[5], x[4]])

    params = NSSParams.from_array(x)
    resid = nss_spot(t, params) - y
    info = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "max_abs_error": float(np.max(np.abs(resid))),
        "residuals": resid,
        "n_starts": len(starts),
        "success": bool(best.success),
    }
    return params, info
