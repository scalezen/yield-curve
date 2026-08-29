import numpy as np
import pytest

from eurocurve.nss import (
    NSSParams,
    fit_nss,
    nss_discount,
    nss_forward,
    nss_par_yield,
    nss_spot,
)

P = NSSParams(beta0=0.028, beta1=-0.010, beta2=0.015, beta3=-0.008, tau1=1.6, tau2=9.0)
GRID = np.array(
    [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30], dtype=float
)


def test_spot_at_zero_is_beta0_plus_beta1():
    """y(0) = beta0 + beta1, because g(0)=1 and both hump terms vanish."""
    assert nss_spot(0.0, P) == pytest.approx(P.beta0 + P.beta1, abs=1e-12)
    assert nss_spot(1e-9, P) == pytest.approx(P.short_rate, abs=1e-9)


def test_forward_at_zero_equals_spot_at_zero():
    """f(0) must equal y(0): the average of a function over an infinitesimal
    interval is the function."""
    assert nss_forward(0.0, P) == pytest.approx(nss_spot(0.0, P), abs=1e-12)


def test_long_end_tends_to_beta0():
    """The forward rate converges to beta0 exponentially fast (in tau1, tau2), so
    tau=500 already nails it to machine precision. The spot rate is the AVERAGE
    of the forward from 0 to tau, and the beta3 hump's tail only decays like
    tau2/tau -- a slow O(1/tau) harmonic tail, not exponential. At tau=500 that
    residual is still ~1.3e-4, wider than the tolerance below; tau=50_000 gives
    it three more orders of magnitude of room.
    """
    assert nss_spot(50_000.0, P) == pytest.approx(P.beta0, abs=1e-4)
    assert nss_forward(500.0, P) == pytest.approx(P.beta0, abs=1e-6)


def test_no_nan_at_zero():
    """The (1-exp(-x))/x term is 0/0 at tau=0. If someone 'simplifies' the safe
    branch out of _g, this fails."""
    vals = nss_spot(np.array([0.0, 1e-12, 1e-8, 1e-4]), P)
    assert np.all(np.isfinite(vals))


def test_spot_is_average_of_forward():
    """The defining relationship: y(T) = (1/T) * integral_0^T f(u) du.

    Checked by dense trapezoid. If the closed forms for spot and forward ever
    drift apart, this is the test that notices.
    """
    for T in (0.5, 3.0, 10.0, 30.0):
        u = np.linspace(0.0, T, 200_001)
        integral = (
            np.trapezoid(nss_forward(u, P), u)
            if hasattr(np, "trapezoid")
            else np.trapz(nss_forward(u, P), u)
        )
        assert integral / T == pytest.approx(float(nss_spot(T, P)), rel=1e-9)


def test_discount_factor_round_trip():
    df = nss_discount(GRID, P)
    assert np.all(df > 0)
    recovered = -np.log(df) / GRID
    np.testing.assert_allclose(recovered, nss_spot(GRID, P), rtol=1e-12)


def test_discount_factors_are_decreasing_when_rates_positive():
    df = nss_discount(GRID, P)
    assert np.all(np.diff(df) < 0)


def test_par_yield_reprices_a_par_bond():
    """A bond with coupon = par yield must price at exactly 100."""
    T = 10.0
    c = nss_par_yield(T, P, freq=1)
    times = np.arange(1, 11, dtype=float)
    dfs = nss_discount(times, P)
    price = c * dfs.sum() + dfs[-1]
    assert price == pytest.approx(1.0, abs=1e-12)


def test_par_yield_lives_inside_the_spot_curve():
    """A par yield is a discount-factor-weighted blend of the spot curve up to
    maturity, so it must land inside the range of spot rates it averages over.

    Two traps this test encodes:
      * it is NOT simply 'below the spot rate' -- that only holds on a
        monotonically upward-sloping curve, and real curves are not monotone;
      * an annual-pay par yield is ANNUALLY compounded, so it must be compared
        against exp(y_cc) - 1, not against y_cc. On a flat 2.6% cc curve the par
        yield is 2.63%, which looks like a bug and is not.
    """
    py = np.asarray(nss_par_yield(GRID, P, freq=1))
    assert np.all(np.isfinite(py))
    for T, c in zip(GRID, py):
        if T < 1.0:
            continue
        spots_annual = np.exp(np.asarray(nss_spot(np.arange(1.0, T + 0.5), P))) - 1.0
        assert spots_annual.min() - 1e-9 <= c <= spots_annual.max() + 1e-9


def test_fit_recovers_the_curve_it_was_given():
    """Fit synthetic data generated from known parameters.

    We assert on the CURVE, not the parameters, deliberately: NSS is only weakly
    identified, so a different-looking parameter set can produce an essentially
    identical curve. Anyone comparing betas across dates or countries needs to
    internalise this.
    """
    y = np.asarray(nss_spot(GRID, P))
    fitted, info = fit_nss(GRID, y)
    assert info["rmse"] < 1e-6
    dense = np.linspace(0.25, 30, 500)  # inside the observed range; see note below
    diff = np.abs(np.asarray(nss_spot(dense, fitted)) - np.asarray(nss_spot(dense, P)))
    assert diff.max() < 1e-5  # under a tenth of a basis point


def test_fit_is_robust_to_noise():
    rng = np.random.default_rng(7)
    y = np.asarray(nss_spot(GRID, P)) + rng.normal(0, 2e-4, GRID.size)  # 2bp noise
    fitted, info = fit_nss(GRID, y)
    assert info["rmse"] < 5e-4
    dense = np.linspace(0.5, 30, 300)
    diff = np.abs(np.asarray(nss_spot(dense, fitted)) - np.asarray(nss_spot(dense, P)))
    assert diff.max() < 2e-3  # within 20bp everywhere despite the noise


def test_fit_rejects_underdetermined_input():
    with pytest.raises(ValueError):
        fit_nss([1.0, 2.0, 3.0], [0.01, 0.02, 0.03])


def test_tau1_reported_shorter_than_tau2():
    y = np.asarray(nss_spot(GRID, P))
    fitted, _ = fit_nss(GRID, y)
    assert fitted.tau1 <= fitted.tau2


def test_negative_rates_are_fine():
    """The euro curve was negative out to 15 years for years. Nothing in the model
    forbids it, and nothing in the code should either."""
    neg = NSSParams(-0.005, -0.003, 0.01, 0.0, 2.0, 10.0)
    assert float(nss_spot(1.0, neg)) < 0
    assert float(nss_discount(1.0, neg)) > 1.0  # a DF above 1: perfectly legal
