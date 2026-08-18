import numpy as np
import pytest

from eurocurve.curve import DiscountCurve

TIMES = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
ZEROS = np.array([0.019, 0.020, 0.022, 0.024, 0.026, 0.027])

@pytest.fixture
def curve():
    return DiscountCurve.from_zero_rates(TIMES, ZEROS)


def test_df_at_zero_is_one(curve):
    assert curve.df(0.0) == pytest.approx(1.0)


def test_zero_round_trip(curve):
    """Pillars must come back exactly -- interpolation only fills the gaps."""
    np.testing.assert_allclose(curve.zero(TIMES), ZEROS, rtol=1e-12)


def test_compounding_conversions_agree(curve):
    """Same curve, three dialects. exp(y_cc) - 1 = y_annual."""
    t = 7.0
    cc = curve.zero(t, "continuous")
    ann = curve.zero(t, "annual")
    simple = curve.zero(t, "simple")
    assert np.exp(cc) - 1 == pytest.approx(ann, rel=1e-12)
    assert (1 + ann) ** t - 1 == pytest.approx(simple * t, rel=1e-12)
    # ordering is a useful sanity heuristic when rates are positive
    assert cc < ann < simple


def test_log_linear_means_piecewise_constant_forwards(curve):
    """This is the defining property of log-linear interpolation, and the reason
    to prefer it. Between two pillars the instantaneous forward is flat."""
    inside = np.array([2.5, 3.0, 3.5, 4.0, 4.5])  # strictly between the 2y and 5y pillars
    f = curve.inst_forward(inside)
    assert np.ptp(f) < 1e-8


def test_forwards_jump_at_pillars(curve):
    """...and they do jump AT the pillars. That is the price you pay. If your
    application cares about smooth forwards, fit a model instead (Stage 1)."""
    before = curve.inst_forward(4.9)
    after = curve.inst_forward(5.1)
    assert abs(after - before) > 1e-6


def test_forward_composition(curve):
    """DF(0,t2) = DF(0,t1) * DF(t1,t2). No arbitrage, expressed as arithmetic."""
    t1, t2 = 2.0, 5.0
    f = curve.forward(t1, t2, simple=False)
    assert curve.df(t1) * np.exp(-f * (t2 - t1)) == pytest.approx(curve.df(t2), rel=1e-12)


def test_simple_and_continuous_forwards_are_consistent(curve):
    t1, t2 = 1.0, 2.0
    fs = curve.forward(t1, t2, simple=True)
    fc = curve.forward(t1, t2, simple=False)
    assert 1 + fs * (t2 - t1) == pytest.approx(np.exp(fc * (t2 - t1)), rel=1e-12)


def test_extrapolation_is_flat_forward(curve):
    """Beyond the last pillar we continue the last log-linear slope, so the
    forward stays constant instead of doing something creative."""
    f_end = curve.inst_forward(29.0)
    f_beyond = curve.inst_forward(40.0)
    assert f_beyond == pytest.approx(f_end, rel=1e-6)


def test_rejects_nonsense_input():
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 2.0], [0.98, -0.1])
    with pytest.raises(ValueError):
        DiscountCurve([2.0, 2.0], [0.98, 0.96])


def test_from_nss_matches_the_analytic_curve():
    from eurocurve.nss import NSSParams, nss_spot

    p = NSSParams(0.027, -0.009, 0.012, -0.006, 1.8, 8.0)
    c = DiscountCurve.from_nss(p)
    probe = np.array([1.0, 3.0, 7.5, 12.0, 25.0])
    np.testing.assert_allclose(c.zero(probe), np.asarray(nss_spot(probe, p)), atol=2e-6)


def test_scalar_in_scalar_out(curve):
    assert isinstance(curve.df(1.0), float)
    assert isinstance(curve.zero(1.0), float)
    assert isinstance(curve.df(np.array([1.0, 2.0])), np.ndarray)
