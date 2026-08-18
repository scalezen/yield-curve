import datetime as dt

import pytest

from eurocurve import ecb


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def _stub_get(monkeypatch, csv_text: str):
    """Replace requests.get with a stub that returns `csv_text` and records calls."""
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append({"url": url, "params": params})
        return _FakeResponse(csv_text)

    monkeypatch.setattr(ecb.requests, "get", fake_get)
    return calls


# -- fetch_curve --------------------------------------------------------------

CURVE_CSV = """TIME_PERIOD,DATA_TYPE_FM,OBS_VALUE
2026-07-31,SR_3M,2.10
2026-08-14,SR_10Y,2.55
2026-08-14,SR_3M,2.05
2026-08-14,SR_6M,2.10
"""


def test_fetch_curve_sorts_by_maturity_and_converts_to_decimal(monkeypatch):
    _stub_get(monkeypatch, CURVE_CSV)
    out = ecb.fetch_curve(date="2026-08-14", tenors=("3M", "6M", "10Y"))

    assert list(out["tenor"]) == ["3M", "6M", "10Y"]
    assert list(out["maturity"]) == pytest.approx([0.25, 0.5, 10.0])
    assert list(out["rate"]) == pytest.approx([0.0205, 0.0210, 0.0255])


def test_fetch_curve_only_keeps_the_latest_observation_date(monkeypatch):
    """The 2-week lookback window can return more than one day; only the most
    recent day's rows should survive into the result."""
    _stub_get(monkeypatch, CURVE_CSV)
    out = ecb.fetch_curve(date="2026-08-14")

    assert len(out) == 3  # the 2026-07-31 row must be dropped
    assert out.attrs["as_of"] == dt.date(2026, 8, 14)


def test_fetch_curve_attrs_record_kind_and_rating(monkeypatch):
    _stub_get(monkeypatch, CURVE_CSV)
    out = ecb.fetch_curve(date="2026-08-14", kind="SR", rating=ecb.RATING_ALL)

    assert out.attrs["kind"] == "SR"
    assert out.attrs["rating"] == ecb.RATING_ALL


def test_fetch_curve_default_date_requests_last_observation(monkeypatch):
    calls = _stub_get(monkeypatch, CURVE_CSV)
    ecb.fetch_curve(date=None)

    assert calls[0]["params"]["lastNObservations"] == 1


def test_fetch_curve_explicit_date_requests_two_week_window(monkeypatch):
    calls = _stub_get(monkeypatch, CURVE_CSV)
    ecb.fetch_curve(date="2026-08-14")

    params = calls[0]["params"]
    assert params["startPeriod"] == "2026-07-31"
    assert params["endPeriod"] == "2026-08-14"


# -- fetch_history --------------------------------------------------------------

HISTORY_CSV = """TIME_PERIOD,OBS_VALUE
2026-08-14,2.50
2026-08-01,2.10
2026-08-07,2.30
"""


def test_fetch_history_sorts_chronologically_and_converts_to_decimal(monkeypatch):
    _stub_get(monkeypatch, HISTORY_CSV)
    s = ecb.fetch_history(tenor="10Y", start="2026-08-01")

    assert list(s.index) == sorted(s.index)
    assert list(s.values) == pytest.approx([0.0210, 0.0230, 0.0250])


def test_fetch_history_series_is_named_by_kind_and_tenor(monkeypatch):
    _stub_get(monkeypatch, HISTORY_CSV)
    s = ecb.fetch_history(tenor="10Y", kind="SR")

    assert s.name == "SR_10Y"


# -- fetch_estr --------------------------------------------------------------

ESTR_CSV = """TIME_PERIOD,OBS_VALUE
2026-08-14,3.65
2026-08-01,3.60
2026-08-07,3.62
"""


def test_fetch_estr_picks_the_latest_row_even_if_csv_is_unordered(monkeypatch):
    _stub_get(monkeypatch, ESTR_CSV)
    as_of, rate = ecb.fetch_estr(date="2026-08-14")

    assert as_of == dt.date(2026, 8, 14)
    assert rate == pytest.approx(0.0365)


def test_fetch_estr_default_date_requests_last_observation(monkeypatch):
    calls = _stub_get(monkeypatch, ESTR_CSV)
    ecb.fetch_estr(date=None)

    assert calls[0]["params"]["lastNObservations"] == 1
