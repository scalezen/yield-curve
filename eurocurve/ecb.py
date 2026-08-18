"""Client for the ECB Data Portal (SDMX REST). No API key, no registration.

Endpoint shape:

    https://data-api.ecb.europa.eu/service/data/{DATASET}/{SERIES_KEY}?format=csvdata

The series key is a dot-separated tuple of dimension codes, positional. For the
`YC` (yield curve) dataset the dimensions are:

    FREQ . REF_AREA . CURRENCY . PROVIDER_FM . INSTRUMENT_FM . PROVIDER_FM_ID . DATA_TYPE_FM

so `B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y` reads as:

    B        business-daily
    U2       euro area (changing composition)
    EUR      euro
    4F       ECB is the data provider
    G_N_A    Government bond / Nominal / AAA-rated issuers only  (G_N_C = all issuers)
    SV_C_YM  SVensson model, Continuous compounding, Yield-error Minimisation
    SR_10Y   Spot Rate, 10-year point

Other DATA_TYPE_FM codes worth knowing:
    PY_10Y   par yield          (what a 10y bond issued at par would coupon)
    IF_10Y   instantaneous forward rate at t=10
    BETA0..BETA3, TAU1, TAU2    the ECB's own fitted Svensson parameters

That last group is the gift of this project: you can fit the curve yourself and
then compare against the official parameters for the same day.

IMPORTANT: the API returns rates in PERCENT. Every function here divides by 100
before handing anything back, so the rest of the package only ever sees decimals.
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Iterable, Sequence

import pandas as pd
import requests

BASE = "https://data-api.ecb.europa.eu/service/data"

#: The maturity ladder the ECB publishes. 3M-30Y, in years.
DEFAULT_TENORS: tuple[str, ...] = (
    "3M", "6M", "9M",
    "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
    "12Y", "15Y", "20Y", "25Y", "30Y",
)

#: `G_N_A` = AAA-rated issuers only (the "risk-free" euro curve).
#: `G_N_C` = all euro area issuers (includes Italy, Greece -- credit risk in it).
RATING_AAA = "G_N_A"
RATING_ALL = "G_N_C"


def _get(dataset: str, key: str, params: dict) -> pd.DataFrame:
    url = f"{BASE}/{dataset}/{key}"
    q = {"format": "csvdata", "detail": "dataonly", **params}
    print(f"fetching URL: {url}")
    r = requests.get(url, params=q, timeout=60, headers={"Accept": "text/csv"})
    r.raise_for_status()
    if not r.text.strip():
        raise RuntimeError(f"ECB returned no data for {dataset}/{key} with {params}")
    df = pd.read_csv(io.StringIO(r.text))
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    return df


def _series_key(tenor: str, kind: str, rating: str) -> str:
    return f"B.U2.EUR.4F.{rating}.SV_C_YM.{kind}_{tenor}"


def fetch_curve(
    date: str | dt.date | None = None,
    tenors: Sequence[str] = DEFAULT_TENORS,
    kind: str = "SR",
    rating: str = RATING_AAA,
) -> pd.DataFrame:
    """Fetch one day's curve.

    Parameters
    ----------
    date
        A date string 'YYYY-MM-DD'. If None, takes the most recent published day.
        The ECB publishes with a lag and skips TARGET holidays, so asking for a
        specific date can legitimately return nothing -- we fall back to the last
        observation on or before it.
    kind
        'SR' spot rates, 'PY' par yields, 'IF' instantaneous forwards.

    Returns
    -------
    DataFrame with columns [tenor, maturity, rate] where `rate` is a DECIMAL
    continuously-compounded rate, sorted by maturity.
    """
    from .daycount import parse_tenor

    key = _series_key("+".join(tenors), kind, rating)
    # NB: SDMX lets you OR multiple codes in one dimension with '+', so this is
    # a single HTTP round trip for the whole curve rather than 18 of them.
    if date is None:
        params = {"lastNObservations": 1}
    else:
        d = pd.to_datetime(date).date()
        # grab a two-week window ending on the requested date, then take the last day
        params = {"startPeriod": str(d - dt.timedelta(days=14)), "endPeriod": str(d)}

    df = _get("YC", key, params)
    asof = df["TIME_PERIOD"].max()
    df = df[df["TIME_PERIOD"] == asof]

    df["tenor"] = df["DATA_TYPE_FM"].str.split("_").str[1]
    df["maturity"] = df["tenor"].map(parse_tenor)
    df["rate"] = df["OBS_VALUE"] / 100.0
    out = df[["tenor", "maturity", "rate"]].sort_values("maturity").reset_index(drop=True)
    out.attrs["as_of"] = asof.date()
    out.attrs["kind"] = kind
    out.attrs["rating"] = rating
    return out


def fetch_ecb_svensson_params(
    date: str | dt.date | None = None, rating: str = RATING_AAA
) -> dict:
    """The ECB's OWN fitted Svensson parameters for a day.

    We can fit the curve by running `nss.fit_nss`, then compare.
    Note the ECB quotes betas in percent as well -- converted here.

    Returns a dict with keys beta0, beta1, beta2, beta3, tau1, tau2 and 'as_of'.
    Betas are decimals; taus are in years and are NOT scaled.
    """
    codes = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    frames = []
    for c in codes:
        key = f"B.U2.EUR.4F.{rating}.SV_C_YM.{c}"
        if date is None:
            params = {"lastNObservations": 1}
        else:
            d = pd.to_datetime(date).date()
            params = {"startPeriod": str(d - dt.timedelta(days=14)), "endPeriod": str(d)}
        f = _get("YC", key, params)
        f = f[f["TIME_PERIOD"] == f["TIME_PERIOD"].max()]
        frames.append(f[["DATA_TYPE_FM", "TIME_PERIOD", "OBS_VALUE"]])

    df = pd.concat(frames)
    asof = df["TIME_PERIOD"].max().date()
    raw = dict(zip(df["DATA_TYPE_FM"], df["OBS_VALUE"]))
    return {
        "as_of": asof,
        "beta0": raw["BETA0"] / 100.0,
        "beta1": raw["BETA1"] / 100.0,
        "beta2": raw["BETA2"] / 100.0,
        "beta3": raw["BETA3"] / 100.0,
        "tau1": raw["TAU1"],
        "tau2": raw["TAU2"],
    }

def fetch_history(
    tenor: str = "10Y",
    start: str = "2015-01-01",
    kind: str = "SR",
    rating: str = RATING_AAA,
) -> pd.Series:
    """Time series of one point on the curve. Handy for 'is today weird?' checks."""
    key = _series_key(tenor, kind, rating)
    df = _get("YC", key, {"startPeriod": start})
    s = pd.Series(df["OBS_VALUE"].values / 100.0, index=df["TIME_PERIOD"], name=f"{kind}_{tenor}")
    return s.sort_index()


def fetch_estr(date: str | dt.date | None = None) -> tuple[dt.date, float]:
    """The euro short-term rate, as a decimal. The overnight anchor of Stage 2.

    Series `EST.B.EU000A2X2A25.WT` -- 'WT' is the volume-weighted trimmed mean,
    which is the published €STR fixing itself.
    """
    if date is None:
        params = {"lastNObservations": 1}
    else:
        d = pd.to_datetime(date).date()
        params = {"startPeriod": str(d - dt.timedelta(days=14)), "endPeriod": str(d)}
    df = _get("EST", "B.EU000A2X2A25.WT", params)
    row = df.sort_values("TIME_PERIOD").iloc[-1]
    return row["TIME_PERIOD"].date(), float(row["OBS_VALUE"]) / 100.0



if __name__ == "__main__":
    # runner to see if functions work

    result = fetch_history()
    print(result)