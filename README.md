# euro-curve

A toy euro yield curve, built twice, from free data.

**Stage 1** — fit a Nelson–Siegel–Svensson curve to the ECB's published AAA euro area
government bond yields, then check the fitted parameters against the ones the ECB
publishes for the same day.

**Stage 2** — bootstrap a €STR OIS discount curve from swap par rates. No model,
no fitting: pure sequential arithmetic.

Both stages produce the same three objects — discount factors, zero rates, instantaneous
forwards — from completely different starting points.

## Quick start

```bash
conda env create -f environment.yaml -n yield_curve
pytest -q                       # 20-odd tests, all offline
python -m scripts.run_stage1    # pulls ECB data, fits NSS, saves a plot
python -m scripts.run_stage2    # bootstraps the OIS curve, saves a plot
jupyter lab notebooks/          # the actual teaching material
```

Notebooks are in order. The scripts are just the notebooks' conclusions in runnable form.

## Layout

```
eurocurve/
  daycount.py    year fractions, roll, schedule generation
  ecb.py         ECB Data Portal client (no API key needed)
  nss.py         Nelson-Siegel-Svensson: spot, forward, discount, fitting
  curve.py       DiscountCurve — interpolation, zero rates, forwards
  bootstrap.py   OIS swap pricing and sequential bootstrap
data/
  eur_ois_sample.csv   static EUR OIS par rates (see "About the data")
notebooks/
  01_govt_curve_nss.ipynb
  02_ois_bootstrap.ipynb
scripts/
  run_stage1.py, run_stage2.py
tests/
```

## About the data

**ECB Data Portal** (`data-api.ecb.europa.eu`) is a public SDMX API. No key, no registration,
no rate limit worth worrying about. Two datasets matter here:

- `YC` — the euro area yield curves. The ECB fits Svensson daily to AAA-rated euro area
  government bonds and publishes both the *outputs* (spot rates `SR_1Y`, par yields `PY_1Y`,
  instantaneous forwards `IF_1Y`) and the *fitted parameters* (`BETA0`…`BETA3`, `TAU1`, `TAU2`).
- `EST` — €STR, the euro short-term rate. The overnight anchor of the OIS curve.

Series keys look like `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y`. Decoded:
`B`=business daily, `U2`=euro area, `4F`=ECB as provider, `G_N_A`=government/nominal/AAA-only
(`G_N_C` is all issuers), `SV_C_YM`=Svensson, continuously compounded, fitted by yield-error
minimisation, `SR_10Y`=10-year spot rate.

**OIS quotes** are the one thing that is genuinely not free. Live EUR OIS par rates live behind
Bloomberg/Refinitiv/ICAP. So `data/eur_ois_sample.csv` is a static, plausible EUR OIS curve
bundled with the repo — it makes Stage 2 fully reproducible offline, and the bootstrap
mechanics are identical whatever numbers you feed it. Replace that file with real quotes and
everything downstream still works. The O/N pillar is pulled live from €STR so at least one
point is real.

Values in ECB `YC` are **percent** and **continuously compounded**. Getting either of those
wrong leads to catastrophic bugs, so `ecb.py` converts to decimals at
the boundary and everything inside the package is in decimals.

