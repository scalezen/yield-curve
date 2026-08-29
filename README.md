# euro-curve

[![CI](https://github.com/scalezen/yield-curve/actions/workflows/ci.yml/badge.svg)](https://github.com/scalezen/yield-curve/actions/workflows/ci.yml)

**A euro yield curve built twice, from opposite ends, and checked against a
published answer.** Once by fitting a smooth parametric form to observed
government bond yields; once by bootstrapping discount factors out of OIS swap
par rates with no model at all. Both routes end at the same three objects —
discount factors, zero rates, instantaneous forwards — which is what makes them
worth building side by side.

**Stage 2** — bootstrap a €STR OIS discount curve from swap par rates. No model or fitting: pure sequential arithmetic.

---

## Stage 1 — Nelson–Siegel–Svensson fit

Fit a Svensson curve to the ECB's published AAA euro-area government bond yields.

The reason this stage is worth doing rather than reading about: **the ECB
publishes its own fitted parameters** (`BETA0`…`BETA3`, `TAU1`, `TAU2`) alongside
the rates it derived from them. So the fit has a real answer to be marked against,
not a plausibility check.

That comparison teaches the thing nobody warns you about first: **Svensson is only
weakly identified.** Two visibly different parameter sets can produce curves that
agree to well under a basis point. `tests/test_nss.py` therefore asserts on the
*curve*, never on the betas — a discipline that matters the moment anyone tries to
compare fitted betas across dates or across countries and reads meaning into the
difference.

## Stage 2 — OIS bootstrap

Bootstrap a €STR OIS discount curve from swap par rates. No fitting, no
optimiser over a shape: sequential arithmetic, one pillar at a time, each solved
so that the swap that defined it prices to exactly par.

Which gives the strongest test in the repo:

```
test_bootstrap_reprices_every_input_exactly
    max |error| < 1e-6 bp        -- to machine precision, not approximately
test_par_swaps_have_zero_pv
    |PV| < 0.1 cent on EUR 100m
test_one_year_matches_the_closed_form
    DF(T) = 1 / (1 + S·alpha)    -- root-finder vs pencil and paper
```

A bootstrapped curve must return its own inputs. One assertion catches day-count
errors, schedule errors, and percent-versus-decimal errors at once.

---

## Quick start

```bash
conda env create -f environment.yaml -n yield_curve
pytest -q                       # whole suite, offline, seconds
python -m scripts.run_stage1    # pulls ECB data, fits NSS, saves a plot
python -m scripts.run_stage2    # bootstraps the OIS curve, saves a plot
jupyter lab notebooks/          # the actual teaching material
```

Notebooks are in order. The scripts are the notebooks' conclusions in runnable form.

## Layout

```
eurocurve/
  daycount.py    year fractions, roll conventions, schedule generation
  ecb.py         ECB Data Portal client (no API key needed)
  nss.py         Nelson-Siegel-Svensson: spot, forward, discount, par yield, fitting
  curve.py       DiscountCurve -- interpolation, zero rates, forwards
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

**ECB Data Portal** (`data-api.ecb.europa.eu`) is a public SDMX API. No key, no registration and
no rate limit. Two datasets matter here:

- `YC` — euro area yield curves. The ECB fits Svensson daily to AAA-rated euro area
  government bonds and publishes both the *outputs* (spot rates `SR_1Y`, par yields
  `PY_1Y`, instantaneous forwards `IF_1Y`) and the *fitted parameters*
  (`BETA0`…`BETA3`, `TAU1`, `TAU2`).
- `EST` — €STR, the euro short-term rate. The overnight anchor of the OIS curve.

Series keys look like `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y`. Decoded:
`B`=business daily, `U2`=euro area, `4F`=ECB as provider, `G_N_A`=government/nominal/AAA-only
(`G_N_C` is all issuers), `SV_C_YM`=Svensson, continuously compounded, fitted by yield-error
minimisation, `SR_10Y`=10-year spot rate.

**OIS quotes** are not free. Live EUR OIS par rates are available from
Bloomberg/Refinitiv/ICAP. So `data/eur_ois_sample.csv` is a static, plausible EUR OIS curve
bundled with the repo — it makes Stage 2 fully reproducible offline, and the bootstrap
mechanics are identical independent of the source input dataset - real or synthetic quotes. 
The O/N pillar is pulled live from €STR so at least one point is real.

Values in ECB `YC` are **percent** and **continuously compounded**. Getting units
wrong leads to catastrophic bugs, so `ecb.py` converts to decimals at
the boundary and everything inside the package is in decimals.

