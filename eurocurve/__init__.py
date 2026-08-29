"""A toy euro yield curve library.

Two ways to build a curve:
  * `nss`       -- fit a smooth parametric form to observed government bond yields
  * `bootstrap` -- solve out discount factors sequentially from OIS swap par rates

Both end up producing a `curve.DiscountCurve`, which is the only object the rest of
the world needs to see.

Convention notes, true everywhere inside this package:
  * rates are DECIMALS (0.025), never percent
  * zero rates are CONTINUOUSLY COMPOUNDED unless a function says otherwise
  * time is measured in YEARS as a float
"""

from .curve import DiscountCurve
from .nss import (
    NSSParams,
    nss_spot,
    nss_forward,
    nss_discount,
    fit_nss,
)
from .bootstrap import OISSwap, bootstrap_ois, par_rate

__all__ = [
    "DiscountCurve",
    "NSSParams",
    "nss_spot",
    "nss_forward",
    "nss_discount",
    "fit_nss",
    "OISSwap",
    "bootstrap_ois",
    "par_rate",
]
