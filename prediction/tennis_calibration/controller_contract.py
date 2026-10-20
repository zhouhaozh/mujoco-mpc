"""Shared deployment-domain rules, used identically by scoring and service."""
import numpy as np

def residual_domain_gain(horizons, trained_max=.7):
    """Fade over the last 200 ms of the observed training domain, then zero.

    Existing data contain post-recovery residual labels only through H700
    (sparse at H700). This gate prevents a short-range head being extrapolated
    across the controller's full 1.5s physics horizon.
    """
    if not np.isfinite(trained_max) or trained_max <= 0:
        raise ValueError('positive finite residual training limit required')
    h=np.asarray(horizons,float)
    if not np.isfinite(h).all() or np.any(h<0):
        raise ValueError('invalid horizon')
    return np.clip((trained_max-h)/min(.2,trained_max),0.,1.)
