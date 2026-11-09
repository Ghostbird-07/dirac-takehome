"""Step 2 of the pipeline: fit a model's parameters to the measured curves, then
predict held-out clips.

What is compared
----------------
Each clip gives the PRESSER path (how far the nozzle/bottle pushed in, over time)
and the duck's FREE surface after the presser has left. The presser path is the
model's INPUT; the free recovery is what the model must PREDICT. During contact
the surface simply sits under the presser, so it carries no information about the
material: only the free recovery is scored.

    score = root-mean-square error (mm) between predicted and measured surface
            depth over all free frames of the clips being fitted

Any model plugs in by providing:
    names           list of parameter names
    x0, lower, upper  starting guess and bounds (positive values; fitted in log space)
    predict(curve, theta) -> predicted surface depth (mm) at every time in curve.t
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .tracking import Curve


def presser_path(curve: Curve):
    """Presser depth as a function of time; -inf (no contact possible) once it has left."""
    p = curve.presser_mm.copy()
    known = np.isfinite(p)
    last = np.nonzero(known)[0].max() if known.any() else -1
    p[~known] = np.nan
    # gaps while the presser is still around: interpolate; after it has gone: -inf
    idx = np.arange(len(p))
    inside = idx <= last
    if known.sum() >= 2:
        p[inside] = np.interp(curve.t[inside], curve.t[known], p[known])
    p[~inside] = -np.inf
    return p


def free_mask(curve: Curve):
    return np.isfinite(curve.surface_mm)


def residuals(model, theta, curves):
    out = []
    for c in curves:
        pred = model.predict(c, theta)
        m = free_mask(c)
        out.append(pred[m] - c.surface_mm[m])
    return np.concatenate(out)


def rmse(model, theta, curves):
    r = residuals(model, theta, curves)
    return float(np.sqrt(np.mean(r ** 2)))


@dataclass
class FitResult:
    model_name: str
    names: list
    theta: np.ndarray
    rmse_fit: float
    starts: list = field(default_factory=list)

    def as_dict(self):
        return dict(model=self.model_name, rmse_fit_mm=self.rmse_fit,
                    params={n: float(v) for n, v in zip(self.names, self.theta)})


def fit(model, curves, n_starts=8, seed=0, verbose=False) -> FitResult:
    """Least-squares fit in log-parameter space from several random starts (the error
    surface can have more than one valley); the best start wins."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log(model.lower), np.log(model.upper)
    starts = [np.log(model.x0)] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]
    best = None
    tried = []
    for s in starts:
        f = lambda z: residuals(model, np.exp(z), curves)
        r = least_squares(f, s, bounds=(lo, hi), diff_step=1e-2, max_nfev=200 * len(s))
        e = float(np.sqrt(np.mean(r.fun ** 2)))
        tried.append((e, np.exp(r.x)))
        if verbose:
            print("  start -> rmse %.3f mm" % e, dict(zip(model.names, np.round(np.exp(r.x), 4))))
        if best is None or e < best[0]:
            best = (e, np.exp(r.x))
    return FitResult(model.name, list(model.names), best[1], best[0], tried)
