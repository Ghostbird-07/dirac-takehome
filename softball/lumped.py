"""Baseline: a one-point ("lumped") spring-damper model of the dent.

The whole duck is reduced to ONE number, the dent depth d (mm), obeying the same
standard-linear-solid law as one spring of the 3D model (material.py):

    c * d' = -k1 * d - k2 * (d - dm)        main spring k1, damper c, Maxwell arm k2
    dm'    = (d - dm) / tau_m               the Maxwell arm slowly 'remembers' the dent

Only ratios matter for a displacement-driven press, so the parameters are times:
    tau_v = c / k1   [s]  how fast the dent springs back
    beta  = k2 / k1  [-]  strength of the slow 'memory' part
    tau_m            [s]  how long that memory lasts

Contact: the surface cannot be above the presser tip, so d >= presser depth while
the presser is there (the presser can push but not pull).

This baseline cannot buckle or snap; it can only relax smoothly. It is here to show
what a simple spring-damper fit captures and what it misses.
"""
import numpy as np

from .fitting import presser_path


class LumpedSLS:
    name = "lumped spring-damper (1 point)"
    names = ["tau_v", "beta", "tau_m"]
    x0 = np.array([0.1, 0.5, 0.5])
    lower = np.array([1e-3, 1e-3, 1e-2])
    upper = np.array([5.0, 20.0, 20.0])

    def __init__(self, dt=1e-3):
        self.dt = dt

    def predict(self, curve, theta):
        tau_v, beta, tau_m = theta
        p_obs = presser_path(curve)
        t_end = curve.t[-1]
        n = int(np.ceil(t_end / self.dt)) + 1
        tt = np.arange(n) * self.dt
        # presser on the fine time grid (step-hold between frames, -inf after it left)
        k = np.clip(np.searchsorted(curve.t, tt, side="right") - 1, 0, len(curve.t) - 1)
        p = p_obs[k]
        d = max(0.0, p[0]) if np.isfinite(p[0]) else 0.0
        dm = 0.0 if np.isfinite(p[0]) and p[0] <= 0 else d   # a clip that starts pressed: assume held a while
        out = np.empty(n)
        a = 1.0 - np.exp(-self.dt / tau_m)
        for i in range(n):
            # implicit Euler for the fast part (stable for any tau_v)
            d = (d + self.dt / tau_v * beta * dm) / (1.0 + self.dt / tau_v * (1.0 + beta))
            if np.isfinite(p[i]) and d < p[i]:
                d = p[i]
            dm += a * (d - dm)
            out[i] = d
        return np.interp(curve.t, tt, out)
