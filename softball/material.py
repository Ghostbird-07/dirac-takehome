"""Material model: what each spring between two particles does.

Every spring is a 'standard linear solid' (also called a Zener model), drawn as

        ┌──── k1 ────┐            k1 : main spring (long-term stiffness)
    ────┤──── c1 ────├────        c1 : damper in parallel (fast viscous loss)
        └─ k2 ── c2 ─┘            k2, c2 : spring + damper in series ("Maxwell arm")

The Maxwell arm is what lets foam 'remember' a long squeeze: while the ball is
held down, the arm's damper slowly gives way, so after release the arm briefly
holds the ball down and it creeps back slowly. A short squeeze leaves the arm
no time to relax, so the ball springs back quickly.

Setting beta = 0 removes the Maxwell arm and leaves the simpler Kelvin-Voigt
model (spring + damper), which recovers the same way however long it was held.

Parameters are written in a form that separates what video can and cannot see:

    E      [Pa] long-term stiffness of the material. Sets the FORCE scale.
                A video of pressing and releasing (displacement only) barely
                constrains it; a known weight resting on the ball does.
    tau_v  [s]  Kelvin-Voigt time c1/k1: how fast the fast part of recovery is.
    beta   [-]  Maxwell-arm stiffness relative to the main spring, k2/k1.
    tau_m  [s]  Maxwell relaxation time c2/k2: how long the 'memory' lasts.

The times and the ratio shape the recovery curve, which is exactly what a
press-and-release video measures.

Per-spring constants are derived from E with k = E * spacing, so a finer
lattice gives (roughly) the same ball. E is therefore a lattice modulus of the
order of the foam's Young's modulus, not an exact continuum value; the
interpretable output is the ball's force-displacement response, which the
simulator reports directly.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Params:
    E: float = 2.0e4     # Pa
    tau_v: float = 0.05  # s
    beta: float = 0.0    # -
    tau_m: float = 1.0   # s

    def spring_constants(self, spacing: float):
        k1 = self.E * spacing
        c1 = k1 * self.tau_v
        k2 = self.beta * k1
        return k1, c1, k2

    def spring_arrays(self, ball):
        """Per-spring (k1, c1, k2): the solid lattice uses the same values everywhere."""
        k1, c1, k2 = self.spring_constants(ball.spacing)
        S = len(ball.springs)
        return np.full(S, k1), np.full(S, c1), np.full(S, k2)

    tau_air = None   # no air model for the solid ball

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        s = f"E={self.E:.4g} Pa, tau_v={self.tau_v:.4g} s"
        if self.beta > 0:
            s += f", beta={self.beta:.4g}, tau_m={self.tau_m:.4g} s"
        return s


def maxwell_alpha(tau_m: float, dt: float) -> float:
    """Fraction of the gap the Maxwell arm's rest length closes in one step (exact exponential)."""
    return float(1.0 - np.exp(-dt / tau_m))
