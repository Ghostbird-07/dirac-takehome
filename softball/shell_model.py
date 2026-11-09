"""The hollow-shell-with-air model wrapped for the fitting pipeline (fitting.py).

Input : the measured presser path of a clip (nozzle-tip depth over time).
Output: the predicted depth of the top of the head (the point under the nozzle),
        to compare with the measured free recovery.

Geometry of the face press (clips A1, A2): the duck lies on its back; its head is
modelled as a sphere of 55 mm diameter with the nozzle (flat end, ~21 mm across,
measured from the video) pushing straight down on the top. The upright head press
(clips B*) bends the head sideways on its neck while a hand holds the body; a
sphere on a table cannot represent that, so the shell model is only used for
the face press.
"""
import numpy as np

from .presser import Presser
from .shell import ShellParams, make_shell
from .sim import simulate


class ShellModel:
    name = "hollow shell + air (3D)"
    names = ["Y", "kappa", "tau_v", "tau_air"]
    x0 = np.array([1000.0, 1.0, 0.01, 0.2])
    lower = np.array([100.0, 0.05, 1e-3, 0.01])
    upper = np.array([2e4, 20.0, 0.2, 3.0])
    interactions = ("face_press",)

    def __init__(self, level=2, radius=0.0275, nozzle_radius=0.0105, mass=0.015, dt=2e-3, t_pre=0.3,
                 ball=None, name=None):
        self.shell = ball if ball is not None else make_shell(radius=radius, level=level, mass=mass)
        if name:
            self.name = name
        self.nozzle_radius = nozzle_radius
        self.dt = dt
        self.t_pre = t_pre
        self.last = None          # last SimResult, for plots and videos
        self.floor = 0.004        # m, lowest the presser may go (two rubber layers above the table)
        sc = getattr(self.shell, "self_contact", None)
        if sc:
            # with self-contact the face cannot come closer than sc to the back wall, so the
            # presser (which holds the face points at its bottom minus their radius) must stop
            # that far above the highest back-wall point under it
            X = self.shell.rest_pos
            under = (np.hypot(X[:, 0], X[:, 1]) <= nozzle_radius) & (X[:, 2] < 0.5 * X[:, 2].max())
            if under.any():
                self.floor = max(self.floor, X[under, 2].max() + sc + self.shell.particle_radius)

    def presser_for(self, curve):
        """Presser keyframes from the measured path. A short ramp is added in front so
        that a clip which starts already pressed is reached smoothly."""
        top0 = self.shell.rest_top
        p = curve.presser_mm
        ok = np.isfinite(p)
        t = curve.t[ok] + self.t_pre
        z = top0 - p[ok] / 1000.0
        # after the presser has left the frame, keep moving it up and away
        z_away = top0 + 0.05
        key_t = np.concatenate([[0.0], t, [t[-1] + 0.05]])
        key_z = np.concatenate([[top0 + 0.003], z, [z_away]])
        # the presser cannot push the surface through the far side of the shell: stop it two
        # rubber layers (~4 mm) above the table (in clip A1 the face was pushed right down
        # onto the back of the head)
        key_z = np.maximum(key_z, self.floor)
        return Presser(self.nozzle_radius, key_t, key_z)

    def simulate_curve(self, curve, theta, record_frames=False, fps=100.0):
        Y, kappa, tau_v, tau_air = theta
        pr = self.presser_for(curve)
        r = simulate(self.shell, ShellParams(Y, kappa, tau_v, tau_air), pr, curve.t[-1] + self.t_pre,
                     fps=fps, dt=self.dt, record_frames=record_frames, presser_from_settled=True)
        self.last = r
        return r

    def predict(self, curve, theta):
        r = self.simulate_curve(curve, theta)
        rest = r.top[0]
        depth_mm = (rest - r.top) * 1000.0
        return np.interp(curve.t, r.t - self.t_pre, depth_mm)
