"""Accuracy check for the time step: rerun the fitted models with half the time step and
compare the predicted free recovery. If the curves barely move, the step is small enough.

Run:  .venv/Scripts/python scripts/check_convergence.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import load_csv          # noqa: E402
from softball.shell_model import ShellModel     # noqa: E402
from softball.duck import make_duck, DUCK_V2    # noqa: E402


def compare(name, make, theta, dts, clip="A2"):
    c = load_csv(ROOT / "data" / "curves" / f"{clip}.csv")
    m = np.isfinite(c.surface_mm)
    preds = {}
    for dt in dts:
        preds[dt] = make(dt).predict(c, theta)
    ref = preds[dts[-1]]
    for dt in dts:
        d = np.sqrt(np.mean((preds[dt][m] - ref[m]) ** 2))
        print(f"{name}, clip {clip}: dt = {dt * 1000:.1f} ms -> RMS difference from dt = {dts[-1] * 1000:.1f} ms over the "
              f"free recovery: {d:.3f} mm")


def main():
    res = json.loads((ROOT / "outputs" / "fits" / "results.json").read_text())
    if "shell/face_press" in res:
        p = res["shell/face_press"]["params"]
        th = np.array([p[n] for n in ShellModel.names])
        compare("sphere shell", lambda dt: ShellModel(dt=dt), th, [2e-3, 1e-3])
    if "duck/face_press" in res:
        p = res["duck/face_press"]["params"]
        th = np.array([p[n] for n in ShellModel.names])
        compare("duck shell", lambda dt: ShellModel(ball=make_duck("lying", **DUCK_V2), dt=dt), th, [3e-3, 1.5e-3])


if __name__ == "__main__":
    main()
