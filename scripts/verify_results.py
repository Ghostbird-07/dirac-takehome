"""Quick reproducibility check (~2-3 minutes, no videos and no refitting needed).

Takes the fitted parameters saved in outputs/fits/results.json, reruns each model on its clips with
the current code and the included depth curves (data/curves/*.csv), and compares the errors with the
ones reported in the README.

Run:  .venv/Scripts/python scripts/verify_results.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import load_csv            # noqa: E402
from softball.fitting import rmse                 # noqa: E402

# (results.json key, model, pose, clip, which stored number to compare with)
CHECKS = [
    ("lumped/face_press", "lumped", None, "A1", "fit"),
    ("lumped/face_press", "lumped", None, "A2", "test"),
    ("shell/face_press", "sphere", "lying", "A1", "fit"),
    ("shell/face_press", "sphere", "lying", "A2", "test"),
    ("duck/face_press", "duck", "lying", "A1", "fit"),
    ("duck/face_press", "duck", "lying", "A2", "test"),
    ("duck/head_press_transfer", "duck", "upright", "B1", "test"),
]


def build(kind, pose):
    if kind == "lumped":
        from softball.lumped import LumpedSLS
        return LumpedSLS()
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_shell import build_model
    return build_model(kind, pose)


def run(check):
    key, kind, pose, clip, which = check
    res = json.loads((ROOT / "outputs" / "fits" / "results.json").read_text())[key]
    model = build(kind, pose)
    theta = np.array([res["params"][n] for n in model.names])
    stored = res["rmse_fit_mm"] if which == "fit" else res["test_rmse_mm"][clip]
    now = rmse(model, theta, [load_csv(ROOT / "data" / "curves" / f"{clip}.csv")])
    return key, clip, which, stored, now


def main():
    with ProcessPoolExecutor(min(len(CHECKS), os.cpu_count() or 1)) as ex:
        rows = list(ex.map(run, CHECKS))
    print(f"{'model (results.json key)':28s} {'clip':5s} {'role':5s} {'reported':>9s} {'rerun now':>10s}")
    worst = 0.0
    for key, clip, which, stored, now in rows:
        worst = max(worst, abs(now - stored))
        print(f"{key:28s} {clip:5s} {which:5s} {stored:8.2f}  {now:9.2f}   mm")
    print(f"largest difference: {worst:.3f} mm -> " + ("reproduced" if worst < 0.05 else "NOT reproduced"))


if __name__ == "__main__":
    main()
