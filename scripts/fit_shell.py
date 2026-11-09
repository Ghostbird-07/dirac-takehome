"""Fit the hollow-shell-with-air model to the face-press clips, then predict the
held-out clip.

The error surface is not smooth (the shell can snap between buckled and
un-buckled shapes), so slope-based least squares is unreliable. Instead:
  1. a coarse grid over the four parameters (log-spaced), run in parallel;
  2. Nelder-Mead (a slope-free simplex search) in log space from the best few grid
     points, also in parallel; the best result wins.

Run:  .venv/Scripts/python scripts/fit_shell.py [--quick]
Writes outputs/fits/shell_face_press.png, outputs/fits/shell_search.json and
updates outputs/fits/results.json.
"""
import os
# one maths-library thread per worker process: many processes x many threads only fight for the CPU
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import load_csv          # noqa: E402
from softball.fitting import rmse, FitResult    # noqa: E402
from softball.shell_model import ShellModel     # noqa: E402
from softball.duck import make_duck, DUCK_V2    # noqa: E402

_MODEL = None
_CURVES = None


def build_model(kind, pose="lying"):
    """'sphere': the head as a 55 mm sphere (fast). 'duck': the duck-shaped shell."""
    if kind == "sphere":
        return ShellModel()
    m = ShellModel(ball=make_duck(pose, **DUCK_V2), dt=3e-3,
                   nozzle_radius=0.0105 if pose == "lying" else 0.022,
                   name=f"duck-shaped shell + air (3D, {pose})")
    m.upper = np.array([2e5, 20.0, 0.5, 3.0])
    m.lower = np.array([300.0, 0.05, 1e-3, 2e-3])
    return m


def _init(fit_ids, kind):
    global _MODEL, _CURVES
    _MODEL = build_model(kind)
    _CURVES = [load_csv(ROOT / "data" / "curves" / f"{i}.csv") for i in fit_ids]


def _score(theta):
    try:
        return rmse(_MODEL, np.asarray(theta), _CURVES)
    except Exception:              # a blown-up simulation counts as a very bad fit
        return 1e3


def _nelder_mead(args):
    z0, maxfev = args
    lo, hi = np.log(_MODEL.lower), np.log(_MODEL.upper)
    f = lambda z: _score(np.exp(np.clip(z, lo, hi))) + 10.0 * np.sum(np.maximum(0, z - hi) + np.maximum(0, lo - z))
    r = minimize(f, z0, method="Nelder-Mead",
                 options=dict(maxfev=maxfev, xatol=0.02, fatol=0.01, initial_simplex=None))
    return float(r.fun), np.exp(np.clip(r.x, lo, hi)).tolist(), int(r.nfev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid and search, for testing")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--duck", action="store_true", help="use the duck-shaped shell instead of the sphere")
    ap.add_argument("--joint", action="store_true", help="fit on ALL face-press clips (no held-out lying clip)")
    ap.add_argument("--starts", help="JSON file with a grid already run ({'grid': [{theta, rmse}, ...]}): "
                                     "skip the grid and refine from its best points")
    ap.add_argument("--maxfev", type=int, help="runs per Nelder-Mead start")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "data" / "clips.json").read_text())
    fit_ids = [c["id"] for c in cfg["clips"] if c["interaction"] == "face_press" and c["role"] == "fit"]
    test_ids = [c["id"] for c in cfg["clips"] if c["interaction"] == "face_press" and c["role"] == "test"]
    if args.joint:
        fit_ids, test_ids = fit_ids + test_ids, []
    kind = "duck" if args.duck else "sphere"
    if args.duck:
        # duck v2 (even mesh, self-contact, mesh-independent bending); grid centred on the
        # v1 best values (Y 3.5e4, kappa 10, tau_v 0.05, tau_air 0.05), which are also tried
        grid = dict(Y=[1e4, 3e4, 1e5], kappa=[2.0, 5.0, 12.0], tau_v=[0.02, 0.06], tau_air=[0.02, 0.06, 0.2])
        n_best, maxfev = 3, 30
    elif args.quick:
        grid = dict(Y=[500, 2000], kappa=[0.3, 3], tau_v=[0.01], tau_air=[0.03, 0.1, 0.3])
        n_best, maxfev = 2, 30
    if not args.duck and not args.quick:
        grid = dict(Y=[300, 1000, 3000, 10000], kappa=[0.1, 0.5, 2, 8], tau_v=[0.003, 0.02], tau_air=[0.02, 0.06, 0.2, 0.6])
        n_best, maxfev = 4, 120
    pts = list(itertools.product(grid["Y"], grid["kappa"], grid["tau_v"], grid["tau_air"]))
    if args.duck:
        pts.append((3.468e4, 10.33, 0.0498, 0.0479))       # v1 best
    if args.maxfev:
        maxfev = args.maxfev
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(fit_ids, kind)) as ex:
        if args.starts:
            g = json.loads(Path(args.starts).read_text())["grid"]
            pts = [tuple(e["theta"]) for e in g]
            scores = [e["rmse"] for e in g]
        else:
            scores = list(ex.map(_score, pts, chunksize=1))
        order = np.argsort(scores)
        print(f"grid: {len(pts)} runs in {time.time() - t0:.0f} s; best:", flush=True)
        for k in order[:6]:
            print(f"   rmse {scores[k]:.2f} mm  Y={pts[k][0]} kappa={pts[k][1]} tau_v={pts[k][2]} tau_air={pts[k][3]}", flush=True)
        starts = [(np.log(pts[k]), maxfev) for k in order[:n_best]]
        t1 = time.time()
        nm = list(ex.map(_nelder_mead, starts))
    nm.sort(key=lambda r: r[0])
    print(f"Nelder-Mead from {n_best} starts in {time.time() - t1:.0f} s:")
    for e, th, nf in nm:
        print(f"   rmse {e:.2f} mm after {nf} runs: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(ShellModel.names, th)))
    best_e, best_th, _ = nm[0]
    model = build_model(kind)
    curves = {i: load_csv(ROOT / "data" / "curves" / f"{i}.csv") for i in fit_ids + test_ids}
    res = FitResult(model.name, list(model.names), np.array(best_th), best_e)
    out_dir = ROOT / "outputs" / "fits"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{kind}{'_joint' if args.joint else ''}_search.json").write_text(json.dumps(dict(
        grid=[dict(theta=list(p), rmse=s) for p, s in zip(pts, scores)],
        nelder_mead=[dict(rmse=e, theta=th, runs=nf) for e, th, nf in nm]), indent=2))
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_and_predict import plot                      # same figure style as the baseline
    groups = dict(fit=[curves[i] for i in fit_ids], test=[curves[i] for i in test_ids])
    plot(model, res, groups, out_dir / f"{'shell' if kind == 'sphere' else 'duck'}_face_press{'_joint' if args.joint else ''}.png")
    entry = res.as_dict()
    entry["fit_clips"] = fit_ids
    entry["test_rmse_mm"] = {i: rmse(model, res.theta, [curves[i]]) for i in test_ids}
    entry["seconds"] = round(time.time() - t0, 1)
    rp = out_dir / "results.json"
    results = json.loads(rp.read_text()) if rp.exists() else {}
    results[f"{'shell' if kind == 'sphere' else 'duck'}/face_press" + ("_joint" if args.joint else "")] = entry
    if args.duck:
        # same duck, same fitted values, different pose and press: predict the upright clips
        up = build_model("duck", "upright")
        b_ids = [c["id"] for c in cfg["clips"] if c["interaction"] == "head_press" and c["role"] in ("fit", "test")]
        tag = "_joint" if args.joint else ""
        bc = [load_csv(ROOT / "data" / "curves" / f"{i}.csv") for i in b_ids]
        plot(up, res, dict(fit=[], test=bc), out_dir / f"duck_head_press_transfer{tag}.png")
        results[f"duck/head_press_transfer{tag}"] = dict(model=up.name, params=entry["params"], fitted_on=fit_ids,
                                                  test_rmse_mm={i: rmse(up, res.theta, [c]) for i, c in zip(b_ids, bc)})
        print("upright clips predicted with the lying-fit values:", results[f"duck/head_press_transfer{tag}"]["test_rmse_mm"])
    rp.write_text(json.dumps(results, indent=2))
    print("test:", entry["test_rmse_mm"])


if __name__ == "__main__":
    main()
