"""Does the duck-shaped shell still crumple? Runs one clip with fixed parameter values for
several versions of the duck model and reports, for each:
  * run time,
  * folded hinges: neighbouring triangles bent more than 90 degrees away from their rest angle
    (a smooth dent has a few; a crumpled, inside-out head has many),
  * pass-throughs: pairs of points that were far apart at rest (> 6 mm) but end up closer
    than 1 mm, i.e. one wall going through another,
  * the free-recovery error against the video,
and draws the shape at four moments (outputs/fits/duck_collapse_check.png).

Run:  .venv/Scripts/python scripts/check_duck_collapse.py [--clip A2] [--params v1|v2]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import load_csv            # noqa: E402
from softball.shell_model import ShellModel       # noqa: E402
from softball.duck import make_duck, DUCK_V2      # noqa: E402
from softball.bending import hinge_angle_grad     # noqa: E402
from softball.fitting import rmse                 # noqa: E402

VERSIONS = {   # name: make_duck() settings
    "v1: 380 points by ray direction, no self-contact": dict(n_points=380),
    "v2: 900 even points, self-contact, mesh-independent bending": DUCK_V2,
}


def hinge_angles(x, hinges):
    g = np.zeros((4, 3))
    return np.array([hinge_angle_grad(x, h, g) for h in hinges])


def run(args):
    name, clip, theta = args
    ball = make_duck("lying", **VERSIONS[name])
    m = ShellModel(ball=ball, dt=3e-3, name=name)
    cur = load_csv(ROOT / "data" / "curves" / f"{clip}.csv")
    t0 = time.time()
    r = m.simulate_curve(cur, theta, record_frames=True, fps=60.0)
    secs = time.time() - t0
    F = r.frames
    th0 = ball.theta0
    folded = np.array([(np.abs(hinge_angles(x, ball.hinges) - th0) > np.pi / 2).sum() for x in F])
    X0 = ball.rest_pos
    far = np.linalg.norm(X0[:, None] - X0[None], axis=-1) > 0.006
    cross = []
    for x in F:
        d = np.linalg.norm(x[:, None] - x[None], axis=-1)
        cross.append(int(((d < 0.001) & far).sum() // 2))
    err = rmse(m, np.asarray(theta), [cur])
    return dict(name=name, secs=secs, folded=folded.tolist(), cross=cross, rmse=err,
                t=(r.t - m.t_pre).tolist(), frames=F, pz=r.presser_z.tolist(), n=ball.n,
                dp=r.pressure.tolist(), floor=m.floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="A2")
    ap.add_argument("--params", default="v1", help="which fitted duck values to use: v1 (old) or v2")
    a = ap.parse_args()
    if a.params == "v1":
        p = json.loads((ROOT / "outputs" / "fits" / "duck_v1" / "results_duck_v1.json").read_text())["duck/face_press_joint"]["params"]
    else:
        p = json.loads((ROOT / "outputs" / "fits" / "results.json").read_text())["duck/face_press"]["params"]   # v2, fitted on A1
    theta = [p[k] for k in ShellModel.names]
    with ProcessPoolExecutor(len(VERSIONS)) as ex:
        res = list(ex.map(run, [(n, a.clip, theta) for n in VERSIONS]))
    cur = load_csv(ROOT / "data" / "curves" / f"{a.clip}.csv")
    ok = np.isfinite(cur.presser_mm)
    t_deep = cur.t[ok][np.nanargmax(cur.presser_mm)]
    t_rel = cur.t[ok][-1]
    moments = [("before", 0.0), ("deepest press", t_deep), ("nozzle just left", t_rel + 0.05),
               ("0.3 s later", t_rel + 0.35)]
    for r in res:
        print(f"{r['name']:38s} {r['secs']:6.1f} s   folded hinges max {max(r['folded']):4d}   "
              f"pass-throughs max {max(r['cross']):4d}   free-recovery error {r['rmse']:.2f} mm   "
              f"air pressure {min(r['dp']):+.0f}..{max(r['dp']):+.0f} Pa   presser floor {r['floor'] * 1000:.1f} mm")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_demo_video import shell_axes
    fig = plt.figure(figsize=(16, 4.2 * len(res)))
    for i, r in enumerate(res):
        ball = make_duck("lying", **VERSIONS[r["name"]])
        t = np.asarray(r["t"])
        for j, (lab, tm) in enumerate(moments):
            k = int(np.argmin(np.abs(t - tm)))
            ax = fig.add_subplot(len(res), 4, 4 * i + j + 1, projection="3d")
            shell_axes(ax, r["frames"][k], ball, ball.rest_pos, r["pz"][k], 0.0105)
            ax.set_title(f"{lab} (t = {t[k]:.2f} s)\nfolded hinges {r['folded'][k]}, pass-throughs {r['cross'][k]}",
                         fontsize=9)
        fig.text(0.01, 1 - (i + 0.5) / len(res), r["name"] + f"\nerror {r['rmse']:.2f} mm\n{r['secs']:.0f} s per run",
                 fontsize=10, va="center", weight="bold")
    fig.suptitle(f"Clip {a.clip} with the {a.params} fitted values: does the head crumple?", fontsize=13)
    fig.tight_layout(rect=(0.1, 0, 1, 0.97))
    out = ROOT / "outputs" / "fits" / f"duck_collapse_check_{a.params}.png"
    fig.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    main()
