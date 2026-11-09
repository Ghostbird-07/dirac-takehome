"""Step 2-3 of the pipeline: fit each model on the 'fit' clips of a press type, then
predict the held-out 'test' clips with the fitted values (no refitting) and compare.

Writes outputs/fits/<model>_<interaction>.png and outputs/fits/results.json.

Run:  .venv/Scripts/python scripts/fit_and_predict.py [--models lumped]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import load_csv                          # noqa: E402
from softball.fitting import fit, rmse, free_mask, presser_path  # noqa: E402
from softball.lumped import LumpedSLS                           # noqa: E402

MODELS = {"lumped": LumpedSLS}


def plot(model, res, groups, out):
    clips = groups["fit"] + groups["test"]
    fig, axs = plt.subplots(2, len(clips), figsize=(6.2 * len(clips), 7.5), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1, 1.25]))
    for j, (role, c) in enumerate([("fit", c) for c in groups["fit"]] + [("test", c) for c in groups["test"]]):
        pred = model.predict(c, res.theta)
        m = free_mask(c)
        e = float(np.sqrt(np.mean((pred[m] - c.surface_mm[m]) ** 2)))
        p = presser_path(c)
        for row, (ax, zoom) in enumerate(((axs[0, j], False), (axs[1, j], True))):
            ax.plot(c.t, np.where(np.isfinite(p), p, np.nan), color="#805ad5", lw=1, alpha=0.6, label="presser (input)")
            ax.plot(c.t[m], c.surface_mm[m], "o", ms=3, color="#2f855a", label="measured, free recovery")
            ax.plot(c.t, pred, color="#c53030", lw=1.6, label="model")
            ax.axhline(0, color="k", lw=0.6); ax.invert_yaxis(); ax.grid(alpha=0.3)
            if zoom:
                t0 = c.release_t
                ax.set_xlim(t0 - 0.15, min(c.t[-1], t0 + 1.2))
                lo = np.nanmax(c.surface_mm[m]) if m.any() else 10
                ax.set_ylim(max(lo, np.nanmax(pred[(c.t > t0 - 0.15) & (c.t < t0 + 1.2)])) * 1.15 + 1, -2)
                ax.set_title(f"zoom on the free recovery - RMSE {e:.2f} mm", fontsize=9)
                ax.set_xlabel("time s")
            else:
                ax.set_title(f"{c.clip_id}  [{'FITTED on this clip' if role == 'fit' else 'HELD OUT: pure prediction'}]",
                             fontsize=10, color="#2b6cb0" if role == "fit" else "#c53030")
                ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel("mm pushed in")
    fig.suptitle(f"{model.name} - fitted: " + ", ".join(f"{n} = {v:.3g}" for n, v in zip(res.names, res.theta)),
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lumped"])
    args = ap.parse_args()
    cfg = json.loads((ROOT / "data" / "clips.json").read_text())
    curves = {c["id"]: load_csv(ROOT / "data" / "curves" / f"{c['id']}.csv") for c in cfg["clips"]}
    out_dir = ROOT / "outputs" / "fits"; out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for mname in args.models:
        model = MODELS[mname]()
        for inter in sorted({c["interaction"] for c in cfg["clips"]}):
            groups = {r: [curves[c["id"]] for c in cfg["clips"] if c["interaction"] == inter and c["role"] == r]
                      for r in ("fit", "test")}
            if not groups["fit"]:
                continue
            t0 = time.time()
            res = fit(model, groups["fit"])
            entry = res.as_dict()
            entry["fit_clips"] = [c.clip_id for c in groups["fit"]]
            entry["test_rmse_mm"] = {c.clip_id: rmse(model, res.theta, [c]) for c in groups["test"]}
            entry["seconds"] = round(time.time() - t0, 1)
            results[f"{mname}/{inter}"] = entry
            plot(model, res, groups, out_dir / f"{mname}_{inter}.png")
            print(f"[{mname} / {inter}] fit on {entry['fit_clips']}: rmse {res.rmse_fit:.2f} mm; "
                  f"test {entry['test_rmse_mm']}; params {entry['params']}  ({entry['seconds']} s)")
    res_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
