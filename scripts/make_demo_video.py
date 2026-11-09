"""Step 4: demo video. Real clip | simulated shell | measured vs predicted curve.

The simulation is driven ONLY by the measured nozzle path; everything the duck does
after the nozzle leaves is the model's prediction with the parameters fitted on the
'fit' clip. The held-out clip is labelled as such.

Run:  .venv/Scripts/python scripts/make_demo_video.py            (A1 then A2 -> outputs/demo.mp4)
      .venv/Scripts/python scripts/make_demo_video.py --clips A2
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import load_csv, read_frames       # noqa: E402
from softball.shell_model import ShellModel               # noqa: E402
from softball.lumped import LumpedSLS                     # noqa: E402
from softball.duck import make_duck, PART_COLOUR, DUCK_V2  # noqa: E402

GREEN, RED, PURPLE, GREY = "#2f855a", "#c53030", "#805ad5", "#718096"


def crop_box(clip_id):
    # region shown from the real video (x0, y0, x1, y1), 16:9
    return (560, 200, 1520, 740) if clip_id.startswith("A") else (0, 0, 1920, 1080)


def shell_axes(ax, x, ball, rest, pz, nozzle_r):
    faces = ball.faces
    top0 = ball.rest_top
    disp = np.linalg.norm(x - rest, axis=1) * 1000
    tri = x[faces] * 1000
    # colour = how far the rubber has moved, shaded by a light from the front-left so the
    # shape reads as 3D. The duck keeps its own colours (blue hat, cream body), tinted red
    # where it is pushed in; the plain sphere goes from pale (still) to red (25 mm or more).
    a = np.clip(disp[faces].mean(1) / 25.0, 0, 1)
    if getattr(ball, "part", None) is not None:
        own = np.array([PART_COLOUR[ball.part[f[0]]] for f in faces])
        base = (1 - 0.85 * a[:, None]) * own + 0.85 * a[:, None] * np.array([0.85, 0.15, 0.1])
    else:
        base = plt.cm.YlOrRd(0.08 + 0.85 * a)[:, :3]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.45, -0.75, 0.5]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.abs(nrm @ light), 0, 1)
    col = np.column_stack([base * shade[:, None], np.ones(len(base))])
    # the nozzle is drawn on top of the shell (as if seen through it): once it is inside the dent the
    # shell's triangles would otherwise hide it
    ax.computed_zorder = False
    pc = Poly3DCollection(tri, facecolors=col, edgecolors=(0, 0, 0, 0.08), linewidths=0.2, zorder=1)
    ax.add_collection3d(pc)
    if pz * 1000 < top0 * 1000 + 25:
        th = np.linspace(0, 2 * np.pi, 30)
        h = max(18.0, top0 * 1000 - pz * 1000 + 12)
        for zz, lw in ((pz * 1000, 2.2), (pz * 1000 + h, 1.2)):
            ax.plot(nozzle_r * 1000 * np.cos(th), nozzle_r * 1000 * np.sin(th), zz, color=PURPLE, lw=lw, zorder=5)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            ax.plot([nozzle_r * 1000 * np.cos(a)] * 2, [nozzle_r * 1000 * np.sin(a)] * 2,
                    [pz * 1000, pz * 1000 + h], color=PURPLE, lw=1, zorder=5)
    if getattr(ball, "part", None) is None:
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_zlim(0, 75)
        ax.set_box_aspect((1, 1, 1.25)); ax.view_init(14, -90)
    else:
        P = ball.rest_pos * 1000
        c = P.mean(0)
        half = 0.5 * np.ptp(P, axis=0).max() + 6
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_zlim(0, 2 * half)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(12, 0 if ball.pose == "lying" else -60)
    ax.set_axis_off()


def render_clip(cid, role, model, theta, base_theta, writer, size, label=None):
    cur = load_csv(ROOT / "data" / "curves" / f"{cid}.csv")
    cfg = json.loads((ROOT / "data" / "clips.json").read_text())
    clip = next(c for c in cfg["clips"] if c["id"] == cid)
    frames, tv = read_frames(str(ROOT / clip["file"]))
    r = model.simulate_curve(cur, theta, record_frames=True, fps=60.0)
    ts = r.t - model.t_pre
    pred = (r.top[0] - r.top) * 1000
    base = LumpedSLS().predict(cur, base_theta) if base_theta is not None else None
    rest = r.frames[0]
    x0, y0, x1, y1 = crop_box(cid)
    m = np.isfinite(cur.surface_mm)
    for i, (fr, t) in enumerate(zip(frames, tv)):
        k = int(np.argmin(np.abs(ts - t)))
        fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        a1 = fig.add_axes([0.0, 0.13, 0.40, 0.78])
        a1.imshow(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)); a1.axis("off")
        a1.set_title("real duck (video)", fontsize=10)
        tool = "nozzle" if cid.startswith("A") else "bottle"
        a2 = fig.add_axes([0.39, 0.10, 0.29, 0.84], projection="3d")
        shell_axes(a2, r.frames[k], model.shell, rest, r.presser_z[k], model.nozzle_radius)
        what = "duck" if getattr(model.shell, "part", None) is not None else "hollow"
        a2.set_title(f"simulated {what} shell + air\ninside pressure {r.pressure[k] / 1000:+.1f} kPa", fontsize=9)
        a3 = fig.add_axes([0.72, 0.20, 0.265, 0.68])
        p = cur.presser_mm
        a3.plot(cur.t, np.where(np.isfinite(p), p, np.nan), color=PURPLE, lw=1, alpha=0.7, label=f"{tool} (input)")
        a3.plot(cur.t[m], cur.surface_mm[m], "o", ms=2.5, color=GREEN, label="measured (free)")
        a3.plot(ts, pred, color=RED, lw=1.6, label="shell model")
        if base is not None:
            a3.plot(cur.t, base, color=GREY, lw=1, ls="--", label="1-point baseline")
        a3.axvline(t, color="k", lw=0.8)
        a3.axhline(0, color="k", lw=0.5)
        a3.set_ylim(max(60, np.nanmax(p) + 5), -5)
        a3.set_xlabel("time s"); a3.set_ylabel("mm pushed in"); a3.grid(alpha=0.3)
        a3.legend(fontsize=7, loc="lower right")
        ins = a3.inset_axes([0.42, 0.42, 0.55, 0.38])
        w0 = cur.release_t - 0.15
        ins.plot(cur.t[m], cur.surface_mm[m], "o", ms=2, color=GREEN)
        ins.plot(ts, pred, color=RED, lw=1.3)
        if base is not None:
            ins.plot(cur.t, base, color=GREY, lw=0.9, ls="--")
        ins.axvline(t, color="k", lw=0.6)
        ins.set_xlim(w0, w0 + 1.0); ins.set_ylim(np.nanmax(cur.surface_mm[m]) + 6, -1.5)
        ins.tick_params(labelsize=6); ins.set_title("free recovery (zoom)", fontsize=7)
        lab = label or ("FITTED on this clip" if role == "fit" else "HELD-OUT clip: pure prediction (parameters fitted on A1)")
        fig.text(0.5, 0.965, f"clip {cid}  -  {lab}", ha="center", fontsize=11, fontweight="bold",
                 color="#2b6cb0" if role == "fit" else RED)
        fig.text(0.86, 0.055, "played at half speed", fontsize=9, color=GREY)
        fig.text(0.02, 0.055, f"t = {t:5.2f} s     measured dent: "
                 + (f"{cur.surface_mm[i]:5.1f} mm" if np.isfinite(cur.surface_mm[i]) else f"  ({tool} in the way)  ")
                 + f"     model dent: {np.interp(t, ts, pred):5.1f} mm", fontsize=10, family="monospace")
        duck = getattr(model.shell, "part", None) is not None
        fig.text(0.02, 0.028 if duck else 0.015, f"Model input = measured {tool} path only. After the {tool} leaves, "
                 "the shell's recovery is predicted (air refilling through the squeaker hole + rubber springing back).",
                 fontsize=8, color=GREY)
        if duck:
            fig.text(0.02, 0.006, f"While pressed, the model curve is the top of its side outline (the rim around the "
                     f"{tool}); only the free recovery (green dots) is compared.", fontsize=8, color=GREY)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        plt.close(fig)


def main():
    import imageio.v2 as imageio
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sphere", "duck"], default="duck")
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = json.loads((ROOT / "outputs" / "fits" / "results.json").read_text())
    cfg = json.loads((ROOT / "data" / "clips.json").read_text())
    roles = {c["id"]: c["role"] for c in cfg["clips"]}
    key = "shell/face_press" if args.model == "sphere" else "duck/face_press"   # both fitted on A1 only
    sp = res[key]["params"]
    theta = np.array([sp[n] for n in ShellModel.names])

    def base_for(cid):
        k = "lumped/face_press" if cid.startswith("A") else "lumped/head_press"
        bp = res.get(k, {}).get("params")
        return np.array([bp[n] for n in LumpedSLS.names]) if bp else None

    if args.model == "sphere":
        clips = args.clips or ["A1", "A2"]
        out = args.out or str(ROOT / "outputs" / "demo.mp4")
        models = {"A": ShellModel()}
    else:
        clips = args.clips or ["A1", "A2", "B1"]
        out = args.out or str(ROOT / "outputs" / "demo_duck.mp4")
        models = {"A": ShellModel(ball=make_duck("lying", **DUCK_V2), dt=3e-3, nozzle_radius=0.0105),
                  "B": ShellModel(ball=make_duck("upright", **DUCK_V2), dt=3e-3, nozzle_radius=0.022)}
    size = (1600, 600)
    writer = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8)
    for cid in clips:
        label = None
        if args.model == "duck":
            if cid == "A1":
                label = "duck-shaped model FITTED on this clip"
            elif cid.startswith("A"):
                label = "HELD-OUT clip: pure prediction (duck-shaped model fitted on A1)"
            else:
                label = "DIFFERENT POSE: upright duck predicted with the lying-clip values (no refitting) - misses"
        role = ("fit" if cid == "A1" else "test") if args.model == "duck" else roles[cid]
        render_clip(cid, role, models[cid[0]], theta, base_for(cid), writer, size, label)
        print("rendered", cid, flush=True)
    writer.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
