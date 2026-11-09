"""Step 1 of the pipeline: video -> depth curves.

Reads data/clips.json, tracks every clip, writes data/curves/<id>.csv and a check
figure outputs/tracking/<id>.png showing the curve and frames with the detections
drawn on, so the tracking can be verified by eye.

Run:  .venv/Scripts/python scripts/track_clips.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softball.tracking import read_frames, save_csv, track_clip   # noqa: E402

PURPLE, GREEN, ORANGE = "#805ad5", "#2f855a", "#dd6b20"


def check_figure(clip, cur, out):
    frames, t = read_frames(str(ROOT / clip["file"]))
    raw = cur.raw
    fig = plt.figure(figsize=(15, 6.4))
    ax = fig.add_axes([0.05, 0.56, 0.9, 0.36])
    ax.plot(cur.t, cur.presser_mm, ".", ms=3, color=PURPLE, label="presser (nozzle tip / bottle on the hat)")
    ax.plot(cur.t, cur.surface_mm, ".", ms=4, color=GREEN, label="duck surface, free (recovering on its own)")
    ax.axvline(cur.release_t, color=ORANGE, ls="--", lw=1, label=f"free from t = {cur.release_t:.2f} s")
    ax.axhline(0, color="k", lw=0.7)
    ax.invert_yaxis(); ax.set_ylabel("mm pushed in"); ax.set_xlabel("time s")
    ax.set_title(f"{clip['id']} ({clip['interaction']}, role: {clip['role']}) - {cur.px_per_mm:.2f} px/mm", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    # 6 frames: before, pressed, just before release, just after, mid-recovery, end
    ir = int(np.searchsorted(cur.t, cur.release_t))
    pressed = np.nanargmax(cur.presser_mm) if np.isfinite(cur.presser_mm).any() else 0
    picks = [0, int(pressed), max(0, ir - 3), ir, min(len(t) - 1, ir + 8), len(t) - 1]
    for k, i in enumerate(picks):
        fr = frames[i].copy()
        if clip["interaction"] == "face_press":
            xc = raw["xc"]
            cv2.line(fr, (xc, 0), (xc, fr.shape[0]), (0, 255, 255), 2)
            if np.isfinite(raw["cap_bottom"][i]):
                y = int(raw["cap_bottom"][i]); cv2.line(fr, (xc - 120, y), (xc + 120, y), (255, 0, 255), 4)
                y2 = int(raw["cap_bottom"][i] + 228); cv2.line(fr, (xc - 60, y2), (xc + 60, y2), (255, 0, 255), 4)
            if np.isfinite(raw["top"][i]):
                y = int(raw["top"][i]); col = (0, 200, 0) if raw["free"][i] else (0, 0, 255)
                cv2.line(fr, (xc - 80, y), (xc + 80, y), col, 4)
            yr = int(raw["rest_px"]); cv2.line(fr, (xc - 200, yr), (xc - 100, yr), (255, 255, 255), 3)
        else:
            if np.isfinite(raw["top"][i]):
                y = int(raw["top"][i]); col = (0, 200, 0) if raw["free"][i] else (255, 0, 255)
                cv2.line(fr, (0, y), (fr.shape[1], y), col, 4)
            yr = int(raw["rest_px"]); cv2.line(fr, (0, yr), (200, yr), (255, 255, 255), 4)
        a = fig.add_axes([0.01 + k * 0.165, 0.02, 0.16, 0.42])
        a.imshow(cv2.cvtColor(cv2.resize(fr, (640, 360)), cv2.COLOR_BGR2RGB)); a.axis("off")
        a.set_title(f"frame {i}, t = {t[i]:.2f} s", fontsize=8)
    fig.text(0.01, 0.47, "Lines on frames: magenta = cap bottom and nozzle tip / hat top while pressed; green = free surface; "
             "red = surface not yet free; white stub = resting level.", fontsize=8)
    fig.savefig(out, dpi=100); plt.close(fig)


def main():
    cfg = json.loads((ROOT / "data" / "clips.json").read_text())
    (ROOT / "data" / "curves").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "tracking").mkdir(parents=True, exist_ok=True)
    for clip in cfg["clips"]:
        clip = dict(clip, file=str(ROOT / clip["file"]))
        cur = track_clip(clip, cfg["duck"])
        save_csv(cur, ROOT / "data" / "curves" / f"{clip['id']}.csv")
        clip["file"] = str(Path(clip["file"]).relative_to(ROOT))
        check_figure(clip, cur, ROOT / "outputs" / "tracking" / f"{clip['id']}.png")
        fr = np.isfinite(cur.surface_mm)
        print(f"{clip['id']}: {len(cur.t)} frames, {cur.px_per_mm:.2f} px/mm, max press {np.nanmax(cur.presser_mm):.1f} mm, "
              f"free from {cur.release_t:.2f} s with dent {cur.surface_mm[fr][0]:.1f} mm")


if __name__ == "__main__":
    main()
