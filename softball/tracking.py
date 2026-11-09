"""Turn the videos into numbers: how deep the duck is pushed in, frame by frame.

Two kinds of press were filmed, so there are two trackers.

A. face_press  (duck lying on its back, thin glue-bottle nozzle pushed into its face)
   * The nozzle is transparent, but its PURPLE cap is easy to find by colour.
     The nozzle tip is a fixed distance below the cap's lower edge, so
     tip height = cap bottom + nozzle length (measured where the tip is visible).
   * The duck's surface is found against the DARK cloth behind it: in the column
     under the nozzle, the duck's top edge is the lowest pixel that still belongs to
     the dark background region.
   * The surface is only trusted when nothing is above it ("free"): no purple cap
     in view and at least 60 px of background directly above. That is the part
     that shows the duck recovering on its own.

B. head_press  (duck upright and held, flat bottom of a glue bottle pushed down on its head)
   * The duck turns and leans, so instead of a fixed column we follow the BLUE HAT,
     which sits on top of the head: its top edge is the head's height.
   * While pressed, the bottle sits on the hat, so the hat top is also the presser
     position. The frame where the bottle leaves the hat was read off the video by
     eye (clips.json, 'free_start_frame').

Output per clip, all in millimetres measured DOWN from the resting surface
(0 = undisturbed, positive = pushed in):
   t           seconds (real frame timestamps; the phone dropped some frames)
   presser_mm  depth of the presser tip/bottom (NaN when unknown)
   surface_mm  depth of the duck's surface where it is visible and free (NaN otherwise)
"""
from dataclasses import dataclass

import cv2
import numpy as np

# scale for the face-press clips: the ruler in the lying-duck reference clip gives ~8.4 px/mm
# there, so the lying head (438 px tall) is 52 mm deep; in the face-press clips the same head
# is 420-423 px tall -> 8.1 px/mm. Uncertainty about +-4%; it scales mm values, not timings.
PX_PER_MM_FACE = 8.1


@dataclass
class Curve:
    clip_id: str
    t: np.ndarray
    presser_mm: np.ndarray
    surface_mm: np.ndarray
    px_per_mm: float
    release_t: float          # first time the surface is free (presser gone)
    raw: dict                 # pixel-level tracks, for checking


def read_frames(path):
    """All frames plus their real timestamps in seconds."""
    cap = cv2.VideoCapture(path)
    frames, t = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        frames.append(fr)
    t = np.asarray(t)
    return frames, t - t[0]


# ---------------------------------------------------------------------------
# A. face press
# ---------------------------------------------------------------------------
def _purple_cap_bottom(hsv, xc, half=150):
    purple = cv2.inRange(hsv[:, max(0, xc - half):xc + half + 1], (125, 80, 60), (165, 255, 255)) > 0
    rows = np.nonzero(purple.sum(1) > 80)[0]          # the cap is a wide purple band
    return float(rows.max()) if len(rows) else np.nan


def _background_top(hsv, xc, band, y_max, seed_dx=450, seed_y=300):
    """Lowest background row in the column (the duck's top edge is one below it)."""
    dark = ((hsv[:, :, 2] < 80) & (hsv[:, :, 1] < 140)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, lab = cv2.connectedComponents(dark)
    seeds = {lab[seed_y, xc - seed_dx], lab[seed_y, xc + seed_dx]} - {0}
    bg = np.isin(lab, list(seeds))
    col = bg[:y_max, xc - band:xc + band + 1].mean(1) > 0.5
    ys = np.nonzero(col)[0]
    if not len(ys):
        return np.nan, 0.0
    top = ys.max() + 1
    above = col[max(0, top - 60):top].mean()
    return float(top), float(above)


def find_nozzle_column(frames, n=10):
    xs = []
    for fr in frames[:n]:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        purple = cv2.inRange(hsv, (125, 80, 60), (165, 255, 255)) > 0
        rows = np.nonzero(purple.sum(1) > 80)[0]
        if len(rows):
            xs.append(np.median(np.nonzero(purple[rows])[1]))
    return int(np.median(xs))


def track_face_press(clip_id, path, nozzle_len_px=228, band=10, y_max=900,
                     px_per_mm=PX_PER_MM_FACE) -> Curve:
    frames, t = read_frames(path)
    xc = find_nozzle_column(frames)
    cap_b, top, above = [], [], []
    for fr in frames:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        cap_b.append(_purple_cap_bottom(hsv, xc))
        tp, ab = _background_top(hsv, xc, band, y_max)
        top.append(tp)
        above.append(ab)
    cap_b, top, above = map(np.asarray, (cap_b, top, above))
    free = np.isnan(cap_b) & (above > 0.9)
    # only frames AFTER the presser has left count as free (ignore the empty start of a clip)
    first_press = np.nonzero(~np.isnan(cap_b))[0]
    free &= np.arange(len(t)) > (first_press.max() if len(first_press) else -1)
    rest = np.nanmedian(top[free][-30:])                 # resting surface at the end
    tip = cap_b + nozzle_len_px
    presser = (tip - rest) / px_per_mm
    surface = np.where(free, (top - rest) / px_per_mm, np.nan)
    rel = t[np.nonzero(free)[0][0]]
    raw = dict(xc=xc, cap_bottom=cap_b, top=top, free=free, rest_px=rest)
    return Curve(clip_id, t, presser, surface, px_per_mm, float(rel), raw)


# ---------------------------------------------------------------------------
# B. head press
# ---------------------------------------------------------------------------
def _hat_top(hsv, min_area=3000):
    m = cv2.inRange(hsv, (100, 90, 50), (125, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > min_area]
    if not big:
        return np.nan, None
    i = min(big, key=lambda q: st[q, cv2.CC_STAT_TOP])   # the hat is the highest blue blob
    return float(st[i, cv2.CC_STAT_TOP]), st[i]


def head_width_px(frame):
    """Width of the head just under the hat in a resting frame (skin-coloured pixels)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    top, st = _hat_top(hsv)
    x, y, w, h = st[0], st[1], st[2], st[3]
    skin = cv2.inRange(hsv, (5, 25, 120), (30, 140, 255)) > 0
    widths = []
    for yy in range(y + h, min(hsv.shape[0], y + h + 150), 5):
        xs = np.nonzero(skin[yy, max(0, x - 100):x + w + 100])[0]
        if len(xs) > 20:
            widths.append(xs.max() - xs.min())
    return float(np.median(sorted(widths)[-5:])) if widths else np.nan


def track_head_press(clip_id, path, free_start_frame, head_width_mm=55.0, px_per_mm=None) -> Curve:
    frames, t = read_frames(path)
    top = np.asarray([_hat_top(cv2.cvtColor(fr, cv2.COLOR_BGR2HSV))[0] for fr in frames])
    if px_per_mm is None:
        px_per_mm = head_width_px(frames[-1]) / head_width_mm
    rest = np.nanmedian(top[-15:])
    depth = (top - rest) / px_per_mm
    idx = np.arange(len(t))
    presser = np.where(idx < free_start_frame, depth, np.nan)
    surface = np.where(idx >= free_start_frame, depth, np.nan)
    raw = dict(top=top, rest_px=rest, free=idx >= free_start_frame)
    return Curve(clip_id, t, presser, surface, float(px_per_mm), float(t[free_start_frame]), raw)


def track_clip(clip: dict, duck: dict) -> Curve:
    if clip["interaction"] == "face_press":
        return track_face_press(clip["id"], clip["file"], px_per_mm=clip.get("px_per_mm", PX_PER_MM_FACE))
    return track_head_press(clip["id"], clip["file"], clip["free_start_frame"],
                            head_width_mm=duck["width_mm"], px_per_mm=clip.get("px_per_mm"))


def save_csv(curve: Curve, path):
    arr = np.column_stack([curve.t, curve.presser_mm, curve.surface_mm])
    header = (f"clip {curve.clip_id}; px_per_mm {curve.px_per_mm:.3f}; release_t {curve.release_t:.4f}\n"
              "t_s,presser_mm,surface_mm  (mm measured DOWN from the resting surface; nan = not visible)")
    np.savetxt(path, arr, delimiter=",", fmt="%.5f", header=header)


def load_csv(path):
    with open(path) as fh:
        first = fh.readline()
    parts = dict(p.strip().split(" ", 1) for p in first.lstrip("# ").split(";"))
    a = np.loadtxt(path, delimiter=",", comments="#")
    return Curve(parts["clip"], a[:, 0], a[:, 1], a[:, 2], float(parts["px_per_mm"]),
                 float(parts["release_t"]), {})
