# Learning a squeaky rubber duck from video

Take-home assignment, Dirac Robotics (Research Engineer Intern).

I filmed a soft object being pressed and released, measured its motion from the video,
and fitted a small **3D physics model** (a hollow rubber shell with air inside) so that it
reproduces the object's deformation and recovery. The fitted model then **predicts a press it
has never seen** (a held-out clip), and I compare that prediction with the real video.

**Demo videos: [watch on Google Drive](https://drive.google.com/drive/folders/1EO4Ki0_44nIfjDRSMTFZ35n6XyCKYM-V?usp=sharing)** (real duck | simulated shell | measured vs predicted curve,
half speed; also in `outputs/`):
* `outputs/demo_duck.mp4`: **main result**, duck-shaped model fitted on clip A1, then clip A2 predicted (held out),
  then an upright press (a new pose, which it misses).
* `outputs/demo.mp4`: simpler head-sphere model, fitted on A1, A2 predicted.

**Input videos:** [Google Drive folder](https://drive.google.com/drive/folders/12nWauCFJzXfhOsPW3W9WzL6BNKRlgUyI?usp=sharing) (details in [Data](#data)).

### Results at a glance

Error = root-mean-square difference (mm) between the predicted and the measured depth of the duck's surface
while it recovers on its own after the presser has left. **A2 is held out**: the model never saw it.

| model | parameters | fitted on A1 | **predicts held-out A2** |
|---|---|---|---|
| 1-point spring-damper (baseline) | 3 time constants | 1.20 mm | 2.01 mm (misses the pause-and-snap) |
| head as a hollow sphere + air (3D) | `Y`, `kappa`, `tau_v`, `tau_air` | 1.97 mm | 1.46 mm |
| duck-shaped shell + air, first version (v1) | same 4 | 1.2 mm | 7.0 mm (head crumpled, see §5b) |
| **duck-shaped shell + air (v2)** | same 4 | **0.61 mm** | **0.75 mm** |

Both shell models reproduce what the video shows and the baseline cannot: the dent is held in by suction after
the presser leaves, then recovers along an S-shaped curve; the duck v2 also catches the short pause before the
snap in A2, which the sphere skips. Caveat: halving the time step moves the duck prediction by 0.9 mm (§5c), so
its lead over the sphere is not large compared with its numerical uncertainty. A **different pose** (duck
upright, head pressed) is not predicted by any of them (duck v2: 19 mm on clip B1; see §5b).

---

## 1. The story in short

1. **First object: a foam smiley stress ball.** At 240 fps it turned out to recover *faster than a
   hand can let go*: the ball stays stuck to the presser all the way up
   (`outputs/v02_release_frames_clip864.jpg`). The recovery seen on video is the hand's speed,
   not the foam's, so its damping cannot be measured with a hand release. Kept as evidence,
   then switched object.
2. **Second object: a squeaky rubber duck (11 cm tall, 5.5 cm wide).** It recovers slowly enough
   that the presser leaves first and the duck is seen **recovering on its own**.
3. The video showed **how** it deforms: the head **caves in** (the face inverts like a dented
   plastic bottle), stays dented after the presser leaves, then recovers, sometimes with a
   **pause and a snap** (`outputs/d01_duck_dent_curves.png`, `outputs/d02_duck_clip3_frames.jpg`).
4. That observation decided the model: a **hollow shell** (which can cave in and snap back)
   **with air inside and a small hole** (squeezing pushes air out through the squeaker; on release
   the shell cannot re-inflate faster than air flows back in, so the dent is held by suction).
5. A simple one-point spring-damper is fitted as a **baseline**, to show what the shell model adds.

## 2. Pipeline

```
video ──► tracking ──► depth curves ──► fit model on 'fit' clip ──► predict 'test' clip ──► compare + demo video
         (OpenCV)     data/curves/*.csv   (parameters)              (no refitting)
```

| step | script | output |
|---|---|---|
| 1. tracking | `scripts/track_clips.py` | `data/curves/<clip>.csv`, check images `outputs/tracking/<clip>.png` |
| 2. baseline fit + prediction | `scripts/fit_and_predict.py` | `outputs/fits/lumped_*.png` |
| 3. shell model fit + prediction | `scripts/fit_shell.py` | `outputs/fits/shell_face_press.png`, `sphere_search.json` |
| 3b. duck-shaped model | `scripts/fit_shell.py --duck` | `outputs/fits/duck_face_press.png`, `duck_head_press_transfer.png` |
| crumple check (v1 vs v2) | `scripts/check_duck_collapse.py --params v2` | `outputs/fits/duck_collapse_check_v2.png` |
| 4. demo videos | `scripts/make_demo_video.py --model sphere` / `--model duck` | `outputs/demo.mp4`, `outputs/demo_duck.mp4` |
| accuracy check | `scripts/check_convergence.py` | time-step sensitivity (printed) |
| reproducibility check | `scripts/verify_results.py` | reported vs rerun errors (printed) |
| (figures of the solver itself) | `scripts/make_progress_figures.py` | `outputs/p01..p06*` |

All clips, their role (fit / test) and the few values read off the footage by hand are in
`data/clips.json`.

### How to run

Tested with Python 3.10 on Windows 11 (a fresh environment from `requirements.txt`).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # (Linux/macOS: .venv/bin/python)

# quick check (~1-2 min, no videos needed): rerun every fitted model with its saved parameters
# and compare with the errors reported below
.venv/Scripts/python scripts/verify_results.py

# full pipeline
.venv/Scripts/python scripts/track_clips.py                    # needs the videos (see Data)
.venv/Scripts/python scripts/fit_and_predict.py
.venv/Scripts/python scripts/fit_shell.py                      # sphere: ~40 min on 14 cores (--quick: a few minutes)
.venv/Scripts/python scripts/fit_shell.py --duck               # duck shape: ~1 h on 14 cores
.venv/Scripts/python scripts/check_duck_collapse.py --params v2
.venv/Scripts/python scripts/make_demo_video.py --model sphere
.venv/Scripts/python scripts/make_demo_video.py --model duck
.venv/Scripts/python scripts/check_convergence.py
```

The videos must be in `duck videos/` (and `user videos/` for the smiley ball), as listed in `data/clips.json`.
Only tracking and the demo videos need them; the depth curves they produce are included in `data/curves/`.

Everything is deterministic (fixed grids, Nelder–Mead from fixed starts, no random numbers), so a rerun gives the
same numbers. Checked from a clean copy and a fresh environment: tracking gives byte-identical curves, refitting the
baseline and the sphere model from scratch gives the same parameters to the last digit, and `verify_results.py`
reproduces every reported error to 0.00 mm.

## 3. Measuring the motion (tracking)

Two press types were filmed, so there are two trackers (`softball/tracking.py`):

* **Face press (A clips)**: duck lying on its back, the transparent nozzle of a glue bottle pushed into its face.
  * The nozzle's **purple cap** is found by colour; the nozzle tip is a fixed 228 px below the cap's lower edge
    (measured where the tip is visible). That gives the **presser path**, the model's input.
  * The duck's surface is found against the **dark cloth** behind it, in the column under the nozzle. It is only
    trusted when nothing is above it ("free"): this is the duck **recovering on its own**, the model's target.
* **Head press (B clips)**: duck upright and held, flat bottom of a bottle pushed down on its head. The duck leans
  and turns, so the **blue hat** (top of the head) is followed instead of a fixed column. The frame where the bottle
  leaves the hat was read off by eye.
* Real frame timestamps are used (the phone dropped some frames in dim light).
* **Scale:** a ruler in a reference clip (8.1–8.5 px/mm) → 7.8 px/mm for the face-press clips (±10%); for the
  upright clips, the duck's 110 mm height (7.5 px/mm) and the hat width. The scale only changes mm values, not timings.

## 4. Models

### 4a. The 3D simulator (`softball/sim.py`)

Particles joined by springs/dampers, stepped with **linearised backward Euler** (Baraff & Witkin 1998):
each step solves `(M + dt·D + dt²·K) Δv = dt·(f − dt·K·v)`. Implicit stepping is needed because rubber/foam
is heavily damped relative to the tiny particle masses (an explicit solver would need ~1e-7 s steps).
The sparse solve reuses an LU factorisation as a preconditioner for conjugate gradients (~5× faster).
Contacts: a flat-ended cylindrical presser (penalty contact with friction, soft edge), and a table to which a
small base patch is glued. Verified: a resting weight is held up by exactly m·g (`outputs/p03_weight_force_check.png`).

### 4b. Hollow shell + air (`softball/shell.py`, `softball/shell_model.py`)

* **Geometry:** the head as a 55 mm sphere, meshed as an icosphere (162 vertices for fitting); the nozzle (21 mm
  flat end, measured) pushes on the top vertex, and the top vertex's height is compared with the video. (The
  duck-shaped variant is described in §5b.)
* **Rubber:** a spring along each triangle edge (stretch) and across each edge (bend), each with a damper.
* **Air:** pressure from the gas law with the enclosed volume of the triangles, `p = p_atm · air / V`; air leaks
  through the hole at a rate proportional to the pressure difference. The pressure's stiffness is a rank-one
  matrix (all vertices share one pressure), handled implicitly with the Sherman–Morrison formula.
* **Fitted parameters (4):**

| | meaning | what in the video sets it |
|---|---|---|
| `Y` [N/m] | stretch stiffness of the rubber skin | how strongly the dent pushes back against the suction |
| `kappa` [-] | bend / stretch spring ratio | whether the cave-in pops back, and the snap |
| `tau_v` [s] | rubber damping time | smoothness of the spring-back |
| `tau_air` [s] | air-hole time constant | how long the dent is held after release |

Because air pressure has a known absolute scale (p_atm), the rubber stiffness **is identifiable in newtons**
from displacement-only video, unlike a solid foam model where only time constants are.

### 4c. Baseline: one-point spring-damper (`softball/lumped.py`)

The dent depth as a single standard linear solid (spring + damper + slow "memory" arm), with the presser as a
one-sided constraint. It can only relax smoothly: no cave-in, no snap.

## 5. Fitting and testing

* Only the **free recovery** (after the presser has left) is scored: during contact the surface simply follows
  the presser, so it says nothing about the material. Score = RMSE (mm) over free frames.
* **Fit on A1, test on A2** (same press type, different depth and hold time). The test clip is never used to
  choose parameters.
* The shell model's error surface is not smooth (buckled ↔ un-buckled), so the search is a parallel grid over the
  4 parameters followed by Nelder–Mead (slope-free) from the best grid points.

### 5a. Results: head-sphere model

`outputs/fits/shell_face_press.png`, `outputs/demo.mp4`

* Fitted on A1: `Y` ≈ 2.1·10³ N/m, `kappa` ≈ 0.48, `tau_v` ≈ 0.001 s, `tau_air` ≈ 0.027 s. RMSE 1.97 mm.
* Held-out A2: RMSE **1.46 mm** (baseline 2.01 mm). The model shows the held dent and the small kink (snap)
  that the baseline's smooth relaxation misses. It recovers ~0.1 s earlier than the real duck.
* **Identifiability:** the four Nelder–Mead starts ended at different settings with similar errors
  (e.g. `Y` ≈ 2·10³ N/m with `tau_air` ≈ 0.03 s, vs `Y` ≈ 2·10⁴ N/m with `tau_air` ≈ 0.01 s, RMSE 1.97 vs 2.61 mm).
  Stiffness and air-hole time trade off when only one clip is fitted. The bending ratio is consistent (0.4–0.5).
* Baseline on the upright press (fit B1 → predict B2): 0.82 mm → 2.93 mm; it again misses the plateau-then-snap.

### 5b. Duck-shaped model (main result)

`softball/duck.py`, `softball/bending.py`, `outputs/fits/duck_*`, `outputs/demo_duck.mp4`

**Why:** the sphere stands in for the head only. A duck-shaped shell lets one model cover both poses filmed.

**Geometry:** a smooth blend of three ellipsoids (body 48×41×~45 mm, head 51×52×55 mm, hat 42 mm across ×14 mm),
sized from the standing and lying reference clips; the parts touching the table (and held by the hand when
upright) are glued.

**First version (v1) and what it took to make it run** (each found by looking at the simulation):
1. *Pressed in 52 mm = the whole head height.* The lying-clip scale was re-measured from the duck's own head
   (8.1 px/mm, not the 7.8 first assumed) and the presser now stops above the table (face against the back of
   the head, two rubber layers) instead of being driven into the table (which gave 30 kN contact forces).
2. *Contact jitter* (forces up to 19 kN as mesh points flickered in and out of a stiff penalty contact) → the presser
   is a hard contact: points under it are placed on its face.
3. *Face stuck inside out after release.* Across-edge bending springs cannot tell a fold from its mirror image, so
   folds were as happy as the rest shape. Replaced by angle-based bending hinges (Bridson et al. 2003; gradient
   checked against finite differences to 1e-10). The duck then pops back.
4. *What is measured:* the model reports what the camera sees, the top of the outline in the column under the
   nozzle (lying) or the top of the hat (upright), not one mesh point (which slid out from under the nozzle).

v1 fitted A1 (1.2 mm) but predicted the held-out A2 badly (**7.0 mm**; its outputs and demo video are kept in
`outputs/fits/duck_v1/`), and its demo showed why: the head
**crumpled** like paper and the face went **through** the back of the head.

**Why v1 crumpled, and the fixes in v2** (`outputs/fits/duck_collapse_check_v2.png`):

| cause in v1 | fix in v2 |
|---|---|
| Mesh points placed by casting rays from one centre: 301 of 380 points on the head, body triangles 13–22 mm wide. Folds the size of one triangle. | 900 points spread evenly over the surface (farthest-point sampling + Lloyd smoothing): triangles 3.4–7.6 mm everywhere. |
| No self-contact: the pressed face could pass through the back of the head. | Self-contact: two points that were apart at rest and come closer than 3 mm push apart with a stiff spring, solved implicitly like the rest. (Simply moving them apart after each step was tried first; it pumped energy into the air.) |
| Bending stiffness per hinge ∝ edge length², so a coarse mesh made the body stiff plates that flopped (~37 mm). | Bending from the discrete-shell formula (Grinspun et al. 2003): the same material whatever the mesh size. |
| — | The presser stop is computed from the shell (the back wall under the nozzle + self-contact gap): 5.7 mm. |

Both meshes run on clip A2 with the v2 fitted values: at the deepest press, sharply folded hinges 141 → 19 and
wall pass-throughs 17 → 1 (a smooth dent instead of a crumple; the v1 body also twists, v2's stays put). One run
takes ~45 s instead of ~20 s.

**Results (v2, refitted on A1 only; grid of 55 settings + Nelder–Mead from the best 3)**
* Fitted on A1: **0.61 mm**. `Y` ≈ 3·10⁴ N/m, `kappa` ≈ 12, `tau_v` ≈ 0.06 s, `tau_air` ≈ 0.18 s.
* **Held-out A2: 0.75 mm** (v1 7.0 mm, sphere 1.46 mm, baseline 2.01 mm), including the pause-then-snap at ~3.3 s.
* Fitting A1 and A2 together (`--joint`, a check only) ends at the same values (0.68 mm over both clips): the
  two presses need no compromise.
* Parameters are still not unique: other starts ended at `Y` ≈ 3.9·10⁴, `tau_air` ≈ 0.024 s (0.97 mm on A1).
  `kappa` and `tau_v` ended at the top of the grid range, so the best values may lie beyond it.
* **New pose, no refitting:** upright clips B1 **19.1 mm**, B2 10.5 mm (v1: 16.9 mm on B1). The model's head
  stays pushed in for over a second. Upright, the real head tilts sideways at the neck (that is most of the
  70 mm the hat drops) while a hand squeezes the body; the model represents neither and treats the whole drop as
  crushing the head. B2's press input is also unreliable (the hat is hidden while pressed).

**Verdict:** after fixing the mesh and contacts, the duck-shaped model predicts a fresh press of the same kind
well (0.75 mm), better than the sphere. It does not transfer to a different pose; that would need the neck
tilt and the hand in the model.

### 5c. Accuracy of the time step

`scripts/check_convergence.py`: halving the step moves the predicted free recovery by **0.6 mm** RMS for the sphere
(2 → 1 ms) and **0.9 mm** for the v2 duck (3 → 1.5 ms; v1: 1.3 mm). That is as large as the duck's prediction
error (0.75 mm): the fitted values partly compensate for the step used, especially for the snapping duck, so the
step is part of the model. A cleaner version would refit at the smaller step (about twice the compute).

## 6. Assumptions

* **One air chamber, one leak.** The toy is a closed rubber skin with a single small hole (the squeaker).
  Air flow through the hole is taken as proportional to the pressure difference (a linear leak, one time
  constant `tau_air`). Real flow through a small orifice is closer to ∝ √Δp and the squeaker valve may
  behave differently in and out.
* **Air at constant temperature** (isothermal gas law). Presses last seconds, so heat has time to exchange.
* **Uniform rubber.** One stretch stiffness, one bending ratio and one damping time for the whole skin.
  In reality the hat, beak and feet are thicker and stiffer.
* **Rough geometry.** The head is a 55 mm sphere (sphere model) or the duck is a smooth blend of three
  ellipsoids sized from the video (duck model); feet, arms, beak and bow are ignored.
* **Presser = flat rigid disc moved exactly as tracked.** The nozzle path measured from the video is the
  model's input. It is assumed to press straight down on the top of the face/hat and not slide.
* **What the camera sees is the top of the surface under the presser.** The side view gives one depth per
  frame (the dent depth). The model is compared on that single number; its full 3D shape is not measured.
* **Glued contact patch.** The part of the duck resting on the table (and held by the hand in the upright
  clips) does not move.
* **Real-time scale.** Lying clips: 8.1 px/mm (±4%, from the duck's own head height against a ruler clip);
  upright clips: 7.5 and 9.2 px/mm (±10%, from the duck's height and hat width).

## 7. Limitations (what the model can and cannot capture)

**Can capture**
* The press (driven by the measured presser path) and the **free recovery** afterwards, including the dent
  being held for a moment (suction) and recovering along an S-shaped curve with a small snap, which a
  single spring-damper cannot do.
* A **fresh press of the same kind** (held-out clip A2), predicted without refitting, better than the
  baseline.
* Physical meaning: rubber stiffness in N/m (the air pressure gives an absolute force scale), bending,
  rubber damping, air-hole time.

**Cannot capture / not validated**
* **Parameters are not uniquely determined.** Only one clip is used for fitting, and several quite
  different settings fit almost equally well (e.g. stiffer rubber with a faster air hole vs softer rubber
  with a slower one; see `outputs/fits/*_search.json`). The *predicted curves* agree; the individual
  numbers should not be over-interpreted. More clips with varied depth and hold time would separate them.
* **Timing offset:** the model recovers ~0.1 s earlier than the real duck in both lying clips.
* **Force is not validated.** No force was measured (no scale under the duck), so the predicted presser
  force is not compared with anything.
* **Mesh artefacts.** A coarse spring mesh pushed in by ~50 mm folds over locally; with across-edge bending
  springs (sphere model) such folds have no preferred direction and can get stuck. The duck model uses angle-based
  bending hinges, an even 900-point mesh and self-contact to avoid this; ~30 hinges still fold sharply at the
  deepest press.
* **Time-step sensitivity** (§5c): 0.6 mm (sphere) and 0.9 mm (duck) change when the step is halved.
* **A different pose is not predicted** (§5b): the upright press involves the neck tilting and a hand holding
  the body, neither of which is modelled.
* **Contacts are simplified:** the presser stops at a computed gap above the back wall, and the hand holding the
  upright duck is a glued region, not a hand.
* **Only the depth under the presser is compared.** The 3D shape (bulging sides, how the face folds) is shown
  in the demo but not measured quantitatively.
* **Scale uncertainty** (±4–10%) shifts all millimetre values together; it does not change the timings.
* **The first object (foam ball) is not modelled:** it recovers faster than a hand release, so its damping is
  not measurable this way (it would need e.g. a weight knocked off the ball, or a drop test).

## Data

Phone: Motorola Edge 40. Duck clips 1080p at ~60 fps (some frames dropped in dim light; real timestamps are
used). Smiley-ball clips 1080p60, one 4K30, and two **slow-motion clips captured at 240 fps** (stored as
720p at 30 fps, i.e. 8× slowed; real time = frame / 240).

| folder | contents |
|---|---|
| `duck videos/` | 9 duck clips (2 reference clips with a ruler, 2 lying face presses A1/A2, 3 upright head presses B1–B3, 2 unused) |
| `user videos/` | 10 smiley-ball clips (first object; shows why it was replaced) |

The videos (~330 MB) are not in the git repository: **[download from Google Drive](https://drive.google.com/drive/folders/12nWauCFJzXfhOsPW3W9WzL6BNKRlgUyI?usp=sharing)** (folders
`duck videos/` and `user videos/`). Put the two folders in the repository root to rerun the pipeline. The extracted depth curves are included (`data/curves/*.csv`), so the
fitting and plotting steps run without the videos.

## Credits

Libraries: NumPy, SciPy (sparse LU, optimisation), Numba, OpenCV, Matplotlib, imageio/ffmpeg.

Method references:
* Baraff & Witkin, *Large Steps in Cloth Simulation* (SIGGRAPH 1998): implicit integration.
* Bridson, Marino & Fedkiw, *Simulation of Clothing with Folds and Wrinkles* (SCA 2003): angle-based bending hinges.
* Grinspun, Hirani, Desbrun & Schröder, *Discrete Shells* (SCA 2003): mesh-independent bending stiffness.
* Pressure-based soft bodies (gas law on a closed mesh) are a standard technique in physically based animation.
