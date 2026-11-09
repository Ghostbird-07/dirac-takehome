"""Draw everything built so far (before any real video): the model, the bugs found
and fixed, the force check, how the model behaves, solver speed, and an animation.

All material numbers here are GUESSES used only to show what the model can do.
Nothing is fitted yet: fitting waits for the real videos.

Run:  .venv/Scripts/python scripts/make_progress_figures.py
Writes PNGs and an MP4 to outputs/.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import softball.sim as S                                   # noqa: E402
from softball import (Ball, Params, Presser, make_ball,   # noqa: E402
                      press_release, simulate, weight_on_top)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
MM = 1e3
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"

ball = make_ball(radius=0.035, n_across=9, mass=0.03)
print(f"ball: {ball.n} particles, {len(ball.springs)} springs, spacing {ball.spacing*MM:.2f} mm")


# ---------------------------------------------------------------------------
# 1. the ball model
# ---------------------------------------------------------------------------
def fig_lattice():
    p = ball.rest_pos * MM
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    seg = p[ball.springs]
    for s in seg[::3]:
        ax.plot(*s.T, color=GREY, lw=0.3, alpha=0.35)
    ax.scatter(*p.T, s=10, c=BLUE, depthshade=True)
    ax.scatter(*p[ball.base_idx].T, s=16, c=GREEN, label="glued base (contact patch)")
    top = ball.axis_idx[np.argmax(p[ball.axis_idx, 2])]
    ax.scatter(*p[top], s=60, c=RED, label="marker particle (top)")
    ax.set_title("3D lattice: particles + springs\n(1 in 3 springs drawn)")
    ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("z mm")
    ax.set_box_aspect((1, 1, 1)); ax.legend(loc="upper left", fontsize=7)

    ax = fig.add_subplot(1, 3, 2)
    sl = np.abs(p[:, 1]) < 1e-6
    q = p[sl]
    pairs = [(i, j) for i, j in ball.springs if sl[i] and sl[j]]
    for i, j in pairs:
        ax.plot([p[i, 0], p[j, 0]], [p[i, 2], p[j, 2]], color=GREY, lw=0.6)
    ax.scatter(q[:, 0], q[:, 2], s=25, c=BLUE, zorder=3)
    ax.scatter(p[sl & np.isin(np.arange(ball.n), ball.base_idx), 0],
               p[sl & np.isin(np.arange(ball.n), ball.base_idx), 2], s=30, c=GREEN, zorder=4)
    ax.scatter(*p[top, [0, 2]], s=70, c=RED, zorder=5)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(35 * np.cos(th), 35 + 35 * np.sin(th), "k--", lw=1, label="real ball outline (Ø70 mm)")
    ax.axhline(0, color="k", lw=2)
    ax.set_aspect("equal"); ax.set_title("Middle slice (side view)")
    ax.set_xlabel("x mm"); ax.set_ylabel("height mm"); ax.legend(fontsize=7, loc="lower center")

    ax = fig.add_subplot(1, 3, 3)
    deg = np.bincount(ball.springs.ravel(), minlength=ball.n)
    ax.hist(deg, bins=np.arange(deg.min(), deg.max() + 2) - 0.5, color=BLUE)
    ax.set_title("Springs per particle\n(inside particles have all 26 neighbours)")
    ax.set_xlabel("number of springs"); ax.set_ylabel("particles")
    fig.suptitle(f"Figure 1 - The stress-ball model: {ball.n} particles, {len(ball.springs)} springs, "
                 f"spacing {ball.spacing*MM:.2f} mm, mass {ball.mass*1e3:.0f} g", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "p01_ball_model.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# 2. bug: the ball tipped over when it stood on one particle
# ---------------------------------------------------------------------------
def old_ball():
    """The first version: plain lattice sphere standing on a single bottom particle, not glued."""
    R, n = 0.035, 9
    a = 2 * R / n
    idx = np.arange(-(n // 2), n // 2 + 1)
    g = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1).reshape(-1, 3)
    pos = g * a
    keep = np.linalg.norm(pos, axis=1) <= R - 0.5 * a + 1e-6 * R
    pos, g = pos[keep], g[keep]
    pos[:, 2] += -pos[:, 2].min() + 0.5 * a
    pairs = np.array(sorted(cKDTree(pos).query_pairs(np.sqrt(3) * a * 1.01)), dtype=np.int64)
    L0 = np.linalg.norm(pos[pairs[:, 1]] - pos[pairs[:, 0]], axis=1)
    axis = np.where((g[:, 0] == 0) & (g[:, 1] == 0))[0]
    return Ball(R, a, 0.03, pos, pairs, L0, axis, np.array([], dtype=np.int64))


def fig_tipping():
    far = Presser(0.008, np.array([0.0]), np.array([1.0]))
    p = Params(E=2e4, tau_v=0.05)
    res = {}
    for name, b in (("before: stands on 1 particle", old_ball()), ("after: flat glued base", ball)):
        r = simulate(b, p, far, 3.0, fps=40, record_frames=True)
        top = b.axis_idx[np.argmax(b.rest_pos[b.axis_idx, 2])]
        side = np.hypot(*(r.frames[:, top, :2] - r.frames[0, top, :2]).T)
        res[name] = (b, r, side)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1.6, 1, 1]))
    for (name, (b, r, side)), c in zip(res.items(), (RED, GREEN)):
        axs[0].semilogy(r.t, np.maximum(side * MM, 1e-6), color=c, label=name)
        axs[1].plot(r.t, r.top * MM, color=c, label=name)
    axs[0].set_title("Sideways drift of the top particle (log scale)")
    axs[0].set_xlabel("time s"); axs[0].set_ylabel("mm"); axs[0].legend(fontsize=8)
    axs[1].set_title("Height of the marker - nothing is touching the ball!")
    axs[1].set_xlabel("time s"); axs[1].set_ylabel("mm"); axs[1].legend(fontsize=8)
    b, r, _ = res["before: stands on 1 particle"]
    for ax, k, ttl in ((axs[2], 0, "before, t = 0 s"), (axs[3], -1, f"before, t = {r.t[-1]:.0f} s")):
        q = r.frames[k] * MM
        ax.scatter(q[:, 0], q[:, 2], s=6, c=q[:, 1], cmap="coolwarm")
        ax.axhline(0, color="k", lw=2); ax.set_aspect("equal"); ax.set_title(ttl)
        ax.set_xlim(-60, 60); ax.set_ylim(-5, 80); ax.set_xlabel("x mm")
    fig.suptitle("Figure 2 - Bug found and fixed: a ball balanced on one particle tips over like a pencil on its tip",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "p02_tipping_bug.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. force check with a known weight
# ---------------------------------------------------------------------------
def fig_weight():
    p = Params(E=2e4, tau_v=0.05)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for m, c in ((0.1, BLUE), (0.2, ORANGE), (0.5, RED)):
        r = simulate(ball, p, weight_on_top(ball.rest_top, m), 1.0, fps=100)
        axs[0].plot(r.t, r.presser_z * MM, color=c, label=f"{m*1e3:.0f} g weight")
        axs[1].plot(r.t, r.force, color=c, label=f"{m*1e3:.0f} g")
        axs[1].axhline(m * 9.81, color=c, ls=":", lw=1)
        axs[2].plot(r.t, r.table_force, color=c)
        axs[2].axhline(m * 9.81 + ball.mass * 9.81, color=c, ls=":", lw=1)
        sag = (ball.rest_top - r.presser_z[-1]) * MM
        axs[0].annotate(f"sinks {sag:.1f} mm", (r.t[-1], r.presser_z[-1] * MM), fontsize=8,
                        ha="right", va="bottom", color=c)
    axs[0].set_title("Weight sinks into the ball, then stops"); axs[0].set_ylabel("weight height mm")
    axs[1].set_title("Force on the weight (dotted = m × g)"); axs[1].set_ylabel("N")
    axs[2].set_title("Table push-back (dotted = weight + ball's 0.29 N)"); axs[2].set_ylabel("N")
    for ax in axs:
        ax.set_xlabel("time s")
    axs[0].legend(fontsize=8)
    fig.suptitle("Figure 3 - Force check: a resting weight is held up by exactly m × g "
                 "(guessed stiffness E = 20 kPa)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "p03_weight_force_check.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. what the model does: short vs long hold, simple vs memory foam
# ---------------------------------------------------------------------------
def one_spring_recovery(p, depth, t_hold, t_down=0.35, dt=1e-3, t_after=3.0):
    """Same material as ONE spring (1D standard linear solid), pressed by `depth`
    (strain prescribed), then released (zero stress). Returns time after release and
    remaining dent in mm. Reference for what the 3D model 'should roughly' do."""
    k1, c1, k2 = 1.0, p.tau_v, p.beta
    eps, em = 0.0, 0.0
    t = 0.0
    while t < t_down + t_hold:                     # prescribed squeeze
        eps = depth * min(t / t_down, 1.0)
        em += dt / p.tau_m * (eps - em)
        t += dt
    out_t, out_e = [0.0], [eps]
    for i in range(int(t_after / dt)):              # free recovery: k1 e + c1 e' + k2 (e - em) = 0
        de = -(k1 * eps + k2 * (eps - em)) / c1
        eps += dt * de
        em += dt / p.tau_m * (eps - em)
        out_t.append((i + 1) * dt); out_e.append(eps)
    return np.array(out_t), np.array(out_e)


def fig_behaviour():
    kv = Params(E=2e4, tau_v=0.08)
    sls = Params(E=2e4, tau_v=0.08, beta=1.5, tau_m=1.0)
    depth, R_cap = 0.015, 0.015
    holds = (0.3, 3.0)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.2))
    store = {}
    for row, (p, name) in enumerate(((kv, "Simple foam (Kelvin-Voigt: spring + damper)"),
                                     (sls, "Memory foam (standard linear solid: + Maxwell arm)"))):
        for hold, c in zip(holds, (BLUE, ORANGE)):
            pr = press_release(ball.rest_top, depth, hold, radius=R_cap)
            t_rel = pr.key_t[3]
            r = simulate(ball, p, pr, t_rel + 3.0, fps=60)
            store[(row, hold)] = (r, t_rel)
            axs[row, 0].plot(r.t, r.top * MM, color=c, label=f"hold {hold} s")
            axs[row, 0].plot(r.t, np.minimum(r.presser_z, ball.rest_top + 0.006) * MM, color=c, ls=":", lw=1)
            axs[row, 1].plot(r.t, r.force, color=c, label=f"hold {hold} s")
            m = r.t >= t_rel
            dent = (ball.rest_top - r.top[m]) * MM
            axs[row, 2].plot(r.t[m] - t_rel, dent, color=c, label=f"3D model, hold {hold} s")
            t1, e1 = one_spring_recovery(p, dent[0], hold)
            axs[row, 2].plot(t1, e1, color=c, ls="--", lw=1, label=f"one spring alone, hold {hold} s")
        axs[row, 0].set_title(f"{name}\nmarker height (solid) and cap (dotted)", fontsize=9)
        axs[row, 0].set_ylabel("mm"); axs[row, 0].legend(fontsize=8)
        axs[row, 1].set_title("Force on the cap"); axs[row, 1].set_ylabel("N")
        axs[row, 2].set_title("Dent left after release"); axs[row, 2].set_ylabel("mm below rest")
        axs[row, 2].set_xlim(0, 3); axs[row, 2].legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("time s")
    axs[1, 2].set_xlabel("time after release s")
    fig.suptitle("Figure 4 - What the model can do (GUESSED parameters, nothing fitted): press 15 mm with a "
                 "30 mm cap, hold, lift.\nOnly the memory-foam version recovers differently after a long hold.",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "p04_model_behaviour.png"); plt.close(fig)
    return store


# ---------------------------------------------------------------------------
# 5. solver speed
# ---------------------------------------------------------------------------
def fig_speed():
    p = Params(E=2e4, tau_v=0.08, beta=1.5, tau_m=1.0)
    pr = press_release(ball.rest_top, 0.015, 1.0, radius=0.015)
    t0 = time.perf_counter(); simulate(ball, p, pr, 2.0); t_new = time.perf_counter() - t0
    orig = S._Stepper._solve
    S._Stepper._solve = lambda self, A, b, **k: spsolve(A.tocsc(), b)
    t0 = time.perf_counter(); simulate(ball, p, pr, 2.0); t_old = time.perf_counter() - t0
    S._Stepper._solve = orig
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.barh(["fresh factorisation\nevery step (first version)", "reuse old factorisation\n+ conjugate gradients (now)"],
                   [t_old, t_new], color=[GREY, GREEN])
    for b_, v in zip(bars, (t_old, t_new)):
        ax.text(v, b_.get_y() + b_.get_height() / 2, f" {v:.1f} s", va="center")
    ax.set_xlabel("seconds to simulate 2 s of pressing (same answer)")
    ax.set_title(f"Figure 5 - Solver speed-up: {t_old/t_new:.1f}x faster", fontweight="bold")
    ax.set_xlim(0, t_old * 1.25)
    fig.tight_layout(); fig.savefig(OUT / "p05_solver_speed.png"); plt.close(fig)
    print(f"speed: old {t_old:.1f} s, new {t_new:.1f} s")


# ---------------------------------------------------------------------------
# 6. animation
# ---------------------------------------------------------------------------
def make_video():
    import imageio.v2 as imageio
    p = Params(E=2e4, tau_v=0.08, beta=1.5, tau_m=1.0)
    R_cap = 0.015
    pr = press_release(ball.rest_top, 0.015, 1.5, radius=R_cap)
    r = simulate(ball, p, pr, pr.key_t[3] + 2.5, fps=30, record_frames=True)
    rest = r.frames[0]
    writer = imageio.get_writer(OUT / "p06_press_demo.mp4", fps=30, codec="libx264", quality=8,
                                macro_block_size=8)
    dz_all = (r.frames[:, :, 2] - rest[:, 2]) * MM
    vmin = dz_all.min()
    for k in range(len(r.t)):
        fig = plt.figure(figsize=(12, 4.4))
        q = r.frames[k] * MM
        dz = dz_all[k]
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=dz, cmap="viridis_r", vmin=vmin, vmax=0, s=12)
        cz = r.presser_z[k] * MM
        if cz < 90:
            th = np.linspace(0, 2 * np.pi, 40)
            ax.plot(R_cap * MM * np.cos(th), R_cap * MM * np.sin(th), cz, color=RED, lw=2)
        ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_zlim(0, 90)
        ax.set_box_aspect((1, 1, 1)); ax.set_title("3D view (colour = how far each particle moved down)", fontsize=9)
        ax.view_init(18, -60)
        ax = fig.add_subplot(1, 3, 2)
        ax.scatter(q[:, 0], q[:, 2], c=dz, cmap="viridis_r", vmin=vmin, vmax=0, s=14)
        if cz < 90:
            ax.add_patch(plt.Rectangle((-R_cap * MM, cz), 2 * R_cap * MM, 12, color=RED, alpha=0.8))
        ax.axhline(0, color="k", lw=2)
        ax.set_xlim(-50, 50); ax.set_ylim(-3, 90); ax.set_aspect("equal")
        ax.set_title("Side view (what the camera will see)", fontsize=9)
        ax = fig.add_subplot(1, 3, 3)
        ax.plot(r.t, r.top * MM, color=BLUE, label="marker height")
        ax.plot(r.t, np.minimum(r.presser_z * MM, 76), color=RED, ls=":", label="cap height")
        ax.plot(r.t[k], r.top[k] * MM, "o", color=BLUE)
        ax.axvline(r.t[k], color=GREY, lw=0.8)
        ax2 = ax.twinx(); ax2.plot(r.t, r.force, color=ORANGE, lw=1); ax2.set_ylabel("cap force N", color=ORANGE)
        ax.set_xlabel("time s"); ax.set_ylabel("mm"); ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"t = {r.t[k]:.2f} s   force = {r.force[k]:.1f} N", fontsize=9)
        fig.suptitle("Simulated press-and-release (memory foam, GUESSED parameters, not fitted)", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(img)
        if k == int(np.argmin(np.abs(r.t - (pr.key_t[2] + 0.5)))):
            fig.savefig(OUT / "p06_press_demo_frame.png")
        plt.close(fig)
    writer.close()


if __name__ == "__main__":
    t0 = time.time()
    fig_lattice(); print("fig 1 done")
    fig_tipping(); print("fig 2 done")
    fig_weight(); print("fig 3 done")
    fig_behaviour(); print("fig 4 done")
    fig_speed(); print("fig 5 done")
    make_video(); print("video done")
    print(f"all done in {time.time()-t0:.0f} s")
