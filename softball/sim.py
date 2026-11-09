"""The simulator: moves every particle forward in time.

Forces on each particle
  * gravity (real mass)
  * the material springs/dampers between neighbouring particles (material.py)
  * the table: the flat bottom layer of particles is glued to it (a sticky
    contact patch); any other particle that reaches the table is pushed back
    out by a stiff penalty spring with strong sideways friction
  * the presser: a stiff penalty spring pushing particles down out of it,
    also with sideways friction (a fingertip does not slide on foam);
    the sum of these forces is the force the finger feels. For the hollow
    shells the presser is instead a hard contact (points under it are placed on it)
  * air inside a hollow shell: one pressure for the whole inside, from the gas law,
    leaking through the squeaker hole
  * self-contact (duck v2): parts of the shell that were far apart at rest push
    each other apart when they come closer than about two wall thicknesses, so
    the face cannot pass through the back of the head. Like the table, these are
    stiff springs inside the implicit solve (stable and dissipative); an earlier
    try that moved the points apart after each step pumped energy into the air

Time stepping: linearised backward (implicit) Euler, as in Baraff & Witkin
(1998), "Large steps in cloth simulation". Every step solves one sparse linear
system

        (M + dt*D + dt^2*K) dv = dt * (f - dt*K*v)

where M is mass, D is the damping matrix and K the stiffness matrix.

Why implicit? Foam is extremely over-damped: with the real particle mass
(about 0.1 g), a particle's velocity settles in about 1e-7 s, while the
recovery we film takes 0.1 to several seconds. An explicit solver would need
steps shorter than 1e-7 s (tens of millions of steps per clip). Backward
Euler stays stable with millisecond steps at the real mass. An earlier version
inflated the masses instead ("mass scaling") to make an explicit solver
stable. That turned out to give the whole ball tens of kilograms of inertia, so
it wobbled. Implicit stepping avoids the problem without faking inertia.

Backward Euler slightly over-damps: a decay with time constant tau comes out
as about tau + dt/2. For the smooth solid ball (tau >= 50 ms, dt = 2 ms) that is ~2%.
The snapping shells are more sensitive: scripts/check_convergence.py shows that halving
the step moves the predicted recovery by ~0.6 mm (sphere, 2 -> 1 ms) and ~1.3 mm (duck,
3 -> 1.5 ms), so the fitted values partly compensate for the step used.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from numba import njit

from .geometry import Ball
from .material import Params, maxwell_alpha  # noqa: F401
from .presser import Presser
from .bending import assemble_bending, hinge_slots

G = 9.81


# ----------------------------------------------------------------------------
# sparsity pattern of the system matrix (built once per ball)
# ----------------------------------------------------------------------------
def _pattern(ball: Ball, cand: np.ndarray, with_plate: bool):
    """COO (row, col) list in exactly the order _assemble writes values."""
    S = len(ball.springs)
    i = ball.springs[:, 0]
    j = ball.springs[:, 1]
    rows, cols = [], []
    a = np.arange(3)
    ra = np.repeat(a, 3)          # row offsets of a 3x3 block (row-major)
    cb = np.tile(a, 3)            # col offsets
    for (p, q) in ((i, i), (j, j), (i, j), (j, i)):
        rows.append((3 * p)[:, None] + ra[None, :])
        cols.append((3 * q)[:, None] + cb[None, :])
    # interleave: for each spring the 4 blocks in order -> shape (S, 4, 9)
    R = np.stack(rows, 1).reshape(S, 36)
    C = np.stack(cols, 1).reshape(S, 36)
    rows_all = [R.ravel()]
    cols_all = [C.ravel()]
    n = ball.n
    pi = np.arange(n)
    rows_all.append(((3 * pi)[:, None] + ra[None, :]).ravel())
    cols_all.append(((3 * pi)[:, None] + cb[None, :]).ravel())
    ndof = 3 * n + (1 if with_plate else 0)
    if with_plate:
        P = 3 * n
        rows_all.append(np.array([P]))
        cols_all.append(np.array([P]))
        zc = 3 * cand + 2
        rc = np.stack([np.full_like(zc, P), zc], 1).ravel()
        cc = np.stack([zc, np.full_like(zc, P)], 1).ravel()
        rows_all.append(rc)
        cols_all.append(cc)
    rows = np.concatenate(rows_all).astype(np.int64)
    cols = np.concatenate(cols_all).astype(np.int64)
    key = rows * ndof + cols
    uniq, inv = np.unique(key, return_inverse=True)
    ur = uniq // ndof
    uc = uniq % ndof
    indptr = np.concatenate([[0], np.cumsum(np.bincount(ur, minlength=ndof))]).astype(np.int32)
    return ndof, inv.astype(np.int64), indptr, uc.astype(np.int32), len(uniq)


@njit(cache=True, fastmath=True)
def _assemble(x, v, springs, L0, Lm, k1a, c1a, k2a, m, dt, rs,
              pr, h, vh, weight_mode, pz, pv, pm, kc, cc, kt, ctn, ctt,
              cand, inv, data, f, Kv, hw):
    """Fill CSR values `data`, force vector f and K @ v. Returns presser contact force (N)."""
    N = x.shape[0]
    S = springs.shape[0]
    for q in range(data.shape[0]):
        data[q] = 0.0
    for q in range(f.shape[0]):
        f[q] = 0.0
        Kv[q] = 0.0
    dt2 = dt * dt
    k = 0
    # ---- springs ------------------------------------------------------------
    for s in range(S):
        i = springs[s, 0]
        j = springs[s, 1]
        k1 = k1a[s]
        c1 = c1a[s]
        k2 = k2a[s]
        d0 = x[j, 0] - x[i, 0]
        d1 = x[j, 1] - x[i, 1]
        d2 = x[j, 2] - x[i, 2]
        L = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        n0 = d0 / L
        n1 = d1 / L
        n2 = d2 / L
        nv = (n0, n1, n2)
        rel = (v[j, 0] - v[i, 0]) * n0 + (v[j, 1] - v[i, 1]) * n1 + (v[j, 2] - v[i, 2]) * n2
        Fel = k1 * (L - L0[s]) + k2 * (L - Lm[s])
        F = Fel + c1 * rel
        g = Fel / L if Fel > 0.0 else 0.0      # geometric stiffness, clamped >= 0 (keeps matrix SPD)
        ka = k1 + k2
        dvi = (v[i, 0] - v[j, 0], v[i, 1] - v[j, 1], v[i, 2] - v[j, 2])
        for a in range(3):
            f[3 * i + a] += F * nv[a]
            f[3 * j + a] -= F * nv[a]
        # block B = dt*c1*nn^T + dt^2*Ks ; Ks = ka*nn^T + g*(I - nn^T)
        for blk in range(4):
            sign = 1.0 if blk < 2 else -1.0
            for a in range(3):
                for b in range(3):
                    nn = nv[a] * nv[b]
                    ks = ka * nn + g * ((1.0 if a == b else 0.0) - nn)
                    data[inv[k]] += sign * (dt * c1 * nn + dt2 * ks)
                    if blk == 0:
                        Kv[3 * i + a] += ks * dvi[b]
                    elif blk == 1:
                        Kv[3 * j + a] -= ks * dvi[b]
                    k += 1
    # ---- per-particle: mass, gravity, table, presser ------------------------
    fc_tot = 0.0
    pr2 = (pr + hw) * (pr + hw)
    for i in range(N):
        f[3 * i + 2] -= m * G
        dxx = m
        dyy = m
        dzz = m
        # table (penalty + sticky friction), active if predicted to touch
        zpred = x[i, 2] + dt * v[i, 2]
        if zpred < rs:
            f[3 * i + 2] += kt * (rs - x[i, 2]) - ctn * v[i, 2]
            Kv[3 * i + 2] += kt * v[i, 2]
            dzz += dt2 * kt + dt * ctn
            f[3 * i] -= ctt * v[i, 0]
            f[3 * i + 1] -= ctt * v[i, 1]
            dxx += dt * ctt
            dyy += dt * ctt
        # presser (kinematic only here; plate coupling handled below)
        if not weight_mode:
            r2 = x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1]
            if r2 < pr2:
                w = (pr + hw - np.sqrt(r2)) / (2.0 * hw)   # smooth edge of the footprint (width 2*hw)
                if w > 1.0:
                    w = 1.0
                pen_pred = zpred + rs - h
                if pen_pred > 0.0 and w > 0.0:
                    fz = w * (kc * (x[i, 2] + rs - h) + cc * (v[i, 2] - vh))
                    f[3 * i + 2] -= fz
                    Kv[3 * i + 2] += w * kc * v[i, 2]
                    dzz += w * (dt2 * kc + dt * cc)
                    f[3 * i] -= w * ctt * v[i, 0]
                    f[3 * i + 1] -= w * ctt * v[i, 1]
                    dxx += w * dt * ctt
                    dyy += w * dt * ctt
                    fc_tot += w * kc * pen_pred
        # write the diagonal 3x3 block (only diagonal entries non-zero)
        for a in range(3):
            for b in range(3):
                if a == b:
                    data[inv[k]] += dxx if a == 0 else (dyy if a == 1 else dzz)
                k += 1
    # ---- weight plate (one extra degree of freedom: its height) ------------
    if weight_mode:
        P = 3 * N
        f[P] -= pm * G
        kpp = 0.0
        cpp = 0.0
        kPP_idx = k
        data[inv[k]] += pm
        k += 1
        for c in range(cand.shape[0]):
            i = cand[c]
            r2 = x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1]
            w = 0.0
            if r2 < pr2:
                w = (pr + hw - np.sqrt(r2)) / (2.0 * hw)
                if w > 1.0:
                    w = 1.0
            zpred = x[i, 2] + dt * v[i, 2]
            pen_pred = zpred + rs - (pz + dt * pv)
            if w > 0.0 and pen_pred > 0.0:
                fz = w * (kc * (x[i, 2] + rs - pz) + cc * (v[i, 2] - pv))
                f[3 * i + 2] -= fz
                f[P] += fz
                Kv[3 * i + 2] += w * kc * (v[i, 2] - pv)
                Kv[P] += w * kc * (pv - v[i, 2])
                # add to particle i's diagonal block (slots: xx = zz-8, yy = zz-4)
                zz = _diag_slot(S, i)
                data[inv[zz]] += w * (dt2 * kc + dt * cc)
                f[3 * i] -= w * ctt * v[i, 0]
                f[3 * i + 1] -= w * ctt * v[i, 1]
                data[inv[zz - 8]] += w * dt * ctt
                data[inv[zz - 4]] += w * dt * ctt
                off = -w * (dt2 * kc + dt * cc)
                data[inv[k]] += off
                data[inv[k + 1]] += off
                kpp += w * kc
                cpp += w * cc
                fc_tot += w * kc * pen_pred
            k += 2
        data[inv[kPP_idx]] += dt2 * kpp + dt * cpp
    return fc_tot


@njit(cache=True, inline="always")
def _diag_slot(S, i):
    """COO index of the zz entry of particle i's diagonal block."""
    return 36 * S + 9 * i + 8


@njit(cache=True, fastmath=True)
def _maxwell_update(x, springs, Lm, alpha):
    for s in range(springs.shape[0]):
        i = springs[s, 0]
        j = springs[s, 1]
        d0 = x[j, 0] - x[i, 0]
        d1 = x[j, 1] - x[i, 1]
        d2 = x[j, 2] - x[i, 2]
        L = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        Lm[s] += alpha * (L - Lm[s])


@njit(cache=True, fastmath=True)
def _contact_forces(x, v, rs, pr, h, vh, kc, cc, kt, ctn, hw):
    """Presser and table forces evaluated at the NEW state (what the implicit step applied)."""
    fp = 0.0
    ft = 0.0
    pr2 = (pr + hw) * (pr + hw)
    for i in range(x.shape[0]):
        if x[i, 2] < rs:
            ft += kt * (rs - x[i, 2]) - ctn * v[i, 2]
        r2 = x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1]
        if r2 < pr2:
            w = (pr + hw - np.sqrt(r2)) / (2.0 * hw)
            if w > 1.0:
                w = 1.0
            pen = x[i, 2] + rs - h
            if pen > 0.0 and w > 0.0:
                fz = w * (kc * pen + cc * (v[i, 2] - vh))
                if fz > 0.0:
                    fp += fz
    return fp, ft


@njit(cache=True, fastmath=True)
def _volume_grad(x, faces, g):
    """Enclosed volume of a closed outward-oriented triangle mesh, and dV/dx per vertex."""
    for i in range(g.shape[0]):
        g[i, 0] = 0.0
        g[i, 1] = 0.0
        g[i, 2] = 0.0
    V = 0.0
    for f in range(faces.shape[0]):
        a = faces[f, 0]
        b = faces[f, 1]
        c = faces[f, 2]
        bc0 = x[b, 1] * x[c, 2] - x[b, 2] * x[c, 1]
        bc1 = x[b, 2] * x[c, 0] - x[b, 0] * x[c, 2]
        bc2 = x[b, 0] * x[c, 1] - x[b, 1] * x[c, 0]
        ca0 = x[c, 1] * x[a, 2] - x[c, 2] * x[a, 1]
        ca1 = x[c, 2] * x[a, 0] - x[c, 0] * x[a, 2]
        ca2 = x[c, 0] * x[a, 1] - x[c, 1] * x[a, 0]
        ab0 = x[a, 1] * x[b, 2] - x[a, 2] * x[b, 1]
        ab1 = x[a, 2] * x[b, 0] - x[a, 0] * x[b, 2]
        ab2 = x[a, 0] * x[b, 1] - x[a, 1] * x[b, 0]
        V += (x[a, 0] * bc0 + x[a, 1] * bc1 + x[a, 2] * bc2) / 6.0
        g[a, 0] += bc0 / 6.0
        g[a, 1] += bc1 / 6.0
        g[a, 2] += bc2 / 6.0
        g[b, 0] += ca0 / 6.0
        g[b, 1] += ca1 / 6.0
        g[b, 2] += ca2 / 6.0
        g[c, 0] += ab0 / 6.0
        g[c, 1] += ab1 / 6.0
        g[c, 2] += ab2 / 6.0
    return V


P_ATM = 101325.0


@njit(cache=True, fastmath=True)
def _hard_contact(x, v, rs, pr, h, vh, m, dt, is_base):
    """Points under the flat presser cannot be above its bottom face: put them on it, stop
    them moving up faster than it, and let them stick (no sliding). Returns the force the
    presser exerts (momentum removed per second)."""
    F = 0.0
    zmax = h - rs
    for i in range(x.shape[0]):
        if is_base[i]:
            continue
        if x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1] <= pr * pr and x[i, 2] > zmax:
            x[i, 2] = zmax
            if v[i, 2] > vh:
                F += m * (v[i, 2] - vh) / dt
                v[i, 2] = vh
            v[i, 0] = 0.0
            v[i, 1] = 0.0
    return F


@njit(cache=True, fastmath=True)
def _self_contact(x, v, x0, dmin, rex, k, c, dt, is_base, f, Kv, rows, cols, vals):
    """The rubber cannot pass through itself (duck v2).

    Two mesh points that were far apart on the surface at rest (rest distance > rex) and are
    now closer than dmin (about the thickness of two rubber walls pressed together) push
    each other apart with a stiff spring k and a damper c along the line joining them.
    Like the presser and the table, these springs go INTO the implicit solve: their force
    is added to f, K v to Kv, and the matrix blocks (dt^2 k + dt c) n n^T are returned as
    (rows, cols, vals) triplets. Glued points take no part. Returns the number of triplets."""
    n = x.shape[0]
    d2 = dmin * dmin
    r2x = rex * rex
    a = dt * dt * k + dt * c
    cnt = 0
    cap = rows.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j, 0] - x[i, 0]
            dy = x[j, 1] - x[i, 1]
            dz = x[j, 2] - x[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= d2:
                continue
            ex = x0[j, 0] - x0[i, 0]
            ey = x0[j, 1] - x0[i, 1]
            ez = x0[j, 2] - x0[i, 2]
            if ex * ex + ey * ey + ez * ez < r2x:
                continue                       # neighbours on the surface: the springs handle them
            if is_base[i] and is_base[j]:
                continue
            if cnt + 36 > cap:
                return cnt
            r = np.sqrt(r2) + 1e-12
            nv = (dx / r, dy / r, dz / r)
            vrel = ((v[j, 0] - v[i, 0]) * nv[0] + (v[j, 1] - v[i, 1]) * nv[1]
                    + (v[j, 2] - v[i, 2]) * nv[2])
            fmag = k * (dmin - r) - c * vrel        # > 0 pushes the two apart
            for q in range(3):
                if not is_base[i]:
                    f[3 * i + q] -= fmag * nv[q]
                    Kv[3 * i + q] -= k * nv[q] * vrel
                if not is_base[j]:
                    f[3 * j + q] += fmag * nv[q]
                    Kv[3 * j + q] += k * nv[q] * vrel
            for (pi, pj, sg) in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
                if is_base[pi] or is_base[pj]:
                    continue
                for q in range(3):
                    for w in range(3):
                        rows[cnt] = 3 * pi + q
                        cols[cnt] = 3 * pj + w
                        vals[cnt] = sg * a * nv[q] * nv[w]
                        cnt += 1
    return cnt


@dataclass
class SimResult:
    t: np.ndarray          # s
    top: np.ndarray        # m, height of the marker on top of the ball
    peak: np.ndarray       # m, highest point of the ball anywhere
    width: np.ndarray      # m, side-view width of the ball
    force: np.ndarray      # N, force between presser and ball
    presser_z: np.ndarray  # m, presser bottom height
    table_force: np.ndarray  # N, table pushing up on the ball (= weight + presser force at rest)
    pressure: np.ndarray = None   # Pa, gauge air pressure inside (hollow shell only)
    frames: np.ndarray = field(default=None, repr=False)   # (T, N, 3) positions if requested
    dt: float = 0.0


class _Stepper:
    """Holds the matrix pattern and contact constants for one ball/presser pair."""

    def __init__(self, ball: Ball, presser: Presser, params, dt: float):
        self.ball = ball
        self.dt = dt
        self.k1, self.c1, self.k2 = params.spring_arrays(ball)
        ka = float((self.k1 + self.k2).max())
        c1max = float(self.c1.max())
        shell = ball.faces is not None
        self.kc = (10.0 if shell else 50.0) * ka   # presser contact stiffness
        self.kt = (10.0 if shell else 50.0) * ka   # table contact stiffness
        # half-width of the soft edge of the presser footprint: a hard edge makes mesh points
        # flicker in and out of contact when they are as coarse as the presser
        self.hw = 0.5 * ball.spacing if shell else ball.particle_radius
        self.cc = c1max                        # contact damping
        self.ctn = c1max
        self.ctt = 50.0 * max(c1max, 0.01 * ka)   # sideways friction (effectively sticky)
        r_rest = np.hypot(ball.rest_pos[:, 0], ball.rest_pos[:, 1])
        self.cand = np.where(r_rest <= presser.radius + 2.0 * ball.spacing)[0].astype(np.int64)
        self.weight = presser.is_weight
        self.pm = max(presser.weight_mass, 1e-9)
        ndof, self.inv, self.indptr, self.indices, nnz = _pattern(ball, self.cand, self.weight)
        self.ndof = ndof
        self.data = np.zeros(nnz)
        self.f = np.zeros(ndof)
        self.Kv = np.zeros(ndof)
        self.m = ball.particle_mass
        self.rs = ball.particle_radius
        self.pr = presser.radius
        self.lu = None
        self.n_factor = 0
        # glued base: its degrees of freedom are fixed (Dirichlet). Zero their rows and
        # columns and put 1 on the diagonal, which keeps the matrix symmetric.
        self.base_dof = (3 * ball.base_idx[:, None] + np.arange(3)[None, :]).ravel()
        rows_u = np.repeat(np.arange(ndof), np.diff(self.indptr))
        is_base = np.zeros(ndof, bool)
        is_base[self.base_dof] = True
        self.fix_mask = is_base[rows_u] | is_base[self.indices]
        self.fix_diag = np.where(is_base[rows_u] & (rows_u == self.indices))[0]
        self.base_flag = np.zeros(ball.n, np.bool_)
        self.base_flag[ball.base_idx] = True
        # air inside a hollow shell
        # angle-based bending hinges (shells)
        self.hinge = ball.hinges is not None and hasattr(params, "hinge_arrays")
        if self.hinge:
            self.kb, self.cb = params.hinge_arrays(ball)
            self.hslots = hinge_slots(ball.hinges, self.indptr, self.indices)
        # shells: the presser is a HARD contact (points under it are put on its surface)
        # instead of a stiff spring, which made coarse mesh points flicker in and out of contact
        self.hard_contact = getattr(ball, "hard_contact", False) and not presser.is_weight
        # self-contact (duck v2): the shell cannot pass through itself
        self.self_dmin = float(getattr(ball, "self_contact", 0.0) or 0.0)
        self.self_rex = 2.0 * self.self_dmin
        self.n_touch = 0
        if self.self_dmin > 0.0:
            cap = 36 * 30 * ball.n
            self.sc_rows = np.zeros(cap, np.int64)
            self.sc_cols = np.zeros(cap, np.int64)
            self.sc_vals = np.zeros(cap)
        self.tau_air = getattr(params, "tau_air", None)
        self.gas = self.tau_air is not None and ball.faces is not None
        self.g = np.zeros((ball.n, 3))
        self.V0 = None
        self.air = None          # amount of air, as its volume at atmospheric pressure
        self.dp = 0.0            # gauge pressure, Pa

    def start_gas(self, x):
        """Air at atmospheric pressure in the current shape (the toy has been resting)."""
        if self.gas:
            self.V0 = _volume_grad(x, self.ball.faces, self.g)
            self.air = self.V0
            self.dp = 0.0

    def _factor(self, A):
        self.lu = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        self.n_factor += 1

    def _solve(self, A, b, u=None, tol=1e-9, max_it=12):
        """Solve (A + u u^T) x = b  (u = None: just A x = b).

        The matrix changes only a little between steps, so a sparse LU factorisation of an
        earlier A is reused as the preconditioner for conjugate gradients (the matrix is
        symmetric positive definite). The rank-one air-pressure term u u^T is folded into the
        preconditioner with the Sherman-Morrison formula, so it costs one extra back-solve
        per step instead of a second full solve. Refactor when CG needs many iterations,
        e.g. when the presser touches or leaves the ball."""
        if self.lu is None:
            self._factor(A)

        def make_prec():
            if u is None:
                return self.lu.solve
            Lu = self.lu.solve(u)
            den = 1.0 + u @ Lu
            return lambda r: (lambda y: y - Lu * ((u @ y) / den))(self.lu.solve(r))

        def op(p):
            return A @ p if u is None else A @ p + u * (u @ p)

        prec = make_prec()
        x = prec(b)
        r = b - op(x)
        bn = np.linalg.norm(b) + 1e-300
        if np.linalg.norm(r) <= tol * bn:
            return x
        z = prec(r)
        p = z.copy()
        rz = r @ z
        for _ in range(max_it):
            Ap = op(p)
            alpha = rz / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            if np.linalg.norm(r) <= tol * bn:
                return x
            z = prec(r)
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
        self._factor(A)                    # preconditioner too stale: refactor and solve exactly
        return make_prec()(b)

    def step(self, x, v, Lm, k1, c1, k2, alpha, h, vh, plate, use_gas=True):
        h_pen = 1e9 if self.hard_contact else h      # hard contact: no penalty presser in the matrix
        _assemble(x, v, self.ball.springs, self.ball.rest_len, Lm, k1, c1, k2, self.m, self.dt,
                  self.rs, self.pr, h_pen, vh, self.weight, plate[0], plate[1], self.pm,
                  self.kc, self.cc, self.kt, self.ctn, self.ctt, self.cand, self.inv,
                  self.data, self.f, self.Kv, self.hw)
        n3 = 3 * self.ball.n
        dt = self.dt
        if self.hinge:
            # light damping while settling (the caller passes reduced c1 then)
            cbs = self.cb if np.allclose(c1, self.c1) else np.minimum(self.cb, 0.01 * self.kb)
            assemble_bending(x, v, self.ball.hinges, self.ball.theta0, self.kb, cbs, self.hslots, dt,
                             self.data, self.f, self.Kv)
        gas = use_gas and self.gas and self.air is not None
        if gas:
            # Air pressure pushes every triangle outwards: force = dp * dV/dx.
            # Its stiffness is a rank-one matrix kg * g g^T (all vertices share one pressure),
            # handled implicitly with the Sherman-Morrison formula below.
            V = _volume_grad(x, self.ball.faces, self.g)
            gv = self.g.ravel().copy()
            gv[self.base_dof] = 0.0
            self.dp = P_ATM * (self.air / V - 1.0)
            kg = P_ATM * self.air / V ** 2
            self.f[:n3] += self.dp * gv
            self.Kv[:n3] += kg * gv * (gv @ v.ravel())
        base_reaction = -self.f[self.base_dof[2::3]].sum()   # table holding the glued base up
        self.data[self.fix_mask] = 0.0
        self.data[self.fix_diag] = 1.0
        A = sp.csr_matrix((self.data, self.indices, self.indptr), shape=(self.ndof, self.ndof))
        if self.self_dmin > 0.0:
            nt = _self_contact(x, v, self.ball.rest_pos, self.self_dmin, self.self_rex, self.kc, self.cc,
                               dt, self.base_flag, self.f, self.Kv, self.sc_rows, self.sc_cols, self.sc_vals)
            self.n_touch = nt // 36
            if nt:
                A = A + sp.csr_matrix((self.sc_vals[:nt], (self.sc_rows[:nt], self.sc_cols[:nt])),
                                      shape=(self.ndof, self.ndof))
        rhs = dt * (self.f - dt * self.Kv)
        rhs[self.base_dof] = 0.0
        vel = np.concatenate([v.ravel(), [plate[1]]]) if self.weight else v.ravel()
        u = None
        if gas:
            u = np.zeros(self.ndof)
            u[:n3] = dt * np.sqrt(kg) * gv
        dv = self._solve(A, rhs, u)
        vel = vel + dv
        v[:] = vel[:n3].reshape(-1, 3)
        x += dt * v
        fp_hard = 0.0
        if self.hard_contact:
            fp_hard = _hard_contact(x, v, self.rs, self.pr, h, vh, self.m, dt, self.base_flag)
        if self.weight:
            plate[1] = vel[n3]
            plate[0] += dt * plate[1]
        if k2.max() > 0.0:
            _maxwell_update(x, self.ball.springs, Lm, alpha)
        if gas:
            Vn = _volume_grad(x, self.ball.faces, self.g)
            # air through the hole: exact exponential step towards the amount that would give
            # atmospheric pressure in the current shape
            self.air = Vn + (self.air - Vn) * np.exp(-dt * self.V0 / (self.tau_air * Vn))
            self.dp = P_ATM * (self.air / Vn - 1.0)
        hz, hv = (plate[0], plate[1]) if self.weight else (h, vh)
        fp, ft = _contact_forces(x, v, self.rs, self.pr, hz, hv, self.kc, self.cc, self.kt, self.ctn, self.hw)
        if self.hard_contact:
            fp = fp_hard
        return fp, ft + base_reaction


@njit(cache=True, fastmath=True)
def _slice_top(x, edges):
    """Highest point of the surface's cross-section in the plane y = 0 (the camera's view of
    the column under the presser: it sees the top of the outline there, over all depths x)."""
    best = -1e9
    for e in range(edges.shape[0]):
        i = edges[e, 0]
        j = edges[e, 1]
        yi = x[i, 1]
        yj = x[j, 1]
        if yi == 0.0 and x[i, 2] > best:
            best = x[i, 2]
        if (yi < 0.0 < yj) or (yj < 0.0 < yi):
            t = yi / (yi - yj)
            z = x[i, 2] + t * (x[j, 2] - x[i, 2])
            if z > best:
                best = z
    return best


def _observe(ball, x):
    """(top, peak, width). 'top' is what the camera measured: by default the marker point on
    the axis; for the duck model the outline top under the presser ('slice') or the highest
    point of the duck ('peak', the top of the hat in the upright clips)."""
    rs = ball.particle_radius
    peak = x[:, 2].max() + rs
    mode = getattr(ball, "observe", "axis")
    if mode == "slice":
        top = _slice_top(x, ball.edges) + rs
    elif mode == "peak":
        top = peak
    else:
        top = x[ball.axis_idx, 2].max() + rs
    width = x[:, 0].max() - x[:, 0].min() + 2 * rs
    return top, peak, width


def simulate(ball: Ball, params, presser: Presser, t_end: float, fps: float = 60.0,
             dt: float = 2e-3, record_frames: bool = False, settle_time: float = 0.2,
             presser_from_settled: bool = False) -> SimResult:
    """Simulate the ball under the presser for t_end seconds; record at `fps` frames per second.

    presser_from_settled: the presser heights were written relative to ball.rest_top; shift
    them by however much the ball sagged under its own weight while settling, so a measured
    press depth is applied relative to the surface the camera actually saw."""
    st = _Stepper(ball, presser, params, dt)
    k1, c1, k2 = st.k1, st.c1, st.k2
    x = ball.rest_pos.copy()
    v = np.zeros_like(x)
    Lm = ball.rest_len.copy()
    plate = np.zeros(2)

    # 1) settle under gravity with nothing on top. The ball has been sitting on the table
    #    for a long time, so the Maxwell arms are relaxed (k2 off), damping is light and
    #    the air inside is at atmospheric pressure.
    if presser.is_weight:
        plate[:] = (10.0, 0.0)      # weight held far away while settling
        st_settle = _Stepper(ball, Presser(presser.radius), params, dt)
    else:
        st_settle = st
    c_settle = np.minimum(c1, 0.01 * (k1 + k2))
    for _ in range(int(settle_time / dt)):
        st_settle.step(x, v, Lm, k1, c_settle, np.zeros_like(k2), 1.0, 10.0, 0.0, plate, use_gas=False)
    v[:] = 0.0
    Lm = np.linalg.norm(x[ball.springs[:, 1]] - x[ball.springs[:, 0]], axis=1)
    st.start_gas(x)
    if presser_from_settled and not presser.is_weight:
        sag = ball.rest_top - _observe(ball, x)[0]
        presser = Presser(presser.radius, presser.key_t, presser.key_z - sag)
    if presser.is_weight:
        plate[:] = (_observe(ball, x)[1], 0.0)   # place the weight gently on the highest point

    # 2) the experiment
    alpha = maxwell_alpha(getattr(params, "tau_m", 1.0), dt)
    n_steps = int(np.ceil(t_end / dt))
    rec_every = max(1, int(round(1.0 / (fps * dt))))
    out = {k: [] for k in ("t", "top", "peak", "width", "force", "pz", "ft", "dp")}
    frames = []
    fc, ft = 0.0, ball.mass * G
    for n in range(n_steps + 1):
        t = n * dt
        if n % rec_every == 0:
            top, peak, width = _observe(ball, x)
            pz = plate[0] if presser.is_weight else float(presser.height(t))
            for key, val in zip(out, (t, top, peak, width, fc, pz, ft, st.dp)):
                out[key].append(val)
            if record_frames:
                frames.append(x.copy())
        if n == n_steps:
            break
        t1 = t + dt
        h = float(presser.height(t1)) if not presser.is_weight else 0.0
        vh = (h - float(presser.height(t))) / dt if not presser.is_weight else 0.0
        fc, ft = st.step(x, v, Lm, k1, c1, k2, alpha, h, vh, plate)
    arr = {k: np.asarray(val) for k, val in out.items()}
    return SimResult(arr["t"], arr["top"], arr["peak"], arr["width"], arr["force"], arr["pz"],
                     arr["ft"], arr["dp"], np.asarray(frames) if record_frames else None, dt)
