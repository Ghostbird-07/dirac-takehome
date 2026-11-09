"""The squeaky duck as a HOLLOW RUBBER SHELL WITH AIR INSIDE.

Why: the videos show the duck's head caving in (the face inverts like a dented
plastic bottle), staying dented after the presser leaves, then popping back,
sometimes after a pause. A solid block of springs cannot do that. A thin shell
can, and the air inside explains the slow part: squeezing pushes air out through
the squeaker hole; on release the shell cannot re-inflate faster than air flows
back in, so the dent is held by suction until enough air returns and the shell
snaps back.

Model
-----
* Geometry: the head as a sphere of the measured size (55 mm), meshed as an
  icosphere (a sphere made of near-equal triangles). One mesh vertex sits on top
  (where the presser pushes); a small cap at the bottom is glued to the table.
* Rubber: a spring along every triangle edge (resists stretching) and a spring
  across every edge, joining the two opposite corners of its two triangles
  (resists bending). Each spring has a damper in parallel (time constant tau_v).
  Bending stiffness is what lets a shell resist caving in, and what makes it snap
  back once the dent is shallow enough.
* Air: pressure from the gas law, p = p_atm * (amount of air) / (volume), with the
  volume computed from the triangles. Air flows through the hole at a rate
  proportional to the pressure difference (a simple linear 'leak'):
        d(air)/dt = -(V0 / tau_air) * (p - p_atm) / p_atm
  tau_air is how long the hole takes to let a volume of air through.

Parameters fitted from video (ShellParams):
    Y      [N/m]  stretch stiffness of the rubber skin (edge spring constant).
                  Unlike the solid ball, this is identifiable from displacement-only
                  video, because the air pressure supplies a known force scale (p_atm).
    kappa  [-]    bending springs relative to stretch springs
    tau_v  [s]    damping time of the rubber
    tau_air [s]   air-hole time constant
"""
from dataclasses import dataclass, asdict

import numpy as np

from .geometry import Ball
from .bending import build_hinges, hinge_angle_grad

P_ATM = 101325.0


def icosphere(level: int):
    """Unit icosphere: vertices (N,3) and outward triangles (F,3)."""
    phi = (1 + 5 ** 0.5) / 2
    v = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
         (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
         (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    V = np.array(verts)
    F = np.array(faces, dtype=np.int64)
    # make sure every triangle faces outwards
    cen = V[F].mean(1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (nrm * cen).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


def _rotate_to_z(V, k):
    """Rotate so vertex k points straight up (+z)."""
    a = V[k] / np.linalg.norm(V[k])
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, z)
    s, c = np.linalg.norm(axis), a @ z
    if s < 1e-12:
        return V
    axis /= s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * K @ K
    return V @ R.T


def make_shell(radius: float = 0.0275, level: int = 3, mass: float = 0.015,
               base_height: float = 0.12, thickness: float = 0.0015, hinge_bending: bool = False) -> Ball:
    """Hollow sphere sitting on the table.

    radius      : m, head radius (duck head ~55 mm across)
    level       : icosphere refinement (3 -> 642 vertices, edges ~4 mm for this size)
    mass        : kg, mass of the rubber skin
    base_height : fraction of the diameter that forms the glued bottom cap
    thickness   : m, rubber thickness, only used as the contact offset
    """
    V, F = icosphere(level)
    V = _rotate_to_z(V, 0)
    top = int(np.argmax(V[:, 2]))
    pos = V * radius
    pos[:, 2] += radius
    # edges (stretch) and across-edge pairs (bend)
    edge_faces = {}
    for fi, (a, b, c) in enumerate(F):
        for e, opp in (((a, b), c), ((b, c), a), ((c, a), b)):
            edge_faces.setdefault((min(e), max(e)), []).append(opp)
    stretch = np.array(list(edge_faces.keys()), dtype=np.int64)
    bend = np.array([sorted(o) for o in edge_faces.values() if len(o) == 2], dtype=np.int64)
    springs = np.vstack([stretch, bend])
    kind = np.concatenate([np.zeros(len(stretch), np.int64), np.ones(len(bend), np.int64)])
    rest_len = np.linalg.norm(pos[springs[:, 1]] - pos[springs[:, 0]], axis=1)
    spacing = float(rest_len[kind == 0].mean())
    base_idx = np.where(pos[:, 2] < base_height * 2 * radius)[0]
    ball = Ball(radius, spacing, mass, pos, springs, rest_len, np.array([top]), base_idx,
                faces=F, spring_kind=kind, contact_radius=0.5 * thickness)
    # the sphere results in the report use across-edge bending springs (hinge_bending=False)
    return add_hinges(ball) if hinge_bending else ball


def add_hinges(ball: Ball) -> Ball:
    """Attach angle-based bending hinges (one per interior edge) and their rest angles."""
    H = build_hinges(ball.faces)
    g = np.zeros((4, 3))
    ball.hinges = H
    ball.theta0 = np.array([hinge_angle_grad(ball.rest_pos, h, g) for h in H])
    return ball


def even_sphere(n: int):
    """n nearly evenly spread points on the unit sphere (Fibonacci spiral) and their
    outward triangles (convex hull). Lets the mesh size be chosen freely, unlike an icosphere."""
    from scipy.spatial import ConvexHull
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    V = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
    F = ConvexHull(V).simplices.astype(np.int64)
    cen = V[F].mean(1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (nrm * cen).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


@dataclass
class ShellParams:
    Y: float = 200.0       # N/m
    kappa: float = 0.2     # -
    tau_v: float = 0.02    # s
    tau_air: float = 0.2   # s

    def spring_arrays(self, ball: Ball):
        if ball.hinges is not None:
            # bending is done by the angle hinges; the across-edge springs are kept with zero
            # stiffness only so that the solver's sparsity pattern has room for the hinges
            k = np.where(ball.spring_kind == 0, self.Y, 0.0).astype(float)
        else:
            k = np.where(ball.spring_kind == 0, self.Y, self.Y * self.kappa).astype(float)
        return k, k * self.tau_v, np.zeros_like(k)

    def hinge_arrays(self, ball: Ball):
        """Hinge bending stiffness [N m / rad^2] and damping tau_v * kb.

        v1 (ball.bend_ref unset): kb = kappa * Y * (edge length)^2. This makes big triangles
        much stiffer in bending than small ones, so the result depends on the mesh.
        v2 (ball.bend_ref = L): the discrete-shell rule (Grinspun et al. 2003), in which a
        plate of bending rigidity D gives kb = D * 3 |e|^2 / (A1 + A2) for a hinge of length
        |e| between triangles of areas A1, A2 - independent of the mesh size. Here
        D = kappa * Y * L^2 / (2 sqrt 3), chosen so that an equilateral triangle with edges L
        gets kb = kappa * Y * L^2 (the same as v1 for 5 mm triangles when L = 5 mm)."""
        X = ball.rest_pos
        H = ball.hinges
        e = np.linalg.norm(X[H[:, 3]] - X[H[:, 2]], axis=1)
        L = getattr(ball, "bend_ref", None)
        if not L:
            kb = self.kappa * self.Y * e ** 2
        else:
            def area(a, b, c):
                return 0.5 * np.linalg.norm(np.cross(X[b] - X[a], X[c] - X[a]), axis=1)
            A = area(H[:, 0], H[:, 2], H[:, 3]) + area(H[:, 1], H[:, 2], H[:, 3])
            D = self.kappa * self.Y * L ** 2 / (2 * np.sqrt(3))
            kb = D * 3 * e ** 2 / A
        return kb, self.tau_v * kb

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        return f"Y={self.Y:.4g} N/m, kappa={self.kappa:.3g}, tau_v={self.tau_v:.3g} s, tau_air={self.tau_air:.3g} s"
