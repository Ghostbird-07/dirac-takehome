"""A duck-shaped hollow rubber shell (instead of a plain sphere).

Shape: smooth blend of three rounded parts, sized from the standing reference clip
(scale from the measured 110 mm height, 7.7 px/mm) and the lying clip (head depth):

    part   centre (x, y, z) mm      half-sizes (x, y, z) mm     from the video
    body   (0, 0, 24)               (24, 20.5, 25)              48 wide, 41 deep, ~45 tall (overlaps the head)
    head   (0, 0, 70)               (25.5, 26, 27.5)            51 wide, 52 deep, 55 tall
    hat    (0, -3, 103)             (21, 21, 7)                 42 across, ~14 thick, set back

(z up, the face looks along +y.) Feet, arms, beak and bow are left out: they are small
solid bumps that do not change how the hollow body squashes.

The surface is the level set of a smooth union of the three ellipsoids. v1 meshed it by
shooting a ray from a point in the head through evenly spread directions and placing a
vertex where each ray leaves the duck. That crowds the head and starves the body (301 of 380
points on the head, 13-22 mm triangles on the body), and the coarse body flopped about.
v2 (DUCK_V2, see even_duck_surface) spreads 900 points evenly over the surface instead.

Two poses, as filmed:
  'lying'   duck on its back, face up (clips A): the nozzle presses the middle of the face.
            The parts resting on the table are glued.
  'upright' duck standing (clips B): the bottle presses the hat. The bottom of the body,
            held by the hand and resting on the table, is glued.
"""
import numpy as np

from .geometry import Ball
from .shell import icosphere, even_sphere, add_hinges

PARTS = {                      # name: (centre mm, half-sizes mm)
    "body": (np.array([0.0, 0.0, 24.0]), np.array([24.0, 20.5, 25.0])),
    "head": (np.array([0.0, 0.0, 70.0]), np.array([25.5, 26.0, 27.5])),
    "hat": (np.array([0.0, -3.0, 103.0]), np.array([21.0, 21.0, 7.0])),
}
PART_COLOUR = {"body": (0.93, 0.87, 0.74), "head": (0.96, 0.91, 0.80), "hat": (0.30, 0.38, 0.75)}


def _field(p, k=0.25):
    """< 0 inside the duck, > 0 outside (smooth union of ellipsoids, in mm)."""
    vals = []
    for c, a in PARTS.values():
        q = (p - c) / a
        vals.append(np.linalg.norm(q, axis=-1) - 1.0)
    v = np.stack(vals, -1)
    return -k * np.log(np.exp(-v / k).sum(-1))


def _part_of(p):
    vals = np.stack([np.linalg.norm((p - c) / a, axis=-1) for c, a in PARTS.values()], -1)
    names = list(PARTS)
    return np.array([names[i] for i in vals.argmin(-1)])


def duck_surface(n_points=None, level=3, centre=np.array([0.0, 0.0, 64.0])):
    """Vertices (mm, standing pose) and outward triangles of the duck surface.
    n_points: use an evenly spread mesh of this many points (else an icosphere of `level`)."""
    V, F = even_sphere(n_points) if n_points else icosphere(level)
    r = np.linspace(120.0, 0.0, 241)
    pts = centre[None, None, :] + r[None, :, None] * V[:, None, :]          # (N, R, 3)
    inside = _field(pts) < 0
    first = inside.argmax(1)                       # first radius (from outside) that is inside
    r_lo = r[first]
    r_hi = r[np.maximum(first - 1, 0)]
    for _ in range(40):                            # bisection, all rays at once
        rm = 0.5 * (r_hi + r_lo)
        ins = _field(centre + rm[:, None] * V) < 0
        r_lo = np.where(ins, rm, r_lo)
        r_hi = np.where(ins, r_hi, rm)
    return centre + (0.5 * (r_hi + r_lo))[:, None] * V, F


def even_duck_surface(n, centre=np.array([0.0, 0.0, 64.0]), n_dense=40000, lloyd=4):
    """n points spread EVENLY over the duck's surface (same spacing on body, head and hat),
    and outward triangles. duck_surface() spreads its points evenly over ray DIRECTIONS from a
    point in the head, which crowds the head (near) and starves the body (far): 301 of 380
    points on the head, 13-22 mm triangles on the body.

    1. sample the surface densely (rays), 2. pick n well-separated samples (farthest-point
    sampling), 3. even them out with a few Lloyd steps (each point moves to the middle of the
    samples nearest to it), 4. triangulate. The surface is star-shaped about `centre` (every
    ray leaves it once), so the convex hull of the points' directions from `centre` gives a
    valid triangulation of the surface."""
    from scipy.spatial import ConvexHull, cKDTree
    D, _ = duck_surface(n_dense, centre=centre)
    sel = np.zeros(n, np.int64)
    d = np.full(len(D), np.inf)
    sel[0] = np.argmax(D[:, 2])
    for k in range(1, n):
        d = np.minimum(d, np.linalg.norm(D - D[sel[k - 1]], axis=1))
        sel[k] = np.argmax(d)
    P = D[sel]
    tree_d = cKDTree(D)
    for _ in range(lloyd):
        lab = cKDTree(P).query(D)[1]
        cen = np.zeros_like(P)
        np.add.at(cen, lab, D)
        cnt = np.bincount(lab, minlength=n)[:, None]
        cen = np.where(cnt > 0, cen / np.maximum(cnt, 1), P)
        P = D[tree_d.query(cen)[1]]          # back onto the surface
    P = np.unique(P, axis=0)
    U = (P - centre) / np.linalg.norm(P - centre, axis=1, keepdims=True)
    F = ConvexHull(U).simplices.astype(np.int64)
    nrm = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = (nrm * (P[F].mean(1) - centre)).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return P, F


# settings of the second version of the duck model (see README section 5b)
DUCK_V2 = dict(n_points=900, even_mesh=True, self_contact="auto", bend_ref=0.005)


def make_duck(pose="lying", n_points=380, level=3, mass=0.04, thickness=0.0015, glue_mm=None,
              self_contact=None, even_mesh=False, bend_ref=None) -> Ball:
    """Duck shell in metres, posed as filmed, with the presser point at x = y = 0.
    Bending uses angle hinges and the presser is a hard contact (see bending.py, sim.py).
    self_contact: minimum distance (m) kept between parts of the shell that were far apart
    at rest, so the face cannot pass through the back of the head (None = off, as in v1;
    "auto" = 0.6 x the mean edge length, the smallest gap a point cannot slip through).
    even_mesh: spread the points evenly over the surface (v2) instead of over ray directions.
    bend_ref: if set, bending stiffness per hinge does not depend on the mesh (v2), see
    ShellParams.hinge_arrays."""
    P, F = even_duck_surface(n_points) if even_mesh else duck_surface(n_points, level)
    part = _part_of(P)
    if pose == "lying":
        # on its back: the face (+y) points up. Rotate -90 deg about x: (x, y, z) -> (x, -z, y)
        P = np.column_stack([P[:, 0], -P[:, 2], P[:, 1]])
        # press point: the top of the face, at the height of the head centre
        head_c = -PARTS["head"][0][2]
        P[:, 1] -= head_c
        glue = 4.0 if glue_mm is None else glue_mm
    elif pose == "upright":
        # press point: the top of the hat
        top = np.argmax(P[:, 2])
        P[:, 0] -= P[top, 0]
        P[:, 1] -= P[top, 1]
        glue = 14.0 if glue_mm is None else glue_mm
    else:
        raise ValueError(pose)
    P[:, 2] -= P[:, 2].min()
    pos = P / 1000.0
    # springs: triangle edges (stretch) and across-edge pairs (bend)
    edge_faces = {}
    for a, b, c in F:
        for e, opp in (((a, b), c), ((b, c), a), ((c, a), b)):
            edge_faces.setdefault((min(e), max(e)), []).append(opp)
    stretch = np.array(list(edge_faces.keys()), dtype=np.int64)
    bend = np.array([sorted(o) for o in edge_faces.values() if len(o) == 2], dtype=np.int64)
    springs = np.vstack([stretch, bend])
    kind = np.concatenate([np.zeros(len(stretch), np.int64), np.ones(len(bend), np.int64)])
    rest_len = np.linalg.norm(pos[springs[:, 1]] - pos[springs[:, 0]], axis=1)
    spacing = float(rest_len[kind == 0].mean())
    r_xy = np.hypot(pos[:, 0], pos[:, 1])
    near = np.where(r_xy < 1.2 * spacing)[0]
    top = near[np.argmax(pos[near, 2])]
    base_idx = np.where(pos[:, 2] < glue / 1000.0)[0]
    ball = Ball(0.055, spacing, mass, pos, springs, rest_len, np.array([top]), base_idx,
                faces=F, spring_kind=kind, contact_radius=0.5 * thickness)
    ball.part = part
    ball.pose = pose
    ball.hard_contact = True
    # what the camera measured: lying = top of the outline in the column under the nozzle;
    # upright = the highest point (top of the hat)
    ball.observe = "slice" if pose == "lying" else "peak"
    ball.edges = stretch
    ball.self_contact = 0.6 * spacing if self_contact == "auto" else self_contact
    ball.bend_ref = bend_ref
    return add_hinges(ball)
