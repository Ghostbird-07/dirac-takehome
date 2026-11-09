"""Bending of a thin shell by the angle between neighbouring triangles ("hinges").

Each interior edge of the mesh is a hinge shared by two triangles. Its bending angle
theta is the signed angle between the two triangles' normals; the energy is

        E = 1/2 * kb * (theta - theta0)^2

where theta0 is the angle in the rest shape. Unlike a spring joining the two opposite
corners, the angle knows WHICH WAY the hinge is folded, so a region that has been
pushed inside out wants to flip back.

Gradient of theta (Bridson, Marino & Fedkiw 2003, "Simulation of clothing with folds
and wrinkles"), with wing vertices x1, x2 and edge vertices x3, x4:
    E  = x4 - x3,  N1 = (x1 - x3) x (x1 - x4),  N2 = (x2 - x4) x (x2 - x3)
    d theta / d x1 = |E| N1 / |N1|^2
    d theta / d x2 = |E| N2 / |N2|^2
    d theta / d x3 = ((x1-x4).E/|E|) N1/|N1|^2 + ((x2-x4).E/|E|) N2/|N2|^2
    d theta / d x4 = -((x1-x3).E/|E|) N1/|N1|^2 - ((x2-x3).E/|E|) N2/|N2|^2

For the implicit solver the stiffness matrix of a hinge is approximated by its
always-positive part  kb * g g^T  (g = the 12-vector gradient), which keeps the
system matrix symmetric positive definite. A damper along the same direction
(cb * g g^T) gives the rubber's internal friction in bending.
"""
import numpy as np
from numba import njit


def build_hinges(faces):
    """(H, 4) hinges [x1 wing, x2 wing, x3 edge, x4 edge] for every edge shared by two triangles.
    Orientation is taken from the first triangle so that theta has a consistent sign."""
    edge = {}
    for fi, (a, b, c) in enumerate(faces):
        for (p, q, opp) in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(p, q), max(p, q))
            edge.setdefault(key, []).append((p, q, opp))
    H = []
    for key, lst in edge.items():
        if len(lst) != 2:
            continue
        (p, q, w1), (_, _, w2) = lst
        H.append((w1, w2, p, q))
    return np.array(H, dtype=np.int64)


@njit(cache=True, fastmath=True)
def _cross(a0, a1, a2, b0, b1, b2):
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


@njit(cache=True, fastmath=True)
def hinge_angle_grad(x, h, g):
    """Signed angle of hinge h and its gradient written into g (4, 3). Returns theta."""
    i1, i2, i3, i4 = h[0], h[1], h[2], h[3]
    E0 = x[i4, 0] - x[i3, 0]; E1 = x[i4, 1] - x[i3, 1]; E2 = x[i4, 2] - x[i3, 2]
    el = np.sqrt(E0 * E0 + E1 * E1 + E2 * E2)
    a0 = x[i1, 0] - x[i3, 0]; a1 = x[i1, 1] - x[i3, 1]; a2 = x[i1, 2] - x[i3, 2]
    b0 = x[i1, 0] - x[i4, 0]; b1 = x[i1, 1] - x[i4, 1]; b2 = x[i1, 2] - x[i4, 2]
    c0 = x[i2, 0] - x[i4, 0]; c1 = x[i2, 1] - x[i4, 1]; c2 = x[i2, 2] - x[i4, 2]
    d0 = x[i2, 0] - x[i3, 0]; d1 = x[i2, 1] - x[i3, 1]; d2 = x[i2, 2] - x[i3, 2]
    N10, N11, N12 = _cross(a0, a1, a2, b0, b1, b2)
    N20, N21, N22 = _cross(c0, c1, c2, d0, d1, d2)
    n1s = N10 * N10 + N11 * N11 + N12 * N12 + 1e-30
    n2s = N20 * N20 + N21 * N21 + N22 * N22 + 1e-30
    n1 = np.sqrt(n1s); n2 = np.sqrt(n2s)
    # signed angle between unit normals, about the edge direction
    cs = (N10 * N20 + N11 * N21 + N12 * N22) / (n1 * n2)
    X0, X1, X2 = _cross(N10, N11, N12, N20, N21, N22)
    sn = (X0 * E0 + X1 * E1 + X2 * E2) / (n1 * n2 * el)
    theta = np.arctan2(-sn, cs)       # sign chosen to match the gradient formulas (checked numerically)
    for k in range(3):
        g[0, k] = 0.0
    g[0, 0] = el * N10 / n1s; g[0, 1] = el * N11 / n1s; g[0, 2] = el * N12 / n1s
    g[1, 0] = el * N20 / n2s; g[1, 1] = el * N21 / n2s; g[1, 2] = el * N22 / n2s
    s14 = (b0 * E0 + b1 * E1 + b2 * E2) / el      # (x1-x4).E/|E|
    s24 = (c0 * E0 + c1 * E1 + c2 * E2) / el      # (x2-x4).E/|E|
    s13 = (a0 * E0 + a1 * E1 + a2 * E2) / el      # (x1-x3).E/|E|
    s23 = (d0 * E0 + d1 * E1 + d2 * E2) / el      # (x2-x3).E/|E|
    g[2, 0] = s14 * N10 / n1s + s24 * N20 / n2s
    g[2, 1] = s14 * N11 / n1s + s24 * N21 / n2s
    g[2, 2] = s14 * N12 / n1s + s24 * N22 / n2s
    g[3, 0] = -s13 * N10 / n1s - s23 * N20 / n2s
    g[3, 1] = -s13 * N11 / n1s - s23 * N21 / n2s
    g[3, 2] = -s13 * N12 / n1s - s23 * N22 / n2s
    return theta


@njit(cache=True, fastmath=True)
def assemble_bending(x, v, hinges, theta0, kb, cb, slots, dt, data, f, Kv):
    """Add hinge forces to f, K*v to Kv and (dt*cb + dt^2*kb) g g^T to the CSR values.
    slots[h] lists the CSR positions of the 4x4 blocks (row-major, 144 entries)."""
    g = np.zeros((4, 3))
    for h in range(hinges.shape[0]):
        th = hinge_angle_grad(x, hinges[h], g)
        d = th - theta0[h]
        if d > np.pi:
            d -= 2 * np.pi
        elif d < -np.pi:
            d += 2 * np.pi
        gv = 0.0
        for a in range(4):
            vi = hinges[h, a]
            gv += g[a, 0] * v[vi, 0] + g[a, 1] * v[vi, 1] + g[a, 2] * v[vi, 2]
        F = kb[h] * d + cb[h] * gv
        for a in range(4):
            vi = hinges[h, a]
            for k in range(3):
                f[3 * vi + k] -= F * g[a, k]
                Kv[3 * vi + k] += kb[h] * gv * g[a, k]
        w = dt * cb[h] + dt * dt * kb[h]
        # order matches hinge_slots: (block a, block b, k, l)
        q = 0
        for a in range(4):
            for b in range(4):
                for k in range(3):
                    for l in range(3):
                        data[slots[h, q]] += w * g[a, k] * g[b, l]
                        q += 1


def hinge_slots(hinges, indptr, indices):
    """CSR positions for every hinge's 4x4 block structure (order: a, b, k, l)."""
    H = len(hinges)
    out = np.empty((H, 144), dtype=np.int64)
    for h in range(H):
        q = 0
        for a in range(4):
            for b in range(4):
                for k in range(3):
                    r = 3 * hinges[h, a] + k
                    row = indices[indptr[r]:indptr[r + 1]]
                    for l in range(3):
                        c = 3 * hinges[h, b] + l
                        pos = np.searchsorted(row, c)
                        if pos >= len(row) or row[pos] != c:
                            raise ValueError("hinge entry missing from the sparsity pattern")
                        out[h, q] = indptr[r] + pos
                        q += 1
    return out
