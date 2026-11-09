"""Geometry: the stress ball as a lattice of point masses joined by springs.

The ball is a sphere filled with particles on a cubic grid. Every particle is
joined to its 26 neighbours (face, edge and corner neighbours of the grid
cell), so the lattice resists stretching, compression and shear.

This is a deliberately rough proxy for the real ball: the assignment does not
assess 3D reconstruction, and a sphere of the measured diameter is enough to
model pressing on the top.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Ball:
    radius: float          # m, radius of the real ball
    spacing: float         # m, distance between neighbouring particles
    mass: float            # kg, total mass of the ball
    rest_pos: np.ndarray   # (N, 3) particle positions with no load, z up, table at z = 0
    springs: np.ndarray    # (S, 2) particle index pairs
    rest_len: np.ndarray   # (S,) spring rest lengths
    axis_idx: np.ndarray   # particles on the vertical centre line (the top one carries the marker)
    base_idx: np.ndarray   # bottom layer: the flat contact patch, glued to the table
    # hollow-shell extras (None for the solid lattice)
    faces: np.ndarray = None          # (F, 3) outward-oriented surface triangles (for the enclosed volume)
    spring_kind: np.ndarray = None    # (S,) 0 = stretch (triangle edge), 1 = bend (across an edge)
    contact_radius: float = None      # surface offset used for contacts; default half the spacing
    hinges: np.ndarray = None         # (H, 4) bending hinges [wing, wing, edge, edge] (shells)
    theta0: np.ndarray = None         # (H,) rest bending angles

    @property
    def n(self) -> int:
        return len(self.rest_pos)

    @property
    def particle_radius(self) -> float:
        """Each particle stands for a small cube of foam; half the spacing is its 'radius'.
        For a thin shell it is half the rubber thickness instead."""
        return self.contact_radius if self.contact_radius is not None else 0.5 * self.spacing

    @property
    def particle_mass(self) -> float:
        return self.mass / self.n

    @property
    def rest_top(self) -> float:
        """Height of the top surface (marker) with no load, before gravity sag."""
        return self.rest_pos[self.axis_idx, 2].max() + self.particle_radius


def make_ball(radius: float = 0.035, n_across: int = 9, mass: float = 0.03) -> Ball:
    """Build a spherical particle lattice.

    radius   : ball radius in metres (measure the real ball's diameter / 2)
    n_across : particles across the diameter; odd so there is a centre column
    mass     : total mass in kg (weigh the real ball)
    """
    if n_across % 2 == 0:
        raise ValueError("n_across must be odd so a column of particles sits on the axis")
    half = n_across // 2
    idx = np.arange(-half, half + 1)
    grid = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1).reshape(-1, 3)
    # keep grid cells whose little cube lies inside the sphere (grid units: spacing = 1)
    r_cut = n_across / 2.0 - 0.5
    grid = grid[np.linalg.norm(grid, axis=1) <= r_cut + 1e-6]

    # Remove lone "pole" particles. Filling a sphere with cubes leaves a single particle
    # sticking out one spacing beyond its neighbours at the top, bottom and four sides.
    #  * At the bottom it made the ball balance on a point like a pencil on its tip, and it
    #    slowly tipped over (an inverted-pendulum instability).
    #  * At the top it formed a soft 'nipple': a press first squashed that one particle and
    #    then suddenly hit the full layer below, giving fake force spikes.
    # A real ball has neither. Removing every particle that has fewer than 10 neighbours
    # leaves a flat contact patch at the bottom (glued to the table, like a ball flattening
    # under its own weight) and a flat cap on top whose centre carries the marker.
    while True:
        tree = cKDTree(grid.astype(float))
        deg = np.array([len(nb) - 1 for nb in tree.query_ball_point(grid, np.sqrt(3.0) * 1.01)])
        lone = deg < 10
        if not lone.any():
            break
        grid = grid[~lone]

    # Scale so the model is exactly as tall as the real ball (height is what the press acts
    # on and what the camera measures). The lattice is symmetric, so the width matches too.
    layers = grid[:, 2].max() - grid[:, 2].min() + 1
    a = 2.0 * radius / layers
    pos = grid * a
    # sit the ball on the table: bottom layer surface touches z = 0
    pos[:, 2] += -pos[:, 2].min() + 0.5 * a

    tree = cKDTree(pos)
    pairs = np.array(sorted(tree.query_pairs(np.sqrt(3.0) * a * 1.01)), dtype=np.int64)
    rest_len = np.linalg.norm(pos[pairs[:, 1]] - pos[pairs[:, 0]], axis=1)

    axis_idx = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0))[0]
    base_idx = np.where(grid[:, 2] == grid[:, 2].min())[0]
    return Ball(radius, a, mass, pos, pairs, rest_len, axis_idx, base_idx)
