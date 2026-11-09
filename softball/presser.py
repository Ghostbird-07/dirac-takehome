"""The thing that presses on the ball: a rigid flat-bottomed cylinder (fingertip,
bottle cap or lid) centred above the ball.

Two modes:

* Kinematic (a finger): its bottom height over time is GIVEN, for example
  measured from the video. This is the input to the model; the ball's shape is
  the output.
* Weight: a free plate of known mass rests on the ball and sinks under
  gravity until the ball holds it up. This is the one experiment that measures
  the force scale (stiffness in newtons), because the force (m * g) is known.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Presser:
    radius: float                                    # m, contact radius of the flat bottom
    key_t: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    key_z: np.ndarray = field(default_factory=lambda: np.array([1.0]))  # m, bottom height at key_t
    weight_mass: float = 0.0                         # kg; > 0 switches to weight mode
    weight_start_z: float = 0.0                      # m, where the weight is placed (bottom height)

    @property
    def is_weight(self) -> bool:
        return self.weight_mass > 0

    def height(self, t: np.ndarray) -> np.ndarray:
        """Bottom height at times t (kinematic mode); held constant after the last key."""
        return np.interp(t, self.key_t, self.key_z)


def press_release(top0: float, depth: float, t_hold: float, radius: float = 0.008,
                  t_start: float = 0.3, t_down: float = 0.35, t_up: float = 0.04,
                  lift: float = 0.04) -> Presser:
    """A synthetic press: approach, push `depth` below the resting top, hold, lift away fast.

    top0 is the resting top height of the ball. Heights in metres, times in seconds.
    """
    t1 = t_start                  # touches the top
    t2 = t1 + t_down              # fully pressed
    t3 = t2 + t_hold              # starts lifting
    t4 = t3 + t_up                # clear of the ball
    key_t = np.array([0.0, t1, t2, t3, t4])
    key_z = np.array([top0 + 0.005, top0, top0 - depth, top0 - depth, top0 + lift])
    return Presser(radius, key_t, key_z)


def weight_on_top(top0: float, mass: float, radius: float = 0.03) -> Presser:
    """A flat plate of `mass` kg placed gently on the top at t = 0 and left there."""
    return Presser(radius, weight_mass=mass, weight_start_z=top0)


def from_tracked(t: np.ndarray, z: np.ndarray, release_t: float, radius: float,
                 lift_speed: float = 1.0, lift: float = 0.04) -> Presser:
    """Presser path taken from a video.

    t, z      : times and presser-bottom heights while it was in contact
                (during contact the presser bottom IS the ball's top surface)
    release_t : time the presser leaves the ball
    After release the presser moves straight up at lift_speed (m/s), out of the way.
    """
    m = t <= release_t
    tt, zz = list(t[m]), list(z[m])
    z_rel = float(np.interp(release_t, t, z))
    tt += [release_t, release_t + lift / lift_speed]
    zz += [z_rel, z_rel + lift]
    order = np.argsort(tt, kind="stable")
    return Presser(radius, np.asarray(tt)[order], np.asarray(zz)[order])
