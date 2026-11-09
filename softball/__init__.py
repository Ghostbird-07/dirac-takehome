"""softball: a spring-mass model of a stress ball, fitted to video."""
from .geometry import Ball, make_ball
from .material import Params
from .presser import Presser, press_release, weight_on_top, from_tracked
from .sim import simulate, SimResult
from .shell import make_shell, ShellParams
