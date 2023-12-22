"""
Hypercube: a DOE-informed hyperparameter optimization machine
"""

from .legacy.base_ofat import OFAT
from .legacy.base_surf import Surf
from .legacy.base_LHS import LHSTuner
from .legacy.base_cube import cube
from .config.space import unif, log10unif, logEunif

__all__ = [
    "OFAT",
    "Surf",
    "LHSTuner",
    "cube"
]