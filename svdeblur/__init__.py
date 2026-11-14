"""Spatially-varying non-blind image deblurring."""

# Thread policy must be applied before numpy/scipy are imported anywhere.
from .threads import lock_threads

lock_threads()

from .field import KernelField, INTERP_MODES, bilinear_weights, make_control_field
from .operator import SVForward, extend, fold_extend, BoundaryMode
from .solve import TVSolver, SolveConfig, SolveResult

__version__ = "1.0.0"

__all__ = [
    "KernelField", "INTERP_MODES", "bilinear_weights", "make_control_field",
    "SVForward", "extend", "fold_extend", "BoundaryMode",
    "TVSolver", "SolveConfig", "SolveResult", "__version__",
]
