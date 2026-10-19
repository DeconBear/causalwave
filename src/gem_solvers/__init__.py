"""无需训练、无需GPU的GEM数值求解器。"""
from .api import GEMParameters, Drive, GEMResult, solve_gem

__version__ = "0.1.0"
__all__ = ["GEMParameters", "Drive", "GEMResult", "solve_gem"]
