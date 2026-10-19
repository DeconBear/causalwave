"""线性耦合传播的因果推进与GEM专用数值接口，无需训练或GPU。"""
from marcher import march_system
from .api import GEMParameters, Drive, GEMResult, solve_gem

__version__ = "0.2.0"
__all__ = ["march_system", "GEMParameters", "Drive", "GEMResult", "solve_gem"]
