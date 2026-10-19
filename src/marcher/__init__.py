"""单向空间输运与时间演化的对角线梯形推进；精度须按具体问题验收。"""
from .core import march_system
from .protocols import GEM, LambdaGEM, Protocol, Result

__version__ = "0.1.0"
