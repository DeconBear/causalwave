"""两个数值后端的统一GEM接口，驱动采样与输出网格分离。"""
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from ._if2 import solve_gem_cached
from ._if2_reference import _axis


@dataclass(frozen=True)
class GEMParameters:
    """有符号gamma进入+gamma*alpha；物理衰减率Gamma应传gamma=-Gamma。"""
    g: float = 2.0
    N: float = 50.0
    L: float = 2.0
    eta: float = 100.0
    gamma: float = 0.0

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or self.L <= 0:
            raise ValueError("参数必须有限，L必须为正")

    def _legacy(self):
        p = asdict(self)
        p["gama"] = p.pop("gamma")
        return p


@dataclass(frozen=True)
class Drive:
    """从t=0开始：Ein复数分段线性，sw实数左常值保持。"""
    t: np.ndarray
    Ein: np.ndarray
    sw: np.ndarray

    def __post_init__(self):
        t = _axis(self.t, "drive.t").copy()
        u = np.asarray(self.Ein, dtype=np.complex128).copy()
        s = np.asarray(self.sw)
        if t[0] != 0 or len(t) < 2:
            raise ValueError("驱动必须从0开始且至少两个节点")
        if np.iscomplexobj(s) and np.any(s.imag != 0):
            raise ValueError("sw必须实值")
        s = np.asarray(s.real, dtype=float).copy()
        if u.shape != t.shape or s.shape != t.shape or not np.isfinite(u).all() or not np.isfinite(s).all():
            raise ValueError("Ein/sw必须与驱动t同形且有限")
        for name, array in (("t", t), ("Ein", u), ("sw", s)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass
class GEMResult:
    """两个完整复场形状均为(nz, nt)，坐标保持用户请求值。"""
    E: np.ndarray
    alpha: np.ndarray
    z: np.ndarray
    t: np.ndarray
    method: str
    elapsed_seconds: float

    def save(self, path):
        """保存NPZ；文件存在时拒绝覆盖。"""
        with open(path, "xb") as stream:
            np.savez_compressed(stream, E=self.E, alpha=self.alpha, z=self.z, t=self.t,
                                method=self.method, elapsed_seconds=self.elapsed_seconds)


def solve_gem(params: GEMParameters, z, t, drive: Drive, *, method="if2", z_factor=4, t_factor=2):
    """在请求坐标求解；默认加密仅为起点，不是误差保证。"""
    start = perf_counter()
    z, t = _axis(z, "z"), _axis(t, "t")
    if len(z) < 2 or z[0] != 0 or len(t) < 2 or z[-1] > params.L:
        raise ValueError("统一接口要求z至少两点且从0至不超过L，t至少两点")
    if drive.t[-1] < t[-1]:
        raise ValueError("驱动必须覆盖输出终点，不静默向未来延拓")
    for factor in (z_factor, t_factor):
        if not isinstance(factor, (int, np.integer)) or isinstance(factor, bool) or factor < 1:
            raise ValueError("加密倍数必须为正整数")
    if method == "if2":
        out = solve_gem_cached(params._legacy(), z, t, None, None,
                              z_factor=z_factor, t_factor=t_factor, drive_t=drive.t,
                              drive_Ein=drive.Ein, drive_sw=drive.sw)
        e, a = out["E"], out["alpha"]
    elif method == "marcher":
        from marcher import GEM
        timeline = np.unique(np.r_[drive.t[drive.t <= t[-1]], t])
        u = np.interp(timeline, drive.t, drive.Ein)
        control = drive.sw[np.searchsorted(drive.t, timeline, side="right") - 1]
        sim = GEM(g=params.g, N=params.N, L=params.L, eta=params.eta, gamma0=params.gamma)
        out = sim.solve(timeline, z, u, control, z_factor=z_factor, t_factor=t_factor)
        ids = np.searchsorted(timeline, t)
        e, a = out.E[:, ids], out.alpha[:, ids]
    else:
        raise ValueError("method仅支持if2或marcher")
    if not np.isfinite(e).all() or not np.isfinite(a).all():
        raise FloatingPointError("求解器产生非有限复场；请检查参数和离散分辨率")
    return GEMResult(e, a, z.copy(), t.copy(), method, perf_counter() - start)
