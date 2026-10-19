"""物理量后处理：量子光存储研究者关心的观测指标。

口径约定：
- 效率 = 读出窗口内 z=L 处 |E|² 的时间积分 / 输入脉冲 |Ein|² 的时间积分
- 回波峰 = z=L 处 E(t) 在读出窗口（sw<0 或显式 t>t_switch）内的最大值
- 保真度 = 回波与输入脉冲时间反演（相对回波时刻对称）的归一化内积
- 透射谱 = 稳态 T(δ) 扫描（EIT 类协议）
"""
from __future__ import annotations

import numpy as np


def window_mask(t: np.ndarray, sw: np.ndarray | None = None,
                t_switch: float | None = None, t_window: tuple[float, float] | None = None):
    """读出窗口掩码。优先 t_window，其次 t_switch，最后 sw<0。"""
    if t_window is not None:
        return (t >= t_window[0]) & (t <= t_window[1])
    if t_switch is not None:
        return t >= t_switch
    if sw is not None:
        return sw < 0
    raise ValueError("需要 sw、t_switch 或 t_window 之一")


def efficiency(E_out: np.ndarray, Ein: np.ndarray, t: np.ndarray,
               mask: np.ndarray | None = None) -> float:
    """存储/读出效率：读出窗内输出能量 / 输入能量。"""
    mask = mask if mask is not None else np.ones(len(t), dtype=bool)
    if mask.sum() == 0:
        return 0.0
    num = float(np.trapezoid(np.abs(E_out[mask]) ** 2, t[mask]))
    den = float(np.trapezoid(np.abs(Ein) ** 2, t))
    return num / max(den, 1e-30)


def echo_peak(E_out: np.ndarray, t: np.ndarray, mask: np.ndarray) -> dict:
    """回波峰值与时刻（z=L 输出端，读出窗口内）。"""
    m = np.abs(E_out[mask])
    if len(m) == 0:
        return {"peak": 0.0, "time": float("nan")}
    k = int(np.argmax(m))
    return {"peak": float(m[k]), "time": float(t[mask][k])}


def echo_fidelity(E_out: np.ndarray, Ein: np.ndarray, t: np.ndarray,
                  t_echo: float, t_input: float, mask: np.ndarray | None = None,
                  window: float | None = None) -> float:
    """回波复值重叠：与 Ein(t_input+t_echo-t) 的归一化内积模。

    时间反演对称中心 = (t_input + t_echo)/2（GEM 协议回波是输入的镜像）。
    保留既有内积模口径（不平方），保留复相位但不进行额外复共轭时间反演。
    """
    mask = mask if mask is not None else np.ones(len(t), dtype=bool)
    t_rev = t_input + t_echo - t
    Ein_rev = np.interp(t_rev, t, Ein, left=0, right=0)
    a = E_out[mask]
    b = Ein_rev[mask]
    if window is not None:
        sel = np.abs(t[mask] - t_echo) <= window
        a, b = a[sel], b[sel]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.abs(np.vdot(a, b)) / (na * nb))


def transmission_spectrum(solver, delta0s, **solve_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """稳态透射谱 T(δ₀)：对均匀双光子失谐扫描，取长时稳态透射。

    注意：需要 T ≫ 稳态建立时间（透明窗内为慢光渡越时间 N·g·L/Ω²，
    窗外为 1/γ₀），否则返回的是瞬态值。返回 (delta0s, T)。
    """
    Ts = []
    for d0 in delta0s:
        res = solver.solve(delta0=d0, **solve_kwargs)
        Ts.append(float(np.abs(res.E[-1, -1])))
    return np.asarray(delta0s), np.asarray(Ts)
