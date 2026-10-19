"""内存友好的因果 GEM 算子：空间 Volterra 积分与二阶时间推进。

沿用标签的 +gama 符号。Ein 在原始驱动节点间线性插值，sw 左常值保持；
首驱动节点之前和末节点之后均常数延拓。只保留当前细空间状态及请求输出，
不构建稠密空间积分矩阵或细时空系数场。
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import solve_banded


def _axis(values, name):
    values = np.asarray(values)
    if np.iscomplexobj(values) and np.any(values.imag != 0):
        raise ValueError(f"{name} 必须为实值坐标")
    values = np.asarray(values.real, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or values[0] < 0 or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} 必须为非负、有限、严格递增的一维非空网格")
    return values


def _refine(axis, factor):
    if len(axis) == 1:
        return axis.copy()
    interior = (axis[:-1, None]
                + np.diff(axis)[:, None] * (np.arange(factor) / factor))
    return np.r_[interior.ravel(), axis[-1]]


def _integral(values, dz):
    """在当前空间网格上作从零开始的复值累积梯形积分。"""
    return np.r_[0j, np.cumsum(0.5 * dz * (values[:-1] + values[1:]))]


def _cn_step(alpha, integral, hz, h, diagonal, coupling, forcing, u0, u1):
    """CN 的相邻空间行相减后是下双对角系统，求解成本 O(nz)。"""
    c = 1.0 - 0.5 * h * diagonal
    q = 0.5 * h * coupling
    rhs = ((1.0 + 0.5 * h * diagonal) * alpha - q * integral
           + 0.5 * h * forcing * (u0 + u1))
    bands = np.zeros((2, len(alpha)), dtype=np.complex128)
    bands[0] = c
    bands[0, 1:] += q * hz
    bands[1, :-1] = -c[:-1] + q * hz
    rhs[1:] = np.diff(rhs)
    return solve_banded((1, 0), bands, rhs, check_finite=False)


def _exponential_weights(diagonal, h):
    """精确积分局部失谐及线性强迫；小复参数用级数避免消减误差。"""
    x = h * diagonal
    phi1 = np.empty_like(x)
    phi2 = np.empty_like(x)
    small = abs(x) < 1e-4
    y = x[small]
    phi1[small] = 1 + y * (0.5 + y * (1/6 + y * (1/24 + y/120)))
    phi2[small] = 0.5 + y * (1/6 + y * (1/24 + y * (1/120 + y/720)))
    y = x[~small]
    phi1[~small] = np.expm1(y) / y
    phi2[~small] = (np.expm1(y) - y) / (y * y)
    return np.exp(x), h * (phi1 - phi2), h * phi2


def _if2_step(alpha, integral, hz, h, diagonal, coupling, forcing, u0, u1):
    """指数拟合梯形法：局部相位精确，耦合积分强迫在区间内线性近似。

    (I + coupling*w1*Q) alpha_new = R*alpha + w0*F_old + w1*i*g*Ein_new。
    未消除耦合误差或刚性；N=0 时每段线性 Ein 的 ODE 为精确解。
    """
    growth, w0, w1 = _exponential_weights(diagonal, h)
    rhs = (growth * alpha - coupling * w0 * integral
           + forcing * (w0 * u0 + w1 * u1))
    q = coupling * w1
    # 先解累积积分 p=Q*alpha_new 的下双对角系统，再恢复 alpha_new。
    bands = np.zeros((2, len(alpha)), dtype=np.complex128)
    bands[0] = 1.0
    bands[0, 1:] += hz * q[1:]
    bands[1, :-1] = -(1.0 - hz * q[:-1])
    p_rhs = np.r_[0j, hz * (rhs[:-1] + rhs[1:])]
    p = solve_banded((1, 0), bands, p_rhs, check_finite=False)
    return rhs - q * p


def solve_gem(params, z, t, Ein, sw, *, z_factor=1, t_factor=1, method="cn",
              drive_t=None, drive_Ein=None, drive_sw=None):
    """求解当前两能级 GEM，返回请求 (z,t) 上的复值 E/alpha。

    params 使用 g、N、L、eta、gama；初始相干固定为物理 t=0 时的零值。
    method 为 cn 或 if2。z_factor 细分空间输出区间，t_factor 细分由零时刻、
    输出时刻、原始驱动节点组成的时间区间。独立驱动必须提供完整三个数组；
    此时 Ein/sw 仅作为请求网格接口参数，实际驱动以 drive_* 为准。
    允许输出从 z>0 或 t>0 开始，内部自动补零坐标。复杂 Ein 不丢弃虚部。
    """
    started = perf_counter()
    z, t = _axis(z, "z"), _axis(t, "t")
    for factor in (z_factor, t_factor):
        if not isinstance(factor, (int, np.integer)) or isinstance(factor, bool) or factor < 1:
            raise ValueError("加密倍数必须为正整数")
    if method not in ("cn", "if2"):
        raise ValueError("method 必须为 cn 或 if2")
    g, density, length, eta, gamma = (float(params[k]) for k in ("g", "N", "L", "eta", "gama"))
    if not np.isfinite([g, density, length, eta, gamma]).all() or length <= 0:
        raise ValueError("介质参数必须有限，L 必须为正")
    if z[-1] > length + 1e-12 * max(1.0, length):
        raise ValueError("输出 z 不能超出样品长度 L")

    supplied = (drive_t is not None, drive_Ein is not None, drive_sw is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("独立原始驱动需要同时提供 drive_t、drive_Ein、drive_sw")
    if not any(supplied):
        drive_t, drive_Ein, drive_sw = t, Ein, sw
    drive_t = _axis(drive_t, "drive_t")
    drive_Ein = np.asarray(drive_Ein, dtype=np.complex128)
    raw_sw = np.asarray(drive_sw)
    if np.iscomplexobj(raw_sw) and np.any(raw_sw.imag != 0):
        raise ValueError("sw 必须为实值控制")
    drive_sw = np.asarray(raw_sw.real, dtype=np.float64)
    if (drive_Ein.shape != drive_t.shape or drive_sw.shape != drive_t.shape
            or not np.isfinite(drive_Ein).all() or not np.isfinite(drive_sw).all()):
        raise ValueError("原始驱动必须与 drive_t 同形且有限")

    z_base = np.unique(np.r_[0.0, z])
    z_fine = _refine(z_base, z_factor)
    zi = np.searchsorted(z_fine, z)
    dz = np.diff(z_fine)
    hz = 0.5 * dz
    # 插入所有驱动节点，包括 Ein 的斜率变化与 sw 的精确事件端点。
    time_base = np.unique(np.r_[0.0, t, drive_t[drive_t <= t[-1]]])
    time_fine = _refine(time_base, t_factor)
    output_indices = np.searchsorted(time_fine, t)
    incident = np.interp(time_fine, drive_t, drive_Ein)
    sw_indices = np.clip(np.searchsorted(drive_t, time_fine[:-1], side="right") - 1,
                         0, len(drive_t) - 1)
    controls = drive_sw[sw_indices]
    alpha = np.zeros(len(z_fine), dtype=np.complex128)
    fields_e = np.empty((len(z), len(t)), dtype=np.complex128)
    fields_a = np.empty_like(fields_e)
    integral = _integral(alpha, dz)
    output_j = 0

    def save(j, value):
        fields_a[:, j] = alpha[zi]
        fields_e[:, j] = value + 1j * density * integral[zi]

    if output_indices[0] == 0:
        save(0, incident[0])
        output_j = 1
    step = _cn_step if method == "cn" else _if2_step
    for k, h in enumerate(np.diff(time_fine)):
        diagonal = gamma + 1j * eta * (z_fine - length / 2) * controls[k]
        alpha = step(alpha, integral, hz, h, diagonal, g * density, 1j * g,
                     incident[k], incident[k + 1])
        if not np.isfinite(alpha).all():
            raise FloatingPointError(f"{method} 在 t={time_fine[k+1]} 产生非有限状态")
        integral = _integral(alpha, dz)
        if output_j < len(t) and k + 1 == output_indices[output_j]:
            save(output_j, incident[k + 1])
            output_j += 1
    return {"E": fields_e, "alpha": fields_a, "t": t.copy(), "z": z.copy(),
            "elapsed_seconds": perf_counter() - started}
