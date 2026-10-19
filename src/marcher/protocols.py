"""协议层：GEM / Λ-GEM(EIT) 求解器与 Protocol 定制基类。

设计原则：物理参数进、场与物理量出；研究者在 solve() 传入网格与驱动曲线，
拿到 Result（场 + 指标）。定制新协议只需继承 Protocol 并覆写 _coefficients。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import march_system, _sample_output
from .observables import echo_fidelity, echo_peak, efficiency, window_mask


@dataclass
class Result:
    """一次求解的结果：场 + 元数据 + 便捷物理量。"""
    E: np.ndarray                     # (nz, nt) 探针场
    alpha: np.ndarray | None = None   # (nz, nt) 两能级相干（GEM）
    sigma: np.ndarray | None = None   # (2, nz, nt)（Λ-GEM: σ₁₃, σ₁₂）
    t: np.ndarray = field(default=None)
    z: np.ndarray = field(default=None)
    sw: np.ndarray = field(default=None)
    Ein: np.ndarray = field(default=None)
    t_switch: float | None = None
    meta: dict = field(default_factory=dict)

    # -- 便捷指标（lazy 计算一次后缓存） --
    def _win(self):
        return window_mask(self.t, sw=self.sw, t_switch=self.t_switch)

    @property
    def output(self) -> np.ndarray:
        """最后一个请求空间坐标 z[-1] 处的探针场 E(t)。"""
        return self.E[-1]

    @property
    def efficiency(self) -> float:
        if "efficiency" not in self.meta:
            self.meta["efficiency"] = efficiency(
                self.output, self.Ein, self.t, self._win())
        return self.meta["efficiency"]

    @property
    def echo(self) -> dict:
        if "echo" not in self.meta:
            self.meta["echo"] = echo_peak(self.output, self.t, self._win())
        return self.meta["echo"]

    def echo_fidelity(self, t_input: float | None = None, window: float | None = None) -> float:
        t_in = t_input if t_input is not None else float(
            self.t[np.argmax(np.abs(self.Ein))])
        return echo_fidelity(self.output, self.Ein, self.t,
                             self.echo["time"], t_in, self._win(), window)


class Protocol:
    """协议基类：覆写 _coefficients 与（可选）_boundary/_initial 定义自己的方程。

    求解的系统（march_system 口径）:
        ∂u_p/∂z = Σ_q A_pq·v_q + s_p      （输运场，z=0 处边界给定）
        ∂v_q/∂t = Σ_l B_ql·v_l + Σ_p C_qp·u_p + r_q   （演化场，t=0 处初始给定）
    """

    # B 中沿时间左常值保持的分量；未选中的分量保持线性插值。
    _B_step_mask = None

    # 子类覆写 -----------------------------------------------------------
    def _coefficients(self, ctx) -> tuple:
        """返回 (A, s, B, C, r) 系数张量（定义在输出网格 (nz, nt) 上）。

        ctx 提供: z, t, nz, nt, sw, curves（用户传入的曲线 dict）与协议参数。
        """
        raise NotImplementedError

    def _u_boundary(self, ctx) -> np.ndarray:
        """输运场 z=0 边界，形状 (P, nt)。默认第一输运场 = 输入曲线 Ein。"""
        return np.asarray(ctx.curves["Ein"], dtype=np.complex128)[np.newaxis, :]

    def _v_initial(self, ctx) -> np.ndarray:
        """演化场 t=0 初始，形状 (Q, nz)。默认全零（未激发介质）。"""
        return np.zeros((ctx.n_evolution, ctx.nz), dtype=np.complex128)

    @property
    def n_transport(self) -> int:
        return 1

    @property
    def n_evolution(self) -> int:
        return 1

    # 求解入口 -----------------------------------------------------------
    def solve(self, t, z, Ein, sw=None, z_factor: int = 8, t_factor: int = 8,
              t_switch: float | None = None, return_fine: bool = False,
              **curve_kwargs) -> Result:
        """求解一次。

        t, z: 严格递增网格（z 可传 L 标量则自动 linspace(0, L, 800)）。
        Ein: 输入脉冲曲线（实数/复数）。
        sw: 梯度开关曲线（None 则默认全 +1，即写入期协议）。
        **curve_kwargs: 协议专属曲线（如 Omega）与参数覆盖。
        GEM/Λ-GEM 的 sw 左常值保持，Ein/Omega 线性插值，与 XMDS 对齐。
        return_fine=True 时 Result._fine 额外保存细网格解；公共字段仍在请求网格。
        """
        t = np.asarray(t, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64) if np.ndim(z) else \
            np.linspace(0.0, float(z), 800)
        Ein = np.asarray(Ein, dtype=np.complex128)
        if sw is None:
            sw = np.ones_like(t)
        sw = np.asarray(sw, dtype=np.float64)
        curves = {"Ein": Ein, "sw": sw, **curve_kwargs}

        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.z, ctx.t = z, t
        ctx.nz, ctx.nt = len(z), len(t)
        ctx.sw, ctx.curves = sw, curves
        ctx.params = self.__dict__ | {
            k: v for k, v in curve_kwargs.items()
            if not isinstance(v, np.ndarray)
        }
        ctx.n_transport = self.n_transport
        ctx.n_evolution = self.n_evolution

        A, s, B, C, r = self._coefficients(ctx)
        out = march_system(
            z, t,
            u_bc=self._u_boundary(ctx),
            v_ic=self._v_initial(ctx),
            A_coef=A, s_coef=s, B_coef=B, C_coef=C, r_coef=r,
            z_factor=z_factor, t_factor=t_factor, return_fine=return_fine,
            B_step_mask=self._B_step_mask,
        )
        if return_fine:
            U_fine, V_fine, z_fine, t_fine = out
            U, V = _sample_output(U_fine, V_fine, z_fine, t_fine, z, t)
        else:
            U, V = out
            z_fine = t_fine = None
        res = self._pack(U, V, ctx, t_switch)
        if return_fine:
            res._fine = (U_fine, V_fine, z_fine, t_fine)
        return res

    def _pack(self, U, V, ctx, t_switch) -> Result:
        res = Result(E=U[0], t=ctx.t, z=ctx.z, sw=ctx.sw,
                     Ein=ctx.curves["Ein"], t_switch=t_switch,
                     meta={"curves": ctx.curves})
        return res


class GEM(Protocol):
    """GEM 两能级梯度回波记忆。

    方程:  dE/dz = i·N·α
           dα/dt = (γ₀ + i·sw(t)·η·(z−L/2))·α + i·g·E

    参数: g（耦合）, N（光学密度）, L（样品长度）, eta（梯度强度）,
          gamma0（沿用 XMDS 的正号系数，正值产生增长）。曲线: Ein(t), sw(t)。
    """

    _B_step_mask = ((True,),)

    def __init__(self, g: float, N: float, L: float, eta: float,
                 gamma0: float = 1e-4):
        self.g, self.N, self.L, self.eta, self.gamma0 = g, N, L, eta, gamma0

    def _coefficients(self, ctx):
        nz, nt = ctx.nz, ctx.nt
        det = (self.gamma0
               + 1j * ctx.sw[np.newaxis, :] * self.eta
               * (ctx.z[:, np.newaxis] - self.L / 2))
        A = (1j * self.N) * np.ones((1, 1, nz, nt), dtype=np.complex128)
        s = np.zeros((1, nz, nt), dtype=np.complex128)
        B = det[np.newaxis, np.newaxis, :, :]
        C = (1j * self.g) * np.ones((1, 1, nz, nt), dtype=np.complex128)
        r = np.zeros((1, nz, nt), dtype=np.complex128)
        return A, s, B, C, r

    def _pack(self, U, V, ctx, t_switch):
        res = super()._pack(U, V, ctx, t_switch)
        res.alpha = V[0]
        return res


class LambdaGEM(Protocol):
    """Λ-GEM 三能级 / EIT 存储（Hétet 2008 式 1，半经典、c→∞）。

    方程:  dE/dz = i·N·σ₁₃
           dσ₁₃/dt = −(γe + γ₀/2 + iΔ)·σ₁₃ + i·g·E + i·Ω(t)·σ₁₂
           dσ₁₂/dt = −(γ₀ + iδ(z,t))·σ₁₂ + i·Ω*(t)·σ₁₃
           δ(z,t) = δ₀ + sw(t)·η·(z−L/2)

    参数: g, N, L, eta（梯度，EIT 场景置 0）, gamma0, gamma_e（激发态衰减）,
          Delta（单光子失谐）, delta0（均匀双光子失谐偏移）。
    曲线: Ein(t), sw(t), Omega(t)（控制场 Rabi，开关脉冲）。
    """

    _B_step_mask = ((True, False), (False, True))

    def __init__(self, g: float, N: float, L: float, eta: float = 0.0,
                 gamma0: float = 1e-4, gamma_e: float = 2.0, Delta: float = 0.0):
        self.g, self.N, self.L, self.eta = g, N, L, eta
        self.gamma0, self.gamma_e, self.Delta = gamma0, gamma_e, Delta

    @property
    def n_evolution(self) -> int:
        return 2

    def _coefficients(self, ctx):
        nz, nt = ctx.nz, ctx.nt
        delta0 = float(ctx.params.get("delta0", 0.0))
        Omega = np.asarray(ctx.curves.get("Omega", np.zeros(nt)),
                           dtype=np.complex128)
        det = (delta0 + ctx.sw[np.newaxis, :] * self.eta
               * (ctx.z[:, np.newaxis] - self.L / 2))
        b00 = -(self.gamma_e + self.gamma0 / 2 + 1j * self.Delta) * np.ones((nz, nt))
        b11 = -(self.gamma0 + 1j * det)
        b01 = 1j * Omega[np.newaxis, :] * np.ones((nz, nt))
        b10 = 1j * np.conj(Omega)[np.newaxis, :] * np.ones((nz, nt))
        B = np.stack([np.stack([b00, b01]), np.stack([b10, b11])])
        A = np.zeros((1, 2, nz, nt), dtype=np.complex128)
        A[0, 0] = 1j * self.N
        C = np.zeros((2, 1, nz, nt), dtype=np.complex128)
        C[0, 0] = 1j * self.g
        s = np.zeros((1, nz, nt), dtype=np.complex128)
        r = np.zeros((2, nz, nt), dtype=np.complex128)
        return A, s, B, C, r

    def _pack(self, U, V, ctx, t_switch):
        res = super()._pack(U, V, ctx, t_switch)
        res.sigma = V
        return res
