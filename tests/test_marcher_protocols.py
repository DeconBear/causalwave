"""pytest 验证用例：marcher 库的正确性（轻量版，秒级）。

独立解析验证见test_marcher_correctness.py；
本套件固化可在 CI 上跑的方程级检验。
"""
from __future__ import annotations

import numpy as np
import pytest

import marcher
from marcher import GEM, LambdaGEM


def _grid():
    t = np.linspace(0.025, 10.0, 400)
    z = np.linspace(0.0, 2.0, 800)
    return t, z




def test_t0_column_maxwell_zero_alpha():
    """α(z,0)=0 ⇒ ∂E/∂z=0 ⇒ 细网格 t=0 列 E(z,0) 应等于 Ein(0)。"""
    t = np.linspace(0.025, 10.0, 40)
    z = np.linspace(0.0, 2.0, 32)
    Ein = np.full_like(t, 0.5)
    sw = np.tanh((t - 5.0) / 0.1)
    res = GEM(g=2.0, N=10.0, L=2.0, eta=20.0, gamma0=0.0).solve(
        t, z, Ein, sw, z_factor=4, t_factor=4, return_fine=True)
    U, V, z_fine, t_fine = res._fine
    assert t_fine[0] == 0.0
    assert np.allclose(V[0, :, 0], 0.0)
    assert np.allclose(U[0, :, 0], U[0, 0, 0], atol=1e-10)
    assert abs(U[0, 0, 0] - 0.5) < 1e-12


def test_gem_conservation():
    """γ₀=0 时守恒律：∂/∂t∫|α|²dz + (g/N)(|E(L)|²−|E(0)|²)=0。"""
    t, z = _grid()
    Ein = np.exp(-((t - 2.0) ** 2))
    sw = np.tanh((t - 5.0) / 0.1)
    res = GEM(g=2.0, N=50.0, L=2.0, eta=100.0, gamma0=0.0).solve(t, z, Ein, sw)
    dz = z[1] - z[0]
    dt = t[1] - t[0]
    intA = np.trapezoid(np.abs(res.alpha) ** 2, dx=dz, axis=0)
    flux = (2.0 / 50.0) * (np.abs(res.E[-1]) ** 2 - np.abs(res.E[0]) ** 2)
    C = intA + np.concatenate([[0.0], np.cumsum((flux[:-1] + flux[1:]) / 2 * dt)])
    drift = np.max(np.abs(C)) / (np.abs(intA).max() + 1e-30)
    assert drift < 5e-2


def test_eit_analytic_transmission():
    """EIT 稳态透射谱 vs 解析解（窗口中心 |δ₀|≤1 相对差 < 2%）。

    参数选择注意：稳态建立时间 N·g·L/Ω² = 5×2×2/4 = 5s ≪ T=10。
    """
    t, z = _grid()
    Ein = np.ones_like(t)
    Omega = 2.0 * np.ones_like(t)
    sim = LambdaGEM(g=2.0, N=5.0, L=2.0, eta=0.0, gamma_e=2.0, gamma0=0.5)

    def T_ana(d0):
        g, N, L, ge, g0, Om = 2.0, 5.0, 2.0, 2.0, 0.5, 2.0
        D = (ge + g0 / 2) * (g0 + 1j * d0) + Om ** 2
        return float(np.exp(-N * g * L * np.real((g0 + 1j * d0) / D)))

    for d0 in (-1.0, -0.5, 0.0, 0.5, 1.0):
        res = sim.solve(t, z, Ein, Omega=Omega, delta0=d0)
        T_num = float(np.abs(res.E[-1, -1]))
        rel = abs(T_num - T_ana(d0)) / T_ana(d0)
        assert rel < 0.02, f"δ₀={d0}: {rel:.4f}"




def test_custom_protocol_subclass():
    """自定义协议子类可运行（DetunedGEM 示例）。"""
    import marcher as mc

    class DetunedGEM(mc.Protocol):
        def __init__(self, g, N, L, eta, delta0=0.0):
            self.g, self.N, self.L, self.eta, self.delta0 = g, N, L, eta, delta0

        def _coefficients(self, ctx):
            nz, nt = ctx.nz, ctx.nt
            det = (1e-4 + 1j * (self.delta0 + ctx.sw[None, :] * self.eta
                   * (ctx.z[:, None] - self.L / 2)))
            A = (1j * self.N) * np.ones((1, 1, nz, nt), dtype=np.complex128)
            return (A, np.zeros((1, nz, nt), dtype=np.complex128), det[None, None],
                    (1j * self.g) * np.ones((1, 1, nz, nt), dtype=np.complex128),
                    np.zeros((1, nz, nt), dtype=np.complex128))

    t, z = _grid()
    Ein = np.exp(-((t - 2.0) ** 2))
    res = DetunedGEM(g=2.0, N=10.0, L=2.0, eta=50.0, delta0=0.5).solve(t, z, Ein)
    assert res.E.shape == (800, 400)
    assert np.isfinite(res.E).all()


def test_observables_efficiency():
    """效率与回波指标可用且落在物理范围。"""
    t, z = _grid()
    Ein = np.exp(-((t - 2.0) ** 2))
    sw = np.tanh((t - 5.0) / 0.1)
    res = GEM(g=2.0, N=50.0, L=2.0, eta=100.0).solve(t, z, Ein, sw, t_switch=5.0)
    assert 0.0 <= res.efficiency <= 1.5      # 此 GEM 沿用 +γ₀，不能用正衰减模型的 ≤1 作为严格结论
    assert res.echo["time"] > 5.0            # 回波在读出窗内
