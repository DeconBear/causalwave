"""审查缺陷的独立回归用例：解析解、跳变、网格及观测量。"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import j0, j1

from marcher import GEM, LambdaGEM
from marcher.core import march_system
from marcher.observables import echo_fidelity, efficiency


@pytest.mark.parametrize("t", [np.linspace(0, 1, 11), np.linspace(.5, 1, 11),
                                  np.linspace(.025, 10, 400)])
def test_zero_coupling_preserves_requested_boundary(t):
    """零耦合精确解 E(z,t)=Ein(t)，不能因输出采样改变边界。"""
    Ein = (1 + 2j) * t
    result = GEM(g=0, N=0, L=1, eta=0, gamma0=0).solve(
        t, np.linspace(0, 1, 4), Ein, z_factor=2, t_factor=4)
    np.testing.assert_allclose(result.E, np.broadcast_to(Ein, result.E.shape), atol=1e-13)


def test_manufactured_solution_on_nonuniform_grid():
    """U_z=1、V_t=1 的解 U=V=z+t 在实际请求坐标上成立。"""
    z = np.array([0., .1, .4, 1.])
    t = np.array([0., .2, .3, .8, 1.])
    zero = np.zeros((1, 1, len(z), len(t)))
    src = np.ones((1, len(z), len(t)))
    U, V = march_system(z, t, t[None], z[None], zero, src, zero, zero, src,
                        z_factor=3, t_factor=4)
    expected = z[:, None] + t[None]
    np.testing.assert_allclose(U[0], expected, atol=1e-13)
    np.testing.assert_allclose(V[0], expected, atol=1e-13)


def test_gem_bessel_second_order():
    """独立贝塞尔解约束复值 E/alpha 的二阶精度。"""
    errors = []
    for n in (17, 33, 65):
        z = t = np.linspace(0, 1, n)
        res = GEM(g=2, N=5, L=1, eta=0, gamma0=0).solve(
            t, z, np.ones(n), z_factor=1, t_factor=1)
        Z, T = np.meshgrid(z, t, indexing="ij")
        root = np.sqrt(10 * Z * T)
        alpha = np.zeros_like(root, dtype=complex)
        np.divide(2j * T * j1(2 * root), root, out=alpha, where=root != 0)
        alpha[root == 0] = 2j * T[root == 0]
        errors.append([np.max(abs(res.E - j0(2 * root))), np.max(abs(res.alpha - alpha))])
    errors = np.asarray(errors)
    assert np.all(errors[:-1] / errors[1:] > 3.7)
    assert np.all(errors[-1] < .001)


def test_gem_step_switch_against_exact_boundary_ode():
    """N=0 时各空间点独立，常值驱动的分段 ODE 有精确指数解。"""
    t = np.linspace(0, 1, 11)
    z = np.array([0., 1.])
    sw = np.where(t < .6, 1., -1.)
    res = GEM(g=1, N=0, L=1, eta=20, gamma0=0).solve(
        t, z, np.ones_like(t), sw, z_factor=1, t_factor=64)
    expected = np.zeros((2, len(t)), dtype=complex)
    for j in range(1, len(t)):
        k = 20j * sw[j-1] * (z - .5)
        growth = np.exp(k * (t[j] - t[j-1]))
        expected[:, j] = growth * expected[:, j-1] + 1j * (growth - 1) / k
    np.testing.assert_allclose(res.alpha, expected, atol=6e-5, rtol=0)


def test_lambda_mixed_step_sw_and_linear_complex_omega():
    """独立分段 solve_ivp 验证 sw 左常值、Omega 复值线性及共轭符号。"""
    t = np.linspace(0, 1, 9)
    sw = np.where(t < .5, 1., -1.)
    omega = (1 + 2j) * (1 + t)
    sim = LambdaGEM(g=1, N=0, L=1, eta=8, gamma0=.1, gamma_e=2, Delta=.3)
    res = sim.solve(t, np.array([0., 1.]), np.ones_like(t), sw,
                    Omega=omega, z_factor=1, t_factor=32)
    expected = np.zeros((2, len(t)), dtype=complex)
    for j in range(1, len(t)):
        def rhs(tt, v):
            Om = (1 + 2j) * (1 + tt)
            return np.array([-(2.05 + .3j) * v[0] + 1j + 1j * Om * v[1],
                             -(.1 - 4j * sw[j-1]) * v[1] + 1j * Om.conjugate() * v[0]])
        expected[:, j] = solve_ivp(rhs, (t[j-1], t[j]), expected[:, j-1],
                                    rtol=1e-11, atol=1e-13).y[:, -1]
    np.testing.assert_allclose(res.sigma[:, 0], expected, atol=3e-5, rtol=0)


@pytest.mark.parametrize("sim", [GEM(g=1, N=1, L=1, eta=0),
                                  LambdaGEM(g=1, N=1, L=1, eta=0)])
def test_fine_result_retains_consistent_public_output(sim):
    """细网格是额外诊断数据，公共 Result 仍对应请求网格。"""
    t = np.linspace(0, 1, 12)
    z = np.linspace(0, 1, 8)
    res = sim.solve(t, z, np.ones_like(t), t_switch=.5,
                    z_factor=2, t_factor=3, return_fine=True)
    assert res.E.shape == (len(res.z), len(res.t)) == (8, 12)
    assert np.isfinite(res.efficiency)
    assert np.isfinite(res.echo["peak"])


def test_perfect_complex_echo_and_phase_distortion():
    """完美复值时间反演重叠为1，时变相位畸变不能被幅值比较掩盖。"""
    t = np.linspace(0, 10, 1001)
    Ein = np.exp(-((t-2)/.4)**2) * np.exp(3j * t)
    echo = np.exp(-((t-8)/.4)**2) * np.exp(3j * (10-t))
    mask = t > 5
    assert echo_fidelity(echo * 1j, Ein, t, 8, 2, mask) == pytest.approx(1, abs=1e-12)
    assert echo_fidelity(echo * np.exp(10j*t), Ein, t, 8, 2, mask) < .5



def test_multiple_coupled_fields_with_known_sources():
    """非均匀网格上的 P=2、Q=3 制造解验证块联立和源项端点。"""
    z, t = np.array([0., .1, .4, 1.]), np.array([0., .2, .7, 1.])
    u = np.array([1+1j, 2-1j])
    v = np.array([1-2j, 2+1j, 3.])
    U = u[:, None, None] * (z[:, None] + t[None])
    V = v[:, None, None] * (z[:, None] + 2*t[None])
    A = np.broadcast_to(np.arange(6).reshape(2, 3, 1, 1)*.02j, (2, 3, 4, 4)).copy()
    B = np.broadcast_to(-np.eye(3)[:, :, None, None], (3, 3, 4, 4)).copy()
    C = np.broadcast_to(np.ones((3, 2, 1, 1))*.1j, (3, 2, 4, 4)).copy()
    s = u[:, None, None] - np.einsum("pqzt,qzt->pzt", A, V)
    r = 2*v[:, None, None] - np.einsum("qlzt,lzt->qzt", B, V) - np.einsum("qpzt,pzt->qzt", C, U)
    actual_u, actual_v = march_system(z, t, U[:, 0], V[:, :, 0], A, s, B, C, r,
                                      z_factor=2, t_factor=3)
    np.testing.assert_allclose(actual_u, U, atol=1e-12)
    np.testing.assert_allclose(actual_v, V, atol=1e-12)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_refinement_rejected(factor):
    with pytest.raises(ValueError, match="正整数"):
        GEM(g=0, N=0, L=1, eta=0).solve([0, 1], [0, 1], [0, 1], t_factor=factor)


@pytest.mark.parametrize("name,value", [("g", 1+1j), ("eta", np.inf),
                                        ("gamma0", np.nan), ("L", 0)])
@pytest.mark.parametrize("solver", [GEM, LambdaGEM])
def test_protocol_physical_parameters_are_finite_real_scalars(solver, name, value):
    """低层协议也必须在构造时检查实参数，不能产生静默非有限场。"""
    params = dict(g=1, N=1, L=1, eta=0)
    params[name] = value
    with pytest.raises(ValueError):
        solver(**params)


@pytest.mark.parametrize("name,value", [("t", np.array([0, 1+1j])),
                                        ("z", np.array([0, 1+1j])),
                                        ("sw", np.array([1+1j, 1])),
                                        ("sw", [1, np.inf]),
                                        ("Ein", [1, np.nan])])
def test_protocol_rejects_invalid_coordinates_and_drives(name, value):
    """入口不丢弃坐标或控制虚部，并拒绝非有限入射。"""
    args = dict(t=[0, 1], z=[0, 1], Ein=[1, 1], sw=[1, 1])
    args[name] = value
    with pytest.raises(ValueError):
        GEM(g=1, N=1, L=1, eta=0).solve(**args, z_factor=1, t_factor=1)


def test_protocol_rejects_silent_parameter_override():
    """改变构造参数必须重新建对象，Lambda 的额外 delta0 参数仍可用。"""
    sim = GEM(g=1, N=1, L=1, eta=0)
    with pytest.raises(ValueError, match="重新构造"):
        sim.solve([0, 1], [0, 1], [1, 1], eta=123)
    sim = LambdaGEM(g=1, N=0, L=1)
    result = sim.solve([0, 1], [0, 1], [1, 1], Omega=[1+1j, 2j], delta0=.2,
                       z_factor=1, t_factor=1)
    assert np.isfinite(result.sigma).all()
    with pytest.raises(ValueError, match="实"):
        sim.solve([0, 1], [0, 1], [1, 1], delta0=1j)


@pytest.mark.parametrize("name", ["u_bc", "v_ic", "A_coef", "s_coef", "B_coef", "C_coef", "r_coef"])
def test_core_rejects_nonfinite_boundary_initial_coefficients_and_sources(name):
    """通用多场入口不能让任一非有限边界、初态、系数或源进入推进。"""
    args = dict(z=[0, 1], t=[0, 1], u_bc=np.ones((1, 2)), v_ic=np.zeros((1, 2)),
                A_coef=np.zeros((1, 1, 2, 2)), s_coef=np.zeros((1, 2, 2)),
                B_coef=np.zeros((1, 1, 2, 2)), C_coef=np.zeros((1, 1, 2, 2)),
                r_coef=np.zeros((1, 2, 2)))
    args[name].flat[-1] = np.nan
    with pytest.raises(ValueError, match="有限"):
        march_system(**args, z_factor=1, t_factor=1)


@pytest.mark.parametrize("name,shape", [("u_bc", (2,)), ("v_ic", (1, 3)),
                                        ("A_coef", (1, 2, 2, 2)), ("r_coef", (2, 2))])
def test_core_rejects_inconsistent_field_shapes(name, shape):
    """形状错误用稳定的 ValueError 拒绝，不依赖可被优化关闭的 assert。"""
    args = dict(z=[0, 1], t=[0, 1], u_bc=np.ones((1, 2)), v_ic=np.zeros((1, 2)),
                A_coef=np.zeros((1, 1, 2, 2)), s_coef=np.zeros((1, 2, 2)),
                B_coef=np.zeros((1, 1, 2, 2)), C_coef=np.zeros((1, 1, 2, 2)),
                r_coef=np.zeros((1, 2, 2)))
    args[name] = np.zeros(shape)
    with pytest.raises(ValueError, match="形状"):
        march_system(**args, z_factor=1, t_factor=1)


def test_efficiency_integrates_disjoint_windows_separately():
    """非均匀网格的分离窗口不跨间隔；孤立点与空窗口不贡献积分。"""
    t = np.array([0., .1, .5, 1.5, 3., 4., 6.])
    field = np.ones(len(t), dtype=complex)
    mask = np.array([True, True, False, True, True, False, True])
    assert efficiency(field, field, t, mask) == pytest.approx(1.6/6)
    assert efficiency(field, field, t, np.zeros(len(t), dtype=bool)) == 0
    mask = (t >= .5) & (t <= 4)
    expected = np.trapezoid(abs(field[mask])**2, t[mask]) / np.trapezoid(abs(field)**2, t)
    assert efficiency(field, field, t, mask) == expected
