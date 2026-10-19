"""因果 GEM 原型的独立解析解、结构性质与收敛测试，仅使用小型 CPU 网格。"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import j0, j1

from gem_solvers._if2_reference import solve_gem


def parameters(**updates):
    return dict(g=2.0, N=5.0, L=1.0, eta=0.0, gama=0.0) | updates


@pytest.mark.parametrize("method", ["cn", "if2"])
def test_independent_bessel_solution_second_order(method):
    """恒定入射、零失谐的独立贝塞尔解析解同时检验两个复场。"""
    errors = []
    for n in (17, 33, 65):
        z = t = np.linspace(0, 1, n)
        out = solve_gem(parameters(), z, t, np.ones(n), np.ones(n), method=method)
        zz, tt = np.meshgrid(z, t, indexing="ij")
        root = np.sqrt(10 * zz * tt)
        truth_a = np.empty_like(root, dtype=complex)
        np.divide(2j * tt * j1(2 * root), root, out=truth_a, where=root != 0)
        truth_a[root == 0] = 2j * tt[root == 0]
        errors.append([np.max(abs(out["E"] - j0(2*root))),
                       np.max(abs(out["alpha"] - truth_a))])
    errors = np.array(errors)
    assert np.all(errors[:-1] / errors[1:] > 3.7)
    assert np.all(errors[-1] < 0.001)


@pytest.mark.parametrize("method,factor,tolerance", [("cn", 256, 3e-5), ("if2", 1, 2e-11)])
def test_zero_density_piecewise_ode_and_positive_gamma(method, factor, tolerance):
    """N=0 时独立高精度分段 ODE，检验复入射、+gamma 与真实控制事件。"""
    z = np.array([0.0, 0.17, 0.5, 1.0])
    t = np.array([0.0, 0.13, 0.52, 0.73, 1.0])
    drive_t = np.array([0.0, 0.27, 0.41, 0.81, 1.0])
    ein = np.array([1+0.2j, 0.7-0.8j, -0.5j, 0.6+0.2j, 1.2j])
    sw = np.array([1.0, -0.4, 1.5, -1.0, 1.0])
    p = parameters(N=0.0, eta=18.0, gama=0.2)
    out = solve_gem(p, z, t, np.zeros(len(t)), np.ones(len(t)), method=method,
                    t_factor=factor, drive_t=drive_t, drive_Ein=ein, drive_sw=sw)
    events = np.unique(np.r_[t, drive_t])
    expected = np.zeros((len(z), len(events)), dtype=complex)
    for j, (a, b) in enumerate(zip(events[:-1], events[1:])):
        control = sw[max(0, np.searchsorted(drive_t, a, side="right") - 1)]
        diagonal = 0.2 + 18j * (z - 0.5) * control
        expected[:, j+1] = solve_ivp(
            lambda now, state: diagonal * state + 2j * np.interp(now, drive_t, ein),
            (a, b), expected[:, j], rtol=2e-12, atol=1e-13).y[:, -1]
    np.testing.assert_allclose(out["alpha"], expected[:, np.searchsorted(events, t)], atol=tolerance, rtol=0)
    np.testing.assert_allclose(out["E"], np.broadcast_to(np.interp(t, drive_t, ein), out["E"].shape), atol=1e-14)


@pytest.mark.parametrize("method", ["cn", "if2"])
def test_complex_superposition_zero_input_boundary_and_initial(method):
    t = np.array([0.0, 0.12, 0.3, 0.55, 1.0])
    z = np.array([0.0, 0.1, 0.4, 1.0])
    p = parameters(eta=13.0, gama=0.01)
    sw = np.array([1.0, 1.0, -1.0, 0.4, 0.4])
    u, v = 1 + t * (0.2 + 1j), np.cos(t) + 0.7j
    a, b = 0.3 + 0.8j, -0.4 + 0.2j
    results = [solve_gem(p, z, t, wave, sw, method=method, z_factor=2, t_factor=3)
               for wave in (u, v, a*u+b*v, np.zeros_like(u))]
    for field in ("E", "alpha"):
        np.testing.assert_allclose(results[2][field], a*results[0][field]+b*results[1][field], atol=2e-13)
        np.testing.assert_array_equal(results[3][field], 0)
    np.testing.assert_array_equal(results[0]["alpha"][:, 0], 0)
    np.testing.assert_allclose(results[0]["E"][0], u, atol=1e-14)
    np.testing.assert_allclose(results[0]["E"][:, 0], u[0], atol=1e-14)


@pytest.mark.parametrize("method", ["cn", "if2"])
def test_no_future_control_or_incident_leakage(method):
    """修改未来节点会改变其前一插值区间，但不能越过前一驱动节点。"""
    drive_t = np.array([0.0, 0.2, 0.55, 0.8, 1.0])
    t = np.array([0.0, 0.1, 0.2, 0.5, 0.55, 0.7, 0.8, 1.0])
    z = np.linspace(0, 1, 9)
    u, sw = np.ones(5), np.ones(5)
    changed_u, changed_sw = u.copy(), sw.copy()
    changed_u[3:] = [3, -2]
    changed_sw[3:] = [-1, 4]
    outputs = [solve_gem(parameters(eta=7), z, t, np.ones(len(t)), np.ones(len(t)),
                         method=method, t_factor=2, drive_t=drive_t,
                         drive_Ein=e, drive_sw=s)
               for e, s in ((u, sw), (changed_u, sw), (u, changed_sw))]
    for key in ("E", "alpha"):
        np.testing.assert_array_equal(outputs[0][key][:, t <= 0.55], outputs[1][key][:, t <= 0.55])
        np.testing.assert_array_equal(outputs[0][key][:, t <= 0.8], outputs[2][key][:, t <= 0.8])
        assert np.linalg.norm(outputs[0][key][:, -1] - outputs[2][key][:, -1]) > 1e-4


@pytest.mark.parametrize("method", ["cn", "if2"])
def test_nonuniform_grid_first_output_after_zero_and_refinement(method):
    """非均匀请求点精确保留，首点之前完整演化，空间时间加密均二阶收敛。"""
    z = np.array([0.07, 0.2, 0.48, 0.81, 1.0])
    t = np.array([0.04, 0.15, 0.38, 0.7, 1.0])
    root = np.sqrt(10 * z[:, None] * t[None])
    truth_e = j0(2 * root)
    truth_a = 2j * t[None] * j1(2 * root) / root
    errors = []
    for factor in (2, 4, 8):
        out = solve_gem(parameters(), z, t, np.ones(len(t)), np.ones(len(t)),
                        method=method, z_factor=factor, t_factor=factor)
        np.testing.assert_array_equal(out["z"], z)
        np.testing.assert_array_equal(out["t"], t)
        errors.append([np.max(abs(out["E"] - truth_e)), np.max(abs(out["alpha"] - truth_a))])
    errors = np.array(errors)
    assert np.all(errors[:-1] / errors[1:] > 3.7)


@pytest.mark.parametrize("method", ["cn", "if2"])
def test_coupled_detuned_time_refinement_against_independent_mol(method):
    """小网格独立组装 Volterra 矩阵并高精度积分，核查失谐耦合而非只有N=0。"""
    z = np.linspace(0, 1, 9)
    drive_t = np.array([0.0, 0.31, 0.66, 1.0])
    u = np.array([1.0, 0.2+0.3j, 0.6-0.2j, 0.4])
    sw = np.array([1.0, -1.0, 0.7, 0.7])
    p = parameters(eta=15.0, gama=0.1)
    q = np.zeros((len(z), len(z)))
    for i in range(1, len(z)):
        q[i, :i] += np.diff(z[:i+1]) / 2
        q[i, 1:i+1] += np.diff(z[:i+1]) / 2
    truth = np.zeros((len(z), len(drive_t)), dtype=complex)
    for j, (a, b) in enumerate(zip(drive_t[:-1], drive_t[1:])):
        matrix = np.diag(0.1 + 15j * (z - 0.5) * sw[j]) - 10 * q
        truth[:, j+1] = solve_ivp(
            lambda now, state: matrix @ state + 2j * np.interp(now, drive_t, u),
            (a, b), truth[:, j], rtol=2e-12, atol=1e-13).y[:, -1]
    errors = []
    for factor in (8, 16, 32):
        out = solve_gem(p, z, drive_t, u, sw, method=method, t_factor=factor)
        errors.append(np.max(abs(out["alpha"] - truth)))
    assert np.all(np.array(errors[:-1]) / errors[1:] > 3.7)
    assert errors[-1] < 1e-3


def test_invalid_drive_and_refinement_are_rejected():
    args = (parameters(), [0, 1], [0, 1], [1, 1], [1, -1])
    with pytest.raises(ValueError, match="正整数"):
        solve_gem(*args, t_factor=0)
    with pytest.raises(ValueError, match="同时提供"):
        solve_gem(*args, drive_t=[0, 1])
    with pytest.raises(ValueError, match="实值"):
        solve_gem(parameters(), [0, 1], [0, 1], [1, 1], [1j, 1j])


def test_nearly_coincident_output_and_drive_nodes_preserve_exact_events():
    """float32 请求时刻与原始节点近重合时不合并坐标，极短区间保持有限。"""
    t = np.array([0.025, 0.3, 0.5], dtype=np.float32).astype(float)
    drive_t = np.array([0.0, 0.025, 0.3, 0.3 + 1e-14, 0.5])
    sw = np.array([1.0, 1.0, -1.0, 0.3, 0.3])
    z = np.array([0.0, 0.5, 1.0])
    out = solve_gem(parameters(N=0, eta=30, gama=0.1), z, t, np.ones(3), np.ones(3),
                    method="if2", t_factor=2, drive_t=drive_t,
                    drive_Ein=np.ones(5), drive_sw=sw)
    events = np.unique(np.r_[drive_t, t])
    expected = np.zeros((len(z), len(events)), dtype=complex)
    for j, (a, b) in enumerate(zip(events[:-1], events[1:])):
        control = sw[max(0, np.searchsorted(drive_t, a, side="right") - 1)]
        diagonal = 0.1 + 30j * (z - 0.5) * control
        growth = np.exp(diagonal * (b-a))
        expected[:, j+1] = growth * expected[:, j] + 2j * np.expm1(diagonal*(b-a)) / diagonal
    np.testing.assert_array_equal(out["t"], t)
    np.testing.assert_allclose(out["alpha"], expected[:, np.searchsorted(events, t)], atol=2e-13)
