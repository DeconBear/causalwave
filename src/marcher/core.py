"""marcher 求解器框架：单向输运 × 时间演化耦合系统的特征线推进。

适用范围（一般形式）:
    ∂u_p/∂z = Σ_q a_pq(z,t)·v_q + s_p(z,t)         输运场（BC: z=0 给定）
    ∂v_q/∂t = Σ_l b_ql(z,t)·v_l + Σ_p c_qp(z,t)·u_p + r_q(z,t)  演化场（IC: t=0 给定）

即每个场只沿一个坐标单向输运、耦合仅零阶（可含任意 z/t 依赖系数、可含已知源）。
GEM 两能级（P=1,Q=1）与 Λ-GEM 三能级（P=1,Q=2）均为特例。

格式：对角推进（i+j=s）+ 半隐式梯形（Crank-Nicolson），每格点解 (P+Q) 复线性
块系统；光滑区间二阶精度、O(Nz·Nt) 格点推进、对角线向量化。

实现对常用2×2、3×3块使用显式小矩阵公式；其他维度使用NumPy线性求解。
协议定义见protocols.py，验证见仓库tests/。
"""
from __future__ import annotations

import numpy as np


def _fine_grid(z0: np.ndarray, t0: np.ndarray, z_factor: int, t_factor: int):
    """逐区间加密，精确保留原坐标；缺少 t=0 时补初始点。"""
    for name, axis in (("z", z0), ("t", t0)):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} 必须为至少两点的有限严格递增一维坐标")
    if z0[0] != 0 or t0[0] < 0:
        raise ValueError("需要 z[0]=0、t[0]>=0，初始条件定义于 t=0")
    for factor in (z_factor, t_factor):
        if not isinstance(factor, (int, np.integer)) or factor < 1:
            raise ValueError("网格加密倍数必须为正整数")

    def refine(axis, factor):
        cells = np.linspace(axis[:-1], axis[1:], factor, endpoint=False).T.ravel()
        return np.concatenate([cells, axis[-1:]])

    z_fine = refine(z0, z_factor)
    t_base = np.concatenate([[0.0], t0]) if t0[0] > 0 else t0
    t_fine = refine(t_base, t_factor)
    return z_fine, t_fine, t_base


def _interp_weights(x_src: np.ndarray, x_dst: np.ndarray):
    """线性插值索引与权重；越界按端点常值延拓（与 np.interp 语义一致）。"""
    n = len(x_src)
    if n == 1:
        zeros = np.zeros(len(x_dst))
        return np.zeros(len(x_dst), dtype=np.intp), zeros
    idx = np.interp(x_dst, x_src, np.arange(n, dtype=np.float64))
    i0 = np.floor(idx).astype(np.intp)
    i0 = np.clip(i0, 0, n - 2)
    frac = np.clip(idx - i0, 0.0, 1.0)
    return i0, frac


def _interp_axis(X, x_src, x_dst, axis=-1):
    """沿指定轴张量化线性插值（端点常值延拓），等价于逐行 np.interp。"""
    X = np.asarray(X)
    i0, frac = _interp_weights(x_src, x_dst)
    a0 = np.take(X, i0, axis=axis)
    a1 = np.take(X, i0 + 1, axis=axis)
    shape = [1] * X.ndim
    shape[axis] = len(frac)
    w = frac.reshape(shape)
    return a0 * (1.0 - w) + a1 * w


def _extend_and_interp(X, t, t_fine, axis=-1):
    """沿时间线性插值；首采样之前常数延拓，不重复添加零时刻。"""
    return _interp_axis(X, t, t_fine, axis=axis)


def _sample_output(U, V, z_fine, t_fine, z, t):
    """嵌套网格含全部请求节点，直接按真实坐标取值。"""
    zi, ti = np.searchsorted(z_fine, z), np.searchsorted(t_fine, t)
    return (U[np.ix_(np.arange(U.shape[0]), zi, ti)],
            V[np.ix_(np.arange(V.shape[0]), zi, ti)])


# ---------------------------------------------------------------------------
# 小尺寸块运算：P、Q ∈ {1,2} 时用显式公式，避免逐对角线的事件调度开销。
# ---------------------------------------------------------------------------

def _contract_A(Ax, V, P, Q):
    """(P,Q,K)·(Q,K) -> (P,K)"""
    if P == 1:
        if Q == 1:
            return Ax[0, 0] * V[0]
        if Q == 2:
            return Ax[0, 0] * V[0] + Ax[0, 1] * V[1]
    return np.einsum("pqk,qk->pk", Ax, V)


def _contract_B(Bx, V, Q):
    """(Q,Q,K)·(Q,K) -> (Q,K)"""
    if Q == 1:
        return Bx[0, 0] * V[0]
    if Q == 2:
        return np.stack([Bx[0, 0] * V[0] + Bx[0, 1] * V[1],
                         Bx[1, 0] * V[0] + Bx[1, 1] * V[1]])
    return np.einsum("qlk,lk->qk", Bx, V)


def _contract_C(Cx, U, P, Q):
    """(Q,P,K)·(P,K) -> (Q,K)"""
    if P == 1:
        if Q == 1:
            return (Cx[0, 0] * U[0])[None]
        if Q == 2:
            return np.stack([Cx[0, 0] * U[0], Cx[1, 0] * U[0]])
    return np.einsum("qpk,pk->qk", Cx, U)


def _solve_Q(B_new, d2, rhs):
    """解 (I - d2·B)·V = rhs，B_new (Q,Q,K)、d2 (K,)、rhs (Q,K) -> (Q,K)。"""
    Q = rhs.shape[0]
    if Q == 1:
        return (rhs[0] / (1.0 - d2 * B_new[0, 0]))[None]
    if Q == 2:
        b11, b12 = B_new[0, 0], B_new[0, 1]
        b21, b22 = B_new[1, 0], B_new[1, 1]
        m11 = 1.0 - d2 * b11
        m22 = 1.0 - d2 * b22
        m12 = -d2 * b12
        m21 = -d2 * b21
        det = m11 * m22 - m12 * m21
        return np.stack([(rhs[0] * m22 - m12 * rhs[1]) / det,
                         (m11 * rhs[1] - m21 * rhs[0]) / det])
    return np.linalg.solve(
        np.eye(Q)[None] - d2[:, None, None] * B_new.transpose(2, 0, 1),
        rhs.T[:, :, None])[:, :, 0].T


def _solve_block(A_ij, B_ij, C_ij, c2, d2, RHS_u, RHS_v, P, Q):
    """解二阶块系统（对角推进内部点）:

        [ I           -c2·A ] [U]   [RHS_u]
        [ -d2·C   I - d2·B  ] [V] = [RHS_v]

    A (P,Q,K)、B (Q,Q,K)、C (Q,P,K)、RHS_u (P,K)、RHS_v (Q,K)；
    返回 (U (P,K), V (Q,K))。
    """
    if P == 1 and Q == 1:
        a = A_ij[0, 0]
        b = B_ij[0, 0]
        c = C_ij[0, 0]
        ru = RHS_u[0]
        rv = RHS_v[0]
        m22 = 1.0 - d2 * b
        det = m22 - c2 * d2 * a * c
        U = (ru * m22 + c2 * a * rv) / det
        V = (rv + d2 * c * ru) / det
        return U[None], V[None]
    if P == 1 and Q == 2:
        a1, a2 = A_ij[0, 0], A_ij[0, 1]
        b11, b12 = B_ij[0, 0], B_ij[0, 1]
        b21, b22 = B_ij[1, 0], B_ij[1, 1]
        c1, c2_ = C_ij[0, 0], C_ij[1, 0]
        ru = RHS_u[0]
        r1, r2 = RHS_v[0], RHS_v[1]
        m11 = np.ones_like(d2)
        m12 = -c2 * a1
        m13 = -c2 * a2
        m21 = -d2 * c1
        m22 = 1.0 - d2 * b11
        m23 = -d2 * b12
        m31 = -d2 * c2_
        m32 = -d2 * b21
        m33 = 1.0 - d2 * b22
        # 3×3 解析求逆（伴随矩阵 / 行列式）
        k11 = m22 * m33 - m23 * m32
        k12 = m13 * m32 - m12 * m33
        k13 = m12 * m23 - m13 * m22
        k21 = m23 * m31 - m21 * m33
        k22 = m11 * m33 - m13 * m31
        k23 = m13 * m21 - m11 * m23
        k31 = m21 * m32 - m22 * m31
        k32 = m12 * m31 - m11 * m32
        k33 = m11 * m22 - m12 * m21
        det = m11 * k11 + m12 * k21 + m13 * k31
        U = (k11 * ru + k12 * r1 + k13 * r2) / det
        V1 = (k21 * ru + k22 * r1 + k23 * r2) / det
        V2 = (k31 * ru + k32 * r1 + k33 * r2) / det
        return U[None], np.stack([V1, V2])
    K = RHS_u.shape[-1]
    M = np.zeros((K, P + Q, P + Q), dtype=np.complex128)
    M[:, :P, :P] = np.eye(P)[None]
    M[:, :P, P:] = -c2[:, None, None] * A_ij.transpose(2, 0, 1)
    M[:, P:, :P] = -d2[:, None, None] * C_ij.transpose(2, 0, 1)
    M[:, P:, P:] = np.eye(Q)[None] - d2[:, None, None] * B_ij.transpose(2, 0, 1)
    sol = np.linalg.solve(M, np.concatenate([RHS_u, RHS_v], axis=0).T[:, :, None])[:, :, 0].T
    return sol[:P], sol[P:]


def march_system(
    z: np.ndarray,
    t: np.ndarray,
    u_bc,          # (P, nt)：输运场 z=0 边界
    v_ic,          # (Q, nz)：演化场 t=0 初始
    A_coef,        # (P, Q, nz, nt)：∂u_p/∂z 中 v_q 的系数
    s_coef,        # (P, nz, nt)：∂u_p/∂z 的已知源
    B_coef,        # (Q, Q, nz, nt)：∂v_q/∂t 中 v_l 的系数
    C_coef,        # (Q, P, nz, nt)：∂v_q/∂t 中 u_p 的系数
    r_coef,        # (Q, nz, nt)：∂v_q/∂t 的已知源
    z_factor: int = 4,
    t_factor: int = 4,
    return_fine: bool = False,
    B_step_mask=None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对角推进 + CN 半隐式求解。返回 (U, V)，形状 (P, nz, nt)、(Q, nz, nt)。

    系数张量定义在输出网格 (nz, nt) 上；内部细网格加密 z_factor/t_factor 倍，
    默认系数双线性插值（首采样之前常数延拓）。z、t 可非均匀。
    B_step_mask 为 (Q,Q) 布尔矩阵：选中分量沿时间左常值保持；
    跳变点已包含在细网格中，每个积分区间使用其内部的单侧值。
    """
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u_bc = np.asarray(u_bc, dtype=np.complex128)
    v_ic = np.asarray(v_ic, dtype=np.complex128)
    A = np.asarray(A_coef, dtype=np.complex128)
    s = np.asarray(s_coef, dtype=np.complex128)
    B = np.asarray(B_coef, dtype=np.complex128)
    C = np.asarray(C_coef, dtype=np.complex128)
    r = np.asarray(r_coef, dtype=np.complex128)

    P, Q = u_bc.shape[0], v_ic.shape[0]
    nz, nt = len(z), len(t)
    assert A.shape == (P, Q, nz, nt) and s.shape == (P, nz, nt)
    assert B.shape == (Q, Q, nz, nt) and C.shape == (Q, P, nz, nt) and r.shape == (Q, nz, nt)

    z_fine, t_fine, t_base = _fine_grid(z, t, z_factor, t_factor)
    nzf, ntf = len(z_fine), len(t_fine)
    dz_half = np.diff(z_fine) / 2.0
    dt_half = np.diff(t_fine) / 2.0
    step_mask = np.zeros((Q, Q), dtype=bool) if B_step_mask is None else np.asarray(B_step_mask, dtype=bool)
    if step_mask.shape != (Q, Q):
        raise ValueError("B_step_mask 形状必须为 (Q,Q)")

    # BC/IC/系数 -> 细网格（张量化插值；与逐行 np.interp 等价）
    u_bc_f = _extend_and_interp(u_bc, t, t_fine, axis=1)      # (P, ntf)
    v_ic_f = _interp_axis(v_ic, z, z_fine, axis=1)            # (Q, nzf)

    A_f = _interp_axis(_extend_and_interp(A, t, t_fine, axis=-1), z, z_fine, axis=-2)
    s_f = _interp_axis(_extend_and_interp(s, t, t_fine, axis=-1), z, z_fine, axis=-2)
    B_t = _extend_and_interp(B, t, t_fine, axis=-1)
    step_idx = np.clip(np.searchsorted(t, t_fine, side="right") - 1, 0, nt - 1)
    for q, l in zip(*np.nonzero(step_mask)):
        B_t[q, l] = np.take(B[q, l], step_idx, axis=-1)
    B_f = _interp_axis(B_t, z, z_fine, axis=-2)
    del B_t
    C_f = _interp_axis(_extend_and_interp(C, t, t_fine, axis=-1), z, z_fine, axis=-2)
    r_f = _interp_axis(_extend_and_interp(r, t, t_fine, axis=-1), z, z_fine, axis=-2)

    U = np.zeros((P, nzf, ntf), dtype=np.complex128)
    V = np.zeros((Q, nzf, ntf), dtype=np.complex128)
    U[:, 0, :] = u_bc_f    # BC
    V[:, :, 0] = v_ic_f    # IC

    for s_i in range(1, nzf + ntf - 1):
        i_lo = max(0, s_i - (ntf - 1))
        # 含 i=s_i（即 j=0 的 t=0 列）。旧上界 s_i-1 会把该列整段排除，
        # 导致 U[:, i>0, 0] 停在 0；α(z,0)=0 时应有 E(z,0)=Ein(0)。
        i_hi = min(nzf - 1, s_i)
        if i_lo > i_hi:
            continue
        i = np.arange(i_lo, i_hi + 1)
        j = s_i - i

        # 段 1: j==0（t=0 列）——演化场由 IC 固定，输运场纯空间推进
        m0 = (j == 0)
        if m0.any():
            ii = i[m0]
            c2 = dz_half[ii - 1]
            f_new = s_f[:, ii, 0] + _contract_A(A_f[:, :, ii, 0], V[:, ii, 0], P, Q)
            f_old = s_f[:, ii - 1, 0] + _contract_A(A_f[:, :, ii - 1, 0], V[:, ii - 1, 0], P, Q)
            U[:, ii, 0] = U[:, ii - 1, 0] + c2 * (f_new + f_old)

        # 段 2: i==0（z=0 行）——输运场由 BC 固定，演化场纯时间推进
        m1 = (i == 0)
        if m1.any():
            jj = j[m1]
            d2 = dt_half[jj - 1]
            # rhs_q = V_old + d2*(B_old·V_old + C_old·U_old + r_old + C_new·U_new + r_new)
            B_old = B_f[:, :, 0, jj - 1]
            B_new = np.where(step_mask[:, :, None], B_old, B_f[:, :, 0, jj])
            C_old = C_f[:, :, 0, jj - 1]
            C_new = C_f[:, :, 0, jj]
            U_old = U[:, 0, jj - 1]
            U_new = U[:, 0, jj]
            V_old = V[:, 0, jj - 1]
            rhs = (V_old
                   + d2 * (_contract_B(B_old, V_old, Q)
                           + _contract_C(C_old, U_old, P, Q)
                           + r_f[:, 0, jj - 1]
                           + _contract_C(C_new, U_new, P, Q)
                           + r_f[:, 0, jj]))
            V[:, 0, jj] = _solve_Q(B_new, d2, rhs)

        # 段 3: 内部点——(P+Q) 块联立
        m2 = (j != 0) & (i != 0)
        if m2.any():
            ii = i[m2]
            jj = j[m2]
            c2 = dz_half[ii - 1]
            d2 = dt_half[jj - 1]
            U_prev = U[:, ii - 1, jj]
            V_prev = V[:, ii - 1, jj]
            U_old = U[:, ii, jj - 1]
            V_old = V[:, ii, jj - 1]
            A_ij = A_f[:, :, ii, jj]
            A_im = A_f[:, :, ii - 1, jj]
            B_im = B_f[:, :, ii, jj - 1]
            B_ij = np.where(step_mask[:, :, None], B_im, B_f[:, :, ii, jj])
            C_ij = C_f[:, :, ii, jj]
            C_im = C_f[:, :, ii, jj - 1]

            Au = _contract_A(A_im, V_prev, P, Q)
            RHS_u = U_prev + c2 * (Au + s_f[:, ii, jj] + s_f[:, ii - 1, jj])
            Bv = _contract_B(B_im, V_old, Q)
            Cu = _contract_C(C_im, U_old, P, Q)
            RHS_v = V_old + d2 * (Bv + Cu + r_f[:, ii, jj - 1] + r_f[:, ii, jj])

            U_new, V_new = _solve_block(A_ij, B_ij, C_ij, c2, d2, RHS_u, RHS_v, P, Q)
            U[:, ii, jj] = U_new
            V[:, ii, jj] = V_new

    if return_fine:
        return U, V, z_fine, t_fine
    return _sample_output(U, V, z_fine, t_fine, z, t)
