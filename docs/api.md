# Python API 与数据合同

## 统一 GEM 接口

```python
import numpy as np
from gem_solvers import GEMParameters, Drive, GEMResult, solve_gem

p = GEMParameters(g=1, N=2, L=1, eta=4, gamma=-0.01)
nodes = np.linspace(0, 2, 201)
drive = Drive(nodes, np.exp(-((nodes-0.3)/0.1)**2), np.where(nodes<1, 1., -1.))
result = solve_gem(p, np.linspace(0,1,41), np.linspace(0,2,81), drive,
                   method="if2", z_factor=4, t_factor=2)
```

### `GEMParameters`

|字段|默认值|解释|
|---|---|---|
|`g`|2.0|Bloch 方程中光场耦合系数|
|`N`|50.0|Maxwell 方程中的传播耦合系数；其单位由模型归一化决定|
|`L`|2.0|严格正的样品长度|
|`eta`|100.0|线性空间失谐梯度|
|`gamma`|0.0|`+gamma*alpha` 中的有符号系数；负值对应衰减|

全部参数必须有限。默认值不是推荐精度配置。数据类冻结，方便保留求解参数。

### `Drive(t, Ein, sw)`

三个同长一维数组，`t` 至少两点、从零开始、严格递增；`Ein` 为有限复数，`sw` 为有限实数。构造时复制并设置数组只读，避免后续意外修改驱动。`Ein` 线性插值，`sw` 左常值保持；不要预先把它们压缩到输出网格。

### `solve_gem(params, z, t, drive, *, method="if2", z_factor=4, t_factor=2)`

- `z/t` 至少两点、有限、严格递增，`z[0]=0`，`0<z[-1]≤L`，`t[0]≥0`。
- 若首个输出时间大于零，仍从物理初态 `t=0` 推进，只返回请求节点。
- 驱动必须覆盖最后输出时间。每个原始网格区间分别加密，精确保留原始坐标。
- `method` 仅支持 `if2`、`marcher`；加密因子必须为正整数，不接受布尔值。
- 返回 `GEMResult`；两后端均不请求 GPU，也不更新任何权重。

### `GEMResult`

`E`、`alpha` 是 `(len(z),len(t))` 的复数组，`z/t` 是实际请求坐标的副本，`method` 是后端名。`elapsed_seconds` 为统一 API 的本次墙钟耗时，含准备与求解，不含用户后处理和保存。

`save(path)` 写入压缩 NPZ，键为 `E, alpha, z, t, method, elapsed_seconds`，已存在时拒绝覆盖；父目录由调用者准备。结果文件**不含全部原生驱动与参数**，要完整复现时必须另存这些输入。

## marcher 低层接口

```python
from marcher import GEM
solver = GEM(g=1, N=2, L=1, eta=4, gamma0=-0.01)
# 低层接口的驱动与请求时间使用同一节点数组；应显式包含全部事件。
low = solver.solve(nodes, np.linspace(0,1,41), drive.Ein, drive.sw,
                   z_factor=4, t_factor=2, t_switch=1.0)
print(low.E.shape, low.alpha.shape)
```

低层 `GEM.gamma0` 是式中正号的**有符号**系数，其历史默认值 `1e-4` 会产生增长，与统一 API 的 `gamma=0` 默认值不同；建议总是显式指定。

### Λ-GEM 与协议扩展

`marcher.LambdaGEM(g, N, L, eta=0, gamma0=1e-4, gamma_e=2, Delta=0)` 增加光学相干性 $\sigma_{13}$ 与自旋相干性 $\sigma_{12}$：

$$
\begin{aligned}
E_z &= iN\sigma_{13},\\
\dot\sigma_{13}&=-(\gamma_e+\gamma_0/2+i\Delta)\sigma_{13}+igE+i\Omega\sigma_{12},\\
\dot\sigma_{12}&=-(\gamma_0+i\delta)\sigma_{12}+i\Omega^*\sigma_{13},\\
\delta(z,t)&=\delta_0+s(t)\eta(z-L/2).
\end{aligned}
$$

这里 `gamma0/gamma_e` 在方程前已有**负号**，正值表示衰减；不要套用两能级 GEM 的有符号约定。`solve(..., Omega=complex_array, delta0=value)` 传控制场与均匀双光子失谐。返回 `sigma.shape=(2,nz,nt)`，顺序为 $(\sigma_{13},\sigma_{12})$，`alpha=None`。本项目不把两能级面板的精度保证扩展到 Λ-GEM。

继承 `Protocol` 时覆写 `_coefficients(ctx)` 返回 `A,s,B,C,r`，分别对应形状 `(P,Q,nz,nt)`、`(P,nz,nt)`、`(Q,Q,nz,nt)`、`(Q,P,nz,nt)`、`(Q,nz,nt)`；可覆写 `_u_boundary`、`_v_initial` 以及场数属性。需要阶跃的 `B` 分量用 `_B_step_mask` 声明。数学定义见[式 (3)](methods.md)。

## 观测量的解释限制

低层 `Result.output` 是 `E[-1]`，只有 `z[-1]=L` 时才是严格出口。

- `Result.efficiency`：读出窗口内 $|E|^2$ 的梯形积分与整个给定输入时间范围内 $|E_{in}|^2$ 积分之比。它是**有限窗口能量比**，包含所选窗口和坐标的影响，不是自动扣除背景后的量子存储效率。
- `echo`：窗口内振幅模的最大值及对应时间。
- `echo_fidelity(...)`：与时间反转入射曲线的归一化复内积**模，不平方**；按数组点等权内积，不对非均匀采样做积分权重。它不是量子态保真度。
- `transmission_spectrum` 的历史返回值是末时刻**振幅模**，不是强度透射率；必须自行确认达到稳态。

存在多段不连续读出窗口时，直接在筛选后的坐标上做梯形积分会跨越间隔；应逐连续窗口分别积分。公开保留这些低层历史接口，但论文中的物理量必须明确窗口、权重和归一化。
