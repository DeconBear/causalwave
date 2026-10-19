# gem-solvers

**无需训练、无需GPU的梯度回波记忆（GEM）复场数值求解器。**

本项目提供两个可独立使用的Python后端：**marcher**（沿时空网格对角线的耦合梯形推进）和 **IF2**（空间Volterra积分＋二阶指数拟合时间推进），并提供可选的 **XMDS2** 输入生成、隔离编译运行和结果读取接口。

[方法与使用文档](https://deconbear.github.io/gem-solvers/) · [给agent的安装指南](docs/agents.md) · [数学推导](docs/methods.md) · [验证与限制](docs/validation.md) · [问题反馈](https://github.com/DeconBear/gem-solvers/issues)

> **物理符号必须先读：** 本项目保持源工程方程 `alpha_t = (+gamma + i*sw*eta*(z-L/2))*alpha + i*g*E`。`gamma` 是有符号系数，正值表示增长；需要正物理退相干率 `Gamma` 时，请传 `gamma=-Gamma`。不会静默替换方程。所有结果保留复数，不只输出强度。

## 能做什么

|入口|用途|依赖与边界|
|---|---|---|
|`gem_solvers.solve_gem(..., method="if2")`|GEM两能级完整复场；低内存数值基线|NumPy、SciPy；无训练和权重|
|`gem_solvers.solve_gem(..., method="marcher")`|与IF2同输入、同输出的GEM对照|NumPy；本发行包同时安装SciPy|
|`marcher.GEM / LambdaGEM / Protocol`|底层GEM、Λ-GEM/EIT与一般线性耦合系统|保留协议定制接口；Λ-GEM的系数定义见源码|
|`gem_solvers.xmds`|生成GEM示例、编译运行可信XMDS文件、读复场|需另装XMDS2、C++与FFTW；不是pip依赖|

不是神经代理、不包含PINN/FNO训练、没有后台任务。不适用于未经重新推导的非线性介质、反射双向传播、扩散或任意几何域。

## 安装（Python 3.10及以上）

当前从GitHub源码安装，**尚未发布PyPI包**。请勿依赖同名第三方包。

```bash
git clone https://github.com/DeconBear/gem-solvers.git
cd gem-solvers
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
gem-solvers doctor --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -m 'not xmds'
gem-solvers demo --method if2 --output outputs/demo.npz
```

PowerShell激活命令为 `.venv\Scripts\Activate.ps1`；Python数值后端无需XMDS2。XMDS集成主要面向Linux，Windows建议在WSL中安装外部工具链。所有命令均在项目根目录运行，已安装的Python API可从任意目录导入。

如交给自己的agent安装，可复制这段请求：

> 请安装 https://github.com/DeconBear/gem-solvers 。先阅读仓库 AGENTS.md、llms.txt 和 docs/agents.md，在独立虚拟环境中安装，运行 doctor、非XMDS测试和小型IF2演示，报告实际输出。不训练模型、不使用GPU、不默认sudo安装系统软件。若我需要XMDS，再检查外部依赖并运行对应集成测试。

## Python快速开始

```python
import numpy as np
from gem_solvers import GEMParameters, Drive, solve_gem

p = GEMParameters(g=1, N=5, L=1, eta=12, gamma=-0.001)
drive_t = np.linspace(0, 2, 201)
drive = Drive(
    drive_t,
    np.exp(-((drive_t - 0.35) / 0.12)**2),
    np.where(drive_t < 1.0, 1.0, -1.0),
)
z = np.linspace(0, p.L, 61)
t = np.linspace(0, 2, 121)
result = solve_gem(p, z, t, drive, method="if2", z_factor=2, t_factor=2)
print(result.E.shape, result.alpha.shape)  # 都是 (61, 121)，复数
print(result.elapsed_seconds)
# result.save("result.npz")  # 文件存在时拒绝覆盖
```

将 `method` 改为 `"marcher"` 即可对照。参数与网格数目只是轻量演示，不能当成任意条件下的精度配置。可运行 `python examples/quickstart.py`。

### 输入、输出与事件合同

- 统一接口：`z[0]=0`、`z[-1]≤L`，`z/t`严格递增且至少两点，输出形状统一为 `(nz, nt)`。
- 驱动从物理 `t=0` 开始并覆盖输出终点；`Ein`可为复数，节点间线性插值；`sw`为实数，节点间左常值保持。
- **驱动网格和输出网格独立。** 翻转等事件必须出现在驱动节点，不能把原生驱动降采样成输出曲线。
- `alpha(z,0)=0`，`E(0,t)=Ein(t)`。缺省加密 `z_factor=4,t_factor=2` 是计算起点，不是误差承诺。
- `E[-1]` 是最后请求位置的场；只有 `z[-1]=L` 才是严格出口。XMDS样例通常不采样右端点，读取器保留实际坐标。

## XMDS2：可选，不影响两个Python后端

Debian/Ubuntu提供系统包时，可由用户自行安装：

```bash
sudo apt-get update
sudo apt-get install xmds2 g++ libfftw3-dev
xmds2 --help
gem-solvers xmds-example --directory outputs/xmds-first
python examples/xmds_gem.py --directory outputs/xmds-python
python -m pytest -q -m xmds
```

```python
from gem_solvers import GEMParameters
from gem_solvers.xmds import XMDSConfig, run_gem

config = XMDSConfig(params=GEMParameters(g=1, N=2, L=1, eta=4, gamma=-0.001))
run = run_gem("outputs/new-case", config)  # 必须是新目录
fields = run.read()
print(fields.z, fields.t, fields.E.shape)
print(run.compile_seconds, run.run_seconds)
```

每次保留XML、编译日志、运行日志、源摘要与耗时；不会自动删除失败目录。自带模板使用高斯入射、一次精确分段的梯度翻转、ASCII复场输出。通用 `run_file` 仅运行可信且自包含、`<name>gem</name>` 的输入；其数据读取器只支持本项目约定的ASCII字段。XML中可嵌入C++，本接口不是安全沙箱。详见 [XMDS接口](docs/xmds.md)。

## 验证和性能：区分可复现测试与历史数据

本仓库附带贝塞尔解析解、制造解、独立ODE、复线性、边界/初态、真实事件、网格收敛、缓存一致和XMDS集成测试。测试通过不能证明所有参数都满足1%。

源项目的一份24例冻结面板中，IF2 `z_factor=16,t_factor=2` 的两复场逐例误差均小于1%；单核中位耗时2.970秒。另一次3例独立进程测量的峰值RSS约0.605 GiB。这些是**历史测量，不是本项目对任意设备/输入的保证**；源面板和完整参考数据不随包分发，复现信息与缺失数据边界见 [验证页面](docs/validation.md)。

XMDS历史计时与该面板的网格/驱动不同，不能据此宣称严格的3.8倍同精度加速。新问题必须分别加密空间和时间，并检查两个复场的收敛。marcher保留细时空系数和场，加密可能显著增加内存；IF2只保留当前内部状态和请求输出。

## 文档与开发

```bash
python -m pip install -e '.[docs,plot,test]'
python scripts/validate_public.py       # 解析验证及论文页图表；无训练
python -m mkdocs build --strict
python -m mkdocs serve
python -m build
```

- [原理与推导](docs/methods.md)：方程、离散块系统、指数权重、双对角求解、误差与复杂度。
- [API](docs/api.md)：参数、返回值、底层协议与文件格式。
- [给agent的说明](docs/agents.md)、[AGENTS.md](AGENTS.md)、[llms.txt](llms.txt)：可执行安装、验证及故障定位。
- [来源与引用](docs/provenance.md)：代码提取边界、许可证、参考文献。
- CI在Python 3.10/3.12/3.13测试数值后端；独立XMDS任务安装系统工具链。GitHub Pages在主分支构建成功后部署。

MIT许可证适用于本项目代码；XMDS2是独立外部软件，遵循其自己的许可证，本仓库不捆绑其源码/二进制，也不重新许可原研究者的模板。科学使用请保留模型、参数、驱动定义、真实坐标、版本、加密因子及收敛证据。引用入口见 [CITATION.cff](CITATION.cff)。
