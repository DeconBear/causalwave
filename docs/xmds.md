# XMDS2：生成、编译、运行与读取

XMDS2 是独立的微分方程代码生成工具。本项目不捆绑其编译器或二进制，也不修改原研究工程的参考目录。模板根据这里声明的 GEM 方程重新编写；并非复制外部研究者未明确许可的模板。

## 安装与检查

在提供系统包的 Debian/Ubuntu 上：

```bash
sudo apt-get update
sudo apt-get install xmds2 g++ libfftw3-dev
xmds2 --help
gem-solvers doctor --json
python -m pytest -q -m xmds
```

系统安装需要用户自行授权。Python 求解器不依赖这个步骤；不要用 `pip install xmds2` 代替完整工具链。本地集成已在 XMDS2 3.1.0 上验证；CI 在 Ubuntu 系统包上运行。历史官方[手册](https://xmds2.readthedocs.io/en/latest/)可用于理解输入结构，但其版本较旧，具体能力应以实际安装版本为准。

## 配置与调用

```python
from gem_solvers import GEMParameters
from gem_solvers.xmds import XMDSConfig, render_gem, run_gem

config = XMDSConfig(
    params=GEMParameters(g=1, N=2, L=1, eta=4, gamma=-0.01),
    duration=2, switch_time=1,
    pulse_time=0.3, pulse_width=0.1, amplitude=1,
    lattice=128, z_samples=64,
    samples_per_stage=40, steps_per_stage=1000, tolerance=1e-8,
)
xml = render_gem(config)
run = run_gem("outputs/new-xmds-case", config, timeout=120)
fields = run.read()
print(fields.E.shape, fields.alpha.shape, fields.z[-1])
print(run.compile_seconds, run.run_seconds)
```

入射场为实振幅高斯 $u(t)=A\exp[-(t-t_0)^2/\sigma^2]$，直接在生成程序中计算。写入/读出是两个独立积分块，梯度在 `switch_time` 精确切换。时间推进用自适应 ARK89，空间 cross-propagation 用 RK4；`steps_per_stage` 是交给 XMDS 的步数设置，不能解释为自适应运行实际接受的固定步数。`lattice` 须为 `z_samples` 的整数倍。

每个调用使用**新目录**，包含：

|文件|含义|
|---|---|
|`gem.xmds`|实际编译输入|
|`compile.log`|生成与编译输出|
|`simulation.log`|运行日志|
|`run.json`|输入 SHA-256、命令、状态、编译/运行耗时|
|`gem.xsil`|本模板定义的 ASCII 复场数据|
|生成程序及中间文件|由 XMDS 工具链产生，不提交到仓库|

`timeout` 对编译和运行阶段分别生效，超时后终止本次子进程组并保留日志。报错不会删掉失败目录，也不会覆盖旧结果。

## 真实坐标与输出合同

此模板单个采样组同时输出 `E_real,E_imag,alpha_real,alpha_imag`，读取时按实际 `z,t` 组装 `(nz,nt)` 复数组。读取器不凭 `L/T` 重建坐标、不用 `linspace` 冒充真实节点。

XMDS 的这个空间网格通常**不包含右端点 $L$**；例如默认输出最后一行为 $z=0.984375L$。`fields.E[-1]` 因此不能直接称作 $z=L$ 的出口场。本模板 `initial_sample="no"`，第一条输出时间大于零；统一 Python 后端可在同一组实际坐标上求解，从零初态推进到这些节点。

`read_gem_ascii(path)` 仅支持本模板的内嵌 Text/ASCII XSIL、完整矩形坐标以及指定字段；不承诺读取任意 XMDS 二进制、HDF5、多采样组。单独读取时没有运行耗时，返回 `elapsed_seconds=NaN`；通过 `run.read()` 得到的耗时为编译与运行之和。

## 自定义可信输入

```python
from gem_solvers.xmds import run_file
run = run_file("examples/gem.xmds", "outputs/custom-case")
fields = run.read()
```

`run_file` 要求输入自包含且 `<name>gem</name>`，不会自动复制外部数据文件。输出读取仍受上述格式限制。XML 可以嵌入任意 C++，应当与运行一段本地源代码一样只使用可信文件；独立工作目录不是安全沙箱。

`examples/gem.xmds` 为可直接编译的静态示例；参数化模板位于 `src/gem_solvers/templates/gem.xmds`。模板中的中文说明以 XML 字符实体输出，兼容部分 XMDS 安装对源文本 ASCII 编码的要求。

## 与 Python 后端比较

XMDS 示例使用解析高斯，Python API 使用原生节点的分段线性曲线，因此对照时需要足够密的原生驱动，并单独检查驱动逼近误差。集成测试用同一实际输出坐标、精确翻转时刻、4001 个驱动节点，检查有耦合情况下两个复场的相对误差。它是一个小规模跨实现一致性检查，不能替代大参数域的精度验收，也不自动建立同精度速度排名。
