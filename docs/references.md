# 参考文献与相关软件

本页区分基础数值方法、相邻软件与物理模型。列出文献表示说明来源与范围，不表示 CausalWave 复现了其中全部结果、实现了所有模型，或已经完成相互性能比较。软件名称与引用信息核对日期为 **2026-10-20**。

## 基础数值方法

### Gourlay（1970）：梯形法与 Goursat 问题 {#gourlay1970}

A. R. Gourlay, “A note on trapezoidal methods for the solution of initial value problems,” *Mathematics of Computation* **24**, 629–633 (1970). [DOI：10.1090/S0025-5718-1970-0275680-3](https://doi.org/10.1090/S0025-5718-1970-0275680-3)；[AMS 原文](https://www.ams.org/journals/mcom/1970-24-111/S0025-5718-1970-0275680-3/S0025-5718-1970-0275680-3.pdf)。

该文 §2 讨论 Goursat 问题的梯形差分。CausalWave 的 marcher 在常系数、标量且无局部演化项的子类中，经消元可得到对应的四角梯形构造。这是特定子类的代数对应，不是把一般变系数矩阵实现完全归属于该文，也不是声称该文首次提出所有相关方法。

### Hochbruck 与 Ostermann（2010）：指数积分 {#hochbruck2010}

M. Hochbruck and A. Ostermann, “Exponential integrators,” *Acta Numerica* **19**, 209–286 (2010). [DOI：10.1017/S0962492910000048](https://doi.org/10.1017/S0962492910000048)；[作者公开原稿](https://na.math.kit.edu/download/papers/acta-final.pdf)。

§2.2 的 Example 2.6 给出指数梯形求积端点权重；取节点为 0 和 1 后，就是本项目 IF2 使用的 $\varphi_1-\varphi_2$ 与 $\varphi_2$。本项目在这一时间求积上处理 GEM 空间累积积分所产生的隐式耦合。引用该文不表示其全部误差或稳定性定理已经逐项验证适用于当前问题；IF2 是项目简称，不能凭名称与其他积分因子方法等同。

## 相邻软件与工具链

这些项目的方程层次和适用近似不同。选择比较对象时，应先匹配模型与精度，再比较时间和内存。CausalWave 当前没有提供 CoOMBE、clerq 或 mbsolve 的同精度基准测试。

| 软件 | 来源说明的范围 | 与 CausalWave 的关系 |
|---|---|---|
| CoOMBE | 光学 Bloch/Lindblad 演化，以及旋波近似下一个或两个共传播光场的一维 Maxwell–Bloch 方程 | 更广的原子模型软件；弱探针处理与本项目相邻，但方程和归一化须逐项匹配 |
| clerq（原 MaxwellBloch） | 基于 QuTiP 的多能级开放量子系统与经典 Maxwell 传播，含衰减、退相干及 Doppler 展宽 | 适合需要密度矩阵和更一般原子结构的问题；不是 CausalWave 的依赖或内置协议 |
| mbsolve | 一维 Maxwell 方程与 Lindblad 方程耦合，不采用旋波近似的全波处理 | 比当前单向弱场包络模型包含更多物理；不能直接用默认计时比值比较优劣 |
| XMDS2 | 为常微分、偏微分和随机微分方程生成数值仿真程序 | 可选外部工具链；本项目提供受限 GEM 示例及 Python 调用、读取接口 |

### CoOMBE {#coombe2025}

R. M. Potvliege and S. A. Wrathmall, “CoOMBE: A suite of open-source programs for the integration of the optical Bloch equations and Maxwell-Bloch equations,” *Computer Physics Communications* **306**, 109374 (2025). [DOI：10.1016/j.cpc.2024.109374](https://doi.org/10.1016/j.cpc.2024.109374)；[作者预印本及期刊信息](https://arxiv.org/abs/2406.19144)。

其作者说明包括 Fortran 90 实现、稳态密度矩阵、速率方程极限与弱探针近似。这里的 2025 是期刊卷期年份，DOI 中的 2024 与此不冲突。

### clerq / MaxwellBloch {#clerq}

T. P. Ogden, *clerq* (2026), Zenodo，软件概念 DOI：[10.5281/zenodo.23269881](https://doi.org/10.5281/zenodo.23269881)。[作者仓库与当前引用说明](https://github.com/tpogden/clerq)；[官方重命名说明](https://clerq.org/news/rename.html)。

该概念 DOI 覆盖多个版本；复现时仍应记录实际版本，能取得对应版本 DOI 时使用版本 DOI。旧文献和旧结果可能使用 MaxwellBloch 名称。官方说明还指出 0.13.0 修正了传播方程中的系数；历史结果与新版本比较时应核对方程约定，不能仅替换软件名称。此条是软件归档引用，不将其称为一篇已经核实的期刊论文。

### mbsolve {#mbsolve2021}

M. Riesch and C. Jirauschek, “mbsolve: An open-source solver tool for the Maxwell-Bloch equations,” *Computer Physics Communications* **268**, 108097 (2021). [DOI：10.1016/j.cpc.2021.108097](https://doi.org/10.1016/j.cpc.2021.108097)；[作者预印本](https://arxiv.org/abs/2005.05412)；[项目仓库](https://github.com/mriesch-tum/mbsolve)。

该论文采用全波 Maxwell–Lindblad 模型并给出并行求解实现。它是相关研究软件的先例，不是当前 GEM 近似下已经对齐的性能基准。

### XMDS2 {#xmds2013}

G. R. Dennis, J. J. Hope, and M. T. Johnsson, “XMDS2: Fast, scalable simulation of coupled stochastic partial differential equations,” *Computer Physics Communications* **184**, 201–208 (2013). [DOI：10.1016/j.cpc.2012.08.016](https://doi.org/10.1016/j.cpc.2012.08.016)；[官方文档及引用说明](https://xmds2.readthedocs.io/en/latest/)。

所链接文档镜像包含较旧的版本说明，不能据此确定本机或最新发布版本。CausalWave 不捆绑 XMDS2；其模板、工具链和结果格式边界见 [XMDS2 接口](xmds.md)。

## 物理模型背景

这些文献帮助识别当前协议的近似和应用语境。CausalWave 的数值验收见[验证页](validation.md)；并未由下面文献获得真实器件、量子噪声或量子态保真度方面的验证。

### Hétet 等（2008）：两能级 GEM {#hetet2008}

G. Hétet, J. J. Longdell, A. L. Alexander, P. K. Lam, and M. J. Sellars, “Electro-Optic Quantum Memory for Light Using Two-Level Atoms,” *Physical Review Letters* **100**, 023601 (2008). [DOI：10.1103/PhysRevLett.100.023601](https://doi.org/10.1103/PhysRevLett.100.023601)。

这是两能级梯度回波协议的物理背景引用。软件使用的场归一化、参数单位和有符号衰减系数仍以[明确写出的方程](methods.md)为准。

### Hosseini 等（2012）：Raman GEM {#hosseini2012}

M. Hosseini, B. M. Sparkes, G. T. Campbell, P. K. Lam, and B. C. Buchler, “Storage and manipulation of light using a Raman gradient-echo process,” *Journal of Physics B: Atomic, Molecular and Optical Physics* **45**, 124004 (2012). [DOI：10.1088/0953-4075/45/12/124004](https://doi.org/10.1088/0953-4075/45/12/124004)。

用于理解 Raman GEM 与有效模型的应用背景。把三能级体系约化为两能级方程时，必须另外检查失谐、控制场和绝热消除条件；一个类名或同一组符号不能替代这一步。

### Fleischhauer 与 Lukin（2000）：EIT 与暗态极化激元 {#fleischhauer2000}

M. Fleischhauer and M. D. Lukin, “Dark-State Polaritons in Electromagnetically Induced Transparency,” *Physical Review Letters* **84**, 5094–5097 (2000). [DOI：10.1103/PhysRevLett.84.5094](https://doi.org/10.1103/PhysRevLett.84.5094)；[出版方摘要](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.5094)。

提供弱场 EIT 中光与集体相干性耦合的背景。当前 `LambdaGEM` 的线性三能级实现和有限解析透射测试，不等同于已经复现所有光存储过程或量子态性质。

## 数值接口

[SciPy `solve_banded` 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html)规定 IF2 使用的带状矩阵存储和求解接口。需要复现实验时，应同时记录 NumPy/SciPy、Python、CausalWave 版本及线程设置，不能只引用方法论文。
