# 来源、许可证与引用

## 项目定位与方法归属

CausalWave 是线性因果耦合传播的数值工具。其底层 marcher 接口可表达空间单向传播场与时间演化场之间的线性耦合；目前打包的物理协议与主要验证集中在 GEM 和线性三能级模型。IF2 仍是两能级 GEM 的专用实现。更一般的名称没有扩展代码实际支持的方程、边界条件或验证范围。

本项目不声称发明梯形法、指数积分或 Volterra 求积。marcher 的常系数标量子类可对应到经典 Goursat 梯形格式；IF2 的端点指数权重与既有指数梯形求积一致。具体推导对应关系见[方法页](methods.md)，原始来源见[Gourlay（1970）](references.md#gourlay1970)与[Hochbruck、Ostermann（2010）](references.md#hochbruck2010)。这些对应关系不等于证明整个软件与某一已有实现逐行相同，也不构成算法组合的原创性认证。

本项目提供的是代码整理、方程接口、事件与坐标处理、结构化数值求解和可复现验证。评价这些工作应依据代码、测试与同问题的实测结果；不把软件开发贡献混同于基础数值方法的发明。

## 代码来源与提取边界

数值内核提取自作者的 `DeconBear/GEM_PINN` 研究工程，基点为 `c117869`。源文件及其 SHA-256 摘要记录在本仓库 `benchmarks/source_manifest.json`；发布版的文件摘要另列为 `distribution_sha256`，区分源文件与接口整理后的文件。

0.2.0 的更名与发布审查还修复了输入校验、无效参数覆盖、多段窗口积分及 XMDS 中断清理；这些变化记录在版本历史中。源摘要保持不变，发行摘要对应当前文件，不能将二者混用。

- marcher：`core.py`、`protocols.py`、`observables.py`；保留算法和物理语义，整理包入口与过期说明。
- IF2：`physics_operator.py` 与 `physics_operator_cached.py`；作为内部模块提取，去除对研究包路径的导入依赖。
- 新增：统一公开 API、安装元数据、XMDS 接口与自编模板、文档、CI、独立验证脚本。
- 未带入：FNO/PINN、训练权重、训练数据、隐藏测试标签、后台任务、原研究参考目录、外部模板或编译后的程序。

新仓库可以在没有 `GEM_PINN` 的机器上安装运行。研究工程未来变化不会自动改变本发行版。没有必要复制整个研究目录来安装这个数值工具。

## 历史证据与再现边界

`benchmarks/historical_summary.json` 是源实验 exp_123、exp_124、exp_125 的有限汇总，含源路径和报告摘要。原始冻结面板及其完整参考数据不随本仓库发布，因此这些历史数字不能由新仓库单独完整重跑。新仓库能够自包含复现的是 `scripts/validate_public.py` 与 `tests/` 的解析/制造解及小型跨实现检查。

GitHub 上的源工程若权限变化，不影响本包运行。引用历史结果时需要保留这个数据可得性边界，而不能把链接存在当作公开数据齐全。

## 许可

本项目以 MIT 发布，见仓库 `LICENSE`。外部 NumPy、SciPy、Matplotlib、MkDocs 和 XMDS2 分别遵循其许可证；不捆绑这些项目的源码或二进制，不用本项目 MIT 重新许可它们。XMDS 输入模板为按声明方程重新编写的示例；不是对原物理参考目录模板的重新许可。

## 如何引用与复现实验

引用本软件可使用根目录 `CITATION.cff`，同时记录准确 commit 或发行版本。当前不虚构 DOI 或同行评审状态。论文方法至少应写明：方程与符号、两复场输出、单位、参数、原生驱动、真实坐标、空间/时间加密、误差归一化、独立参考精度、软硬件版本、线程数及是否计入编译。

引用软件版本、基础数值方法和物理模型，分别说明复现对象、方法来源与适用假设。建议按实际使用内容选择[参考文献页](references.md)中的条目，而不是把所有相关项目都列为依赖或性能基准：

- 使用两种 Python 后端时，记录 CausalWave 版本及对应的梯形/指数求积来源。
- 使用 XMDS2 生成或运行参考结果时，另引 [Dennis 等（2013）](references.md#xmds2013)，并记录编译器、XMDS2 版本和输入文件。
- 将结果解释为 GEM 或 EIT 时，引用对应物理模型，并明确弱场、单向传播和可能的绝热消除近似。
- CoOMBE、clerq 与 mbsolve 是相邻软件，并非本项目内置后端；未进行同精度比较时，不宣称本项目优于它们。

实现接口遵循 [SciPy `solve_banded` 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html)；站点发布机制见 [GitHub Pages 自定义工作流文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。软件验证不等于真实器件实验验证，参考文献也不替代本地收敛证据。
