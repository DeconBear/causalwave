# 来源、许可证与引用

## 代码来源与提取边界

数值内核提取自作者的 `DeconBear/GEM_PINN` 研究工程，基点为 `c117869`。源文件及其 SHA-256 摘要记录在本仓库 `benchmarks/source_manifest.json`；发布版的文件摘要另列为 `distribution_sha256`，区分源文件与接口整理后的文件。

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

相关外部方法资料：

1. G. R. Dennis, J. J. Hope, M. T. Johnsson, *XMDS2: Fast, scalable simulation of coupled stochastic partial differential equations*, Computer Physics Communications **184**, 201–208 (2013), [DOI](https://doi.org/10.1016/j.cpc.2012.08.016)。[XMDS 官方文档](https://xmds2.readthedocs.io/en/latest/)也给出该引用；文档镜像较旧，不能据此判断当前版本。
2. [SciPy `solve_banded` 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html)：IF2 实现使用的带状矩阵存储与求解接口。
3. [GitHub Pages 自定义工作流文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)：本站的构建和发布方式。

[方法页](methods.md)中的推导解释本仓库实际代码；与原子物理模型对应的物理论文应由具体研究按所用近似另行引用。我们不将软件验证等同于真实器件实验验证。
