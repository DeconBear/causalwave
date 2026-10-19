# 安装与首个算例

## 最小环境

Python 3.10 及以上；NumPy 2.0 及以上；SciPy 1.13.1 及以上。NumPy 2.0 下限来自观测量接口对 `np.trapezoid` 的使用。无 PyTorch、CUDA、训练权重、云服务或 API 密钥依赖。XMDS2 是单独可选的系统工具链。

```bash
git clone https://github.com/DeconBear/gem-solvers.git
cd gem-solvers
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
gem-solvers doctor --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -m 'not xmds'
gem-solvers demo --method if2 --output outputs/first.npz
```

目前没有发布 PyPI 版本；请使用上述 GitHub 地址。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。CPU 共享时用 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1` 限制库线程数；它们不限制外部 XMDS 程序的所有潜在并行配置。

## 检查结果

`doctor` 输出当前 Python、NumPy、SciPy、包版本以及可选编译器路径。`demo` 输出两个复场的形状、耗时和保存位置；它是有限网格的仿真，不是训练。已有结果文件拒绝覆盖，请换新文件名。读取：

```python
import numpy as np
with np.load("outputs/first.npz") as data:
    E, alpha, z, t = (data[k] for k in ("E", "alpha", "z", "t"))
    print(E.shape, alpha.dtype, z[-1], t[-1])
```

更完整的可运行例子见仓库 `examples/quickstart.py`；将 `method="if2"` 改为 `"marcher"` 可比较两后端。开始新的物理问题时，必须保存输入参数、原生驱动和加密因子，分别检查 $E$ 与 $\alpha$ 的复数误差。

## 文档、图表与构建

```bash
python -m pip install -e '.[test,docs,plot]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_public.py
python scripts/build_llms.py
python -m mkdocs build --strict
python -m mkdocs serve
python -m build
```

科学图使用 Matplotlib 生成；MathJax 从 CDN 渲染公式。离线阅读可查看仓库 Markdown 与 `llms-full.txt` 中的 LaTeX 原文。源码包与 wheel 都应包含 XMDS 模板；CI 检查 Python 版本矩阵，独立任务检查真实 XMDS 工具链。

## 常见问题

|症状|处理|
|---|---|
|导入失败|使用同一个虚拟环境中的 `python -m pip` 和 `python`，检查 `doctor`|
|系统 Python 拒绝 pip 写入|创建 venv，不绕过系统环境保护|
|没有 `xmds2`|两个 Python 后端仍可使用；按 [XMDS 页面](xmds.md)单独安装|
|场误差过大|保留原生驱动，先分别加密空间、时间并比较两个复场；不能只加密输出图片|
|内存不足|先降低测试规模；marcher 存储细网格，IF2 存储方式不同；不要同时启动多组大网格|
|输出目录已存在|新建算例目录，保留旧结果和失败日志|
