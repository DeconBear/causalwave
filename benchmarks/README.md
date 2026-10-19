# 发行验证记录

本次整理不训练、不读取隐藏测试、不使用GPU。以下均为已完成的本地检查；远端CI状态以GitHub Actions为准。

- 全部测试：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`，**53 passed，61.39秒**，含3个真实XMDS标记测试。
- Python环境：3.13.11 / NumPy 2.5.4 / SciPy 1.18.1；XMDS2 3.1.0。
- `scripts/validate_public.py`：解析解网格加密通过，结果见 `public_validation.json`；图见 `docs/assets/`。
- `python -m build`：源码分发包和wheel构建成功。独立临时虚拟环境安装wheel，在源目录之外运行doctor与IF2演示并读取包内XMDS模板成功，见 `packaging_validation.json`。
- `mkdocs build --strict`：无文档构建错误。浏览器检查方法页84处数学公式，无MathJax错误节点，中文与公式正常显示。
- `source_manifest.json`：源代码提取与发行文件摘要；`historical_summary.json`：有边界说明的历史报告汇总，非本次新测量。

这份记录不是全域精度认证。CI会测试Python 3.10/3.12/3.13，以及Ubuntu系统XMDS工具链。公开安装不依赖源研究工程、训练权重和原始数据。
