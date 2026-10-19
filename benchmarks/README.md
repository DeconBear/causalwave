# CausalWave 0.2.0 发行验证记录

本次整理不训练、不读取隐藏测试、不使用GPU。以下均为已完成的本地检查；远端CI状态以GitHub Actions为准。

- 全部测试：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`，**97 passed，59.47秒，0 skipped**，含3个真实XMDS标记测试。新增44个回归用例覆盖实输入、形状、参数覆盖、多窗口积分和中断清理。
- Python环境：3.13.11 / NumPy 2.5.4 / SciPy 1.18.1；XMDS2 3.1.0。
- `scripts/validate_public.py`：解析解网格加密通过，结果见 `public_validation.json`；图见 `docs/assets/`。
- `examples/linear_system.py`：非GEM的通用线性系统解析示例通过；U/V最大绝对误差约2.36e-4/2.08e-4，仅代表该示例网格。
- `python -m build`：源码分发包和wheel构建成功。独立临时虚拟环境安装wheel，在源目录之外运行doctor与IF2演示并读取包内XMDS模板成功，见 `packaging_validation.json`。
- `mkdocs build --strict`：无文档构建错误；参考文献、模型边界、方法出处和LLM纯文本入口已同步。在线部署状态以GitHub Pages实际页面为准。
- `source_manifest.json`：源代码提取与发行文件摘要；`historical_summary.json`：有边界说明的历史报告汇总，非本次新测量。

这份记录不是全域精度认证。CI会测试Python 3.10/3.12/3.13，以及Ubuntu系统XMDS工具链。公开安装不依赖源研究工程、训练权重和原始数据。

本次全仓库审查覆盖数值内核、公开API、XMDS子进程、打包/CI、文献归属和当前文件及初始Git历史。发现的输入、窗口和中断缺陷已修复；未发现常见密钥、私钥、个人绝对路径或大型研究产物进入发布内容。审查及测试不能作为任意输入正确性或绝对安全性的保证。
