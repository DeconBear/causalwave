# 给安装和开发agent的合同

这是独立数值求解项目，不含训练。先读README、docs/agents.md及docs/methods.md。

1. 使用项目独立 `.venv`；不得依赖原GEM_PINN工程或操纵用户其他环境。Python>=3.10，NumPy>=2.0，SciPy>=1.13.1。不要默认安装CUDA/PyTorch。
2. `python -m pip install -e '.[test]'` → `causalwave doctor --json` → `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -m 'not xmds'` → `causalwave demo --output outputs/check.npz`。只在返回码成功、结果有限后报告完成。
3. XMDS2是可选系统工具。未安装时报告缺项，先完成Python后端；不默认使用sudo或运行任意下载脚本。用户要求XMDS时按docs/xmds.md处理并执行集成测试。
4. 不训练、不下载权重、不启动后台任务。测试和小规模数值仿真有界运行，CPU单线程优先；大网格marcher会消耗显著内存，需先评估。
5. 必须保留有符号+gamma方程、复数Ein、sw左保持、真实驱动事件以及输出坐标。不得为通过测试改变物理符号、误差门槛或丢弃虚部。
6. 不覆盖已存在的仿真目录或结果；保留失败日志。XMDS文件可含可执行C++，只运行用户信任的源文件。
7. 代码/文档解释使用中文，标识符与外部协议保持原样。数值改动需解析或独立参考回归；修改文档需`mkdocs build --strict`。
8. 不把局部解析检查或历史面板通过称任意输入精度保证，不把不匹配网格的计时比值称公平加速。
9. 不公开密钥、个人路径、训练数据或原工程大型产物。发布/仓库权限更改须来自用户明确任务授权。
