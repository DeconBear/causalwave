# 给 LLM 和用户自有 agent 的执行指南

本页面是安装与验证合同，不要求特定模型、付费 API 或 agent 平台。先读仓库根目录 `AGENTS.md`、`llms.txt` 和本页。安装只需普通 CPU；不要启动训练、GPU 作业、定时器或云机器。

## 可直接复制的用户请求

> 请从 https://github.com/DeconBear/causalwave 安装 CausalWave。在新目录克隆，阅读 AGENTS.md 和 llms.txt，创建独立虚拟环境，安装 test 依赖，运行 doctor、非 XMDS 测试和一次小型 IF2 demo。记录版本、命令、退出码与输出路径。不要改全局 Python，不要训练或用 GPU。如果我还要求 XMDS 仿真，请先检查系统工具链；缺失时说明需要安装哪些软件，不擅自 sudo。只根据真实执行结果报告是否成功。

## 最小执行序列

以下命令以 POSIX shell 为例；Windows 的激活语法见[安装页](installation.md)。必须在克隆目录运行测试。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/causalwave doctor --json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -m 'not xmds'
.venv/bin/causalwave demo --method if2 --output outputs/agent-first.npz
```

成功标准：安装命令退出码为零，doctor 可导入依赖，测试完成且无失败，demo 保存有限复数组。测试数可能随版本变化，**不能硬编码一个旧数字来伪造验收**。若相同文件已存在，保留文件并改用新的输出名。

## 可选 XMDS 验收

```bash
xmds2 --help
.venv/bin/python -m pytest -q -m xmds
.venv/bin/causalwave xmds-example --directory outputs/agent-xmds-first
```

只有实际安装工具链并运行成功才能称为 XMDS 已验收。因依赖缺失而 skipped 与 passed 不同。查看 `compile.log`、`simulation.log`、`run.json` 定位失败；不要删除日志后重复声称成功。不要把随机网上的 XMDS XML 当成纯配置：里面可以含可执行 C++。

## 不得改写的科学语义

- 两能级 `GEMParameters.gamma`、低层 `GEM.gamma0` 是正号项的系数；衰减时传负值。`LambdaGEM.gamma0` 的方程自带负号，不能混淆。
- 输入复相位必须保留，验证 E 和 alpha 两个复场，不能改成只比较强度来让误差变小。
- 原生驱动与输出采样分开。事件点不能丢失，不能把解析高斯和粗采样折线称作完全相同的输入。
- 使用真实坐标。XMDS 最后一行通常不等于出口 $L$。
- 默认网格、测试通过和历史24例通过，都不等于任意新问题达到1%。
- 不读取隐藏研究数据，不扫描用户密钥，不把结果上传第三方。普通安装不需要账号令牌。

## 排错顺序与交付

先检查解释器/包版本，再检查输入合同与坐标，再查看失败日志，最后才考虑修改代码。改动应最小化并增加有意义的回归测试。代理汇报至少包含：来源 URL 与 commit、虚拟环境位置、实际通过/跳过/失败情况、结果文件路径、仍未验证的外部组件。不把“文件已生成”替代“计算已通过”。

[llms.txt](llms.txt) 提供入口；[llms-full.txt](llms-full.txt) 是这些文档的纯文本合并版，可无需 JavaScript 阅读。
