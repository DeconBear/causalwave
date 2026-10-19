"""从版本化中文文档生成LLM可直接读取的纯文本入口。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
PAGES = ('index', 'installation', 'methods', 'validation', 'api', 'xmds', 'agents', 'provenance')
summary = '''# GEM Solvers

> 无需训练/GPU的线性GEM复场数值求解。Python>=3.10；NumPy>=2.0、SciPy>=1.13.1。

## 安装与验证
- [仓库与README](https://github.com/DeconBear/gem-solvers)
- [AGENTS.md](https://github.com/DeconBear/gem-solvers/blob/main/AGENTS.md)
- [安装](https://deconbear.github.io/gem-solvers/installation/)
- [agent执行合同](https://deconbear.github.io/gem-solvers/agents/)
- [完整纯文本](https://deconbear.github.io/gem-solvers/llms-full.txt)

## 科学与接口
- [推导](https://deconbear.github.io/gem-solvers/methods/)
- [验证及边界](https://deconbear.github.io/gem-solvers/validation/)
- [API](https://deconbear.github.io/gem-solvers/api/)
- [XMDS2](https://deconbear.github.io/gem-solvers/xmds/)
- [来源](https://deconbear.github.io/gem-solvers/provenance/)

## 关键合同
两能级gamma为+gamma*alpha的有符号系数，衰减请传负值；LambdaGEM有不同的负号约定。
入射场复数分段线性，sw实数左常值，事件必须在原生驱动中。保留两复场与实际坐标。
默认网格不保证1%。无PyPI发行；从上述GitHub地址安装。XMDS另装，不是必须依赖。
执行测试需报告真实退出码；不训练、不使用GPU、不默认sudo、不删除失败日志。
'''
(ROOT / 'llms.txt').write_text(summary, encoding='utf-8')
(DOCS / 'llms.txt').write_text(summary, encoding='utf-8')
full = ['# GEM Solvers 完整说明\n\n由 scripts/build_llms.py 生成；相对链接按来源页面解析。\n']
for page in PAGES:
    full.append(f'\n\n---\n来源：https://deconbear.github.io/gem-solvers/{page + "/" if page != "index" else ""}\n\n')
    full.append((DOCS / f'{page}.md').read_text(encoding='utf-8'))
(DOCS / 'llms-full.txt').write_text(''.join(full), encoding='utf-8')
