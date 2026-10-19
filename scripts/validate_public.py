"""生成自包含的解析收敛证据与复场图，不读取源工程数据，不进行训练。"""
import json
import platform
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy
from scipy.special import j0, j1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from gem_solvers import GEMParameters, Drive, solve_gem

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
ASSETS.mkdir(exist_ok=True)
# 优先使用可用的中文字体；没有时仍保留数学坐标与可读取的原始JSON。
for name in ('Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei'):
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False


def analytic(z, t, p):
    x = p.g*p.N*z[:, None]*t[None, :]
    root = np.sqrt(x)
    ratio = np.ones_like(root)
    np.divide(j1(2*root), root, out=ratio, where=root != 0)
    return j0(2*root), 1j*p.g*t[None, :]*ratio


def main():
    p = GEMParameters(g=1, N=2, L=1, eta=0, gamma=0)
    drive = Drive([0, 1], [1, 1], [1, 1])
    rows = []
    for method in ('marcher', 'if2'):
        previous = None
        for n in (17, 33, 65):
            z = np.linspace(0, 1, n)
            t = np.linspace(0, 1, n)
            result = solve_gem(p, z, t, drive, method=method, z_factor=1, t_factor=1)
            exact = analytic(z, t, p)
            errors = [float(np.max(np.abs(a-b))) for a, b in zip((result.E, result.alpha), exact)]
            row = dict(method=method, n=n, max_abs_E=errors[0], max_abs_alpha=errors[1],
                       elapsed_seconds=result.elapsed_seconds)
            if previous is not None:
                row['order_E'] = float(np.log2(previous[0]/errors[0]))
                row['order_alpha'] = float(np.log2(previous[1]/errors[1]))
                assert min(row['order_E'], row['order_alpha']) > 1.8, row
            rows.append(row)
            previous = errors
        assert max(previous) < 1e-3
    fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
    for method in ('marcher', 'if2'):
        subset = [r for r in rows if r['method'] == method]
        for field, marker in (('E', 'o'), ('alpha', 's')):
            ax.loglog([1/(r['n']-1) for r in subset], [r['max_abs_'+field] for r in subset],
                      marker=marker, label=f'{method}: {field}')
    ax.set(xlabel='网格间距 h', ylabel='复场最大绝对误差', title='贝塞尔解析解：空间与时间同步加密')
    ax.grid(True, which='both', alpha=.25)
    ax.legend()
    fig.savefig(ASSETS / 'convergence.png', dpi=180)
    plt.close(fig)
    p_demo = GEMParameters(g=1, N=5, L=1, eta=12, gamma=-.001)
    drive_t = np.linspace(0, 2, 201)
    drive_demo = Drive(drive_t, np.exp(-((drive_t-.35)/.12)**2), np.where(drive_t < 1, 1., -1.))
    demo = solve_gem(p_demo, np.linspace(0, 1, 81), np.linspace(0, 2, 161), drive_demo,
                     method='if2', z_factor=4, t_factor=4)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), constrained_layout=True)
    for ax, field, title in zip(axes.flat, (demo.E.real, demo.E.imag, demo.alpha.real, demo.alpha.imag),
                               ('Re E', 'Im E', 'Re α', 'Im α')):
        limit = max(float(np.max(np.abs(field))), 1e-15)
        mesh = ax.pcolormesh(demo.t, demo.z, field, shading='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.axvline(1, color='black', lw=.8, ls='--')
        ax.set(xlabel='t', ylabel='z', title=title)
        fig.colorbar(mesh, ax=ax, shrink=.85)
    fig.suptitle('IF2 复场示例；虚线为梯度翻转时刻')
    fig.savefig(ASSETS / 'gem_fields.png', dpi=180)
    plt.close(fig)
    report = dict(schema_version=1, description='自包含解析收敛；非研究面板，非训练',
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.system(), machine=platform.machine()),
                  analytic=dict(parameters=asdict(p), drive='Ein=1, sw=1, t in [0,1]', rows=rows),
                  illustration=dict(parameters=asdict(p_demo), output_shape=list(demo.E.shape),
                                    z_factor=4, t_factor=4, native_drive_nodes=201,
                                    pulse_time=.35, pulse_width=.12, switch_time=1),
                  passed=True, training_steps=0)
    (ROOT / 'benchmarks/public_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
