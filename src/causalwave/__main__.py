"""命令行自检、轻量演示和XMDS示例。"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from . import Drive, GEMParameters, solve_gem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="环境自检；不安装系统软件、不启动仿真")
    doctor.add_argument("--json", action="store_true")
    demo = sub.add_parser("demo", help="CPU轻量GEM演示，不训练")
    demo.add_argument("--method", choices=("if2", "marcher"), default="if2")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo.npz"))
    xmds = sub.add_parser("xmds-example", help="编译并运行自带XMDS2示例")
    xmds.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        report = {"python": sys.version.split()[0], "numpy": version("numpy"),
                  "scipy": version("scipy"), "causalwave": version("causalwave"),
                  "xmds2": shutil.which("xmds2"), "cxx": shutil.which("g++"),
                  "numerical_backends": ["if2", "marcher"], "gpu_required": False,
                  "note": "XMDS2为可选外部依赖，缺失不影响两个Python数值后端"}
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "demo":
        params = GEMParameters(g=1, N=5, L=1, eta=12, gamma=-.001)
        drive_t = np.linspace(0, 2, 201)
        drive = Drive(drive_t, np.exp(-((drive_t-.35)/.12)**2), np.where(drive_t<1, 1., -1.))
        result = solve_gem(params, np.linspace(0, 1, 61), np.linspace(0, 2, 121), drive,
                           method=args.method, z_factor=2, t_factor=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output)
        print(json.dumps(dict(method=result.method, shape=list(result.E.shape),
                              seconds=result.elapsed_seconds, output=str(args.output)), ensure_ascii=False))
    else:
        from .xmds import run_gem
        result = run_gem(args.directory)
        print(json.dumps(dict(directory=str(result.directory), compile_seconds=result.compile_seconds,
                              run_seconds=result.run_seconds), ensure_ascii=False))


if __name__ == "__main__":
    main()
