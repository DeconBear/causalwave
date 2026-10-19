"""生成自包含GEM示例，调用外部XMDS2，并读取其ASCII XSIL输出。

本模块不捆绑XMDS2。XMDS文件包含可执行C++，只能运行可信文件。
"""
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from string import Template
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from .api import GEMParameters, GEMResult


@dataclass(frozen=True)
class XMDSConfig:
    params: GEMParameters = GEMParameters(g=1, N=2, L=1, eta=4)
    duration: float = 2.0
    switch_time: float = 1.0
    pulse_time: float = 0.3
    pulse_width: float = 0.1
    amplitude: float = 1.0
    lattice: int = 128
    z_samples: int = 64
    samples_per_stage: int = 40
    steps_per_stage: int = 1000
    tolerance: float = 1e-8

    def __post_init__(self):
        real = (self.duration, self.switch_time, self.pulse_time, self.pulse_width,
                self.amplitude, self.tolerance)
        if not np.isfinite(real).all() or not 0 < self.switch_time < self.duration:
            raise ValueError("参数必须有限，翻转时刻须位于(0,duration)")
        if self.pulse_width <= 0 or not 0 < self.tolerance < 1:
            raise ValueError("pulse_width须为正，tolerance须位于(0,1)")
        for value in (self.lattice, self.z_samples, self.samples_per_stage, self.steps_per_stage):
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError("格点、采样和步数须为至少2的整数")
        if self.lattice % self.z_samples or self.steps_per_stage < self.samples_per_stage:
            raise ValueError("lattice须为z_samples的整数倍，steps须不小于samples")


def render_gem(config=XMDSConfig()):
    """返回XML字符串；数值由数据类验证，不接受任意C++片段。"""
    values = asdict(config.params)
    values.update(write_time=config.switch_time, read_time=config.duration-config.switch_time,
                  pulse_time=config.pulse_time, pulse_width=config.pulse_width,
                  amplitude=config.amplitude, lattice=config.lattice, z_samples=config.z_samples,
                  steps=config.steps_per_stage, samples=config.samples_per_stage, tolerance=config.tolerance)
    template = files("gem_solvers").joinpath("templates/gem.xmds").read_text(encoding="utf-8")
    return Template(template).substitute(values).encode("ascii", "xmlcharrefreplace").decode("ascii")


def _command(args, cwd, log_name, timeout):
    start = time.perf_counter()
    with (cwd / log_name).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise TimeoutError(f"命令超时，日志保留于{cwd / log_name}") from None
    if returncode:
        raise RuntimeError(f"命令失败（返回码{returncode}），请查看{cwd / log_name}")
    return time.perf_counter() - start


@dataclass(frozen=True)
class XMDSRun:
    directory: Path
    compile_seconds: float
    run_seconds: float

    def read(self):
        result = read_gem_ascii(self.directory / "gem.xsil")
        result.elapsed_seconds = self.compile_seconds + self.run_seconds
        return result


def _run_text(text, directory, *, timeout):
    executable = shutil.which("xmds2")
    if not executable:
        raise FileNotFoundError("未找到xmds2；请按XMDS2安装文档安装外部编译器，不是pip依赖")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout须为有限正数（每个编译/运行阶段的秒数）")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "gem.xmds"
    source.write_text(text, encoding="utf-8")
    metadata = dict(state="RUNNING", source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    commands=[["xmds2", "gem.xmds"], ["./gem"]], compiler=executable)
    path = directory / "run.json"
    try:
        # 固定可执行文件名，使输出位置可预测；只允许运行自包含的可信输入。
        tree = ET.fromstring(text)
        if tree.findtext("name", "").strip() != "gem":
            raise ValueError("run_file要求simulation/name为gem；可先修改自包含输入副本")
        compiled = _command([executable, "gem.xmds"], directory, "compile.log", timeout)
        metadata["compile_seconds"] = compiled
        elapsed = _command([str(directory / "gem")], directory, "simulation.log", timeout)
        metadata.update(state="COMPLETE", run_seconds=elapsed)
        return XMDSRun(directory, compiled, elapsed)
    except Exception as exc:
        metadata.update(state="FAILED", error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def run_gem(directory, config=XMDSConfig(), *, timeout=120):
    """每次创建新目录；禁止覆盖已有结果，失败日志不删除。"""
    return _run_text(render_gem(config), directory, timeout=timeout)


def run_file(source, directory, *, timeout=120):
    """运行可信、自包含的XMDS文件（name=gem）；解析器仅支持本项目输出格式。"""
    return _run_text(Path(source).read_text(encoding="utf-8"), directory, timeout=timeout)


def read_gem_ascii(path):
    """读取本项目生成的ASCII XSIL；不猜测/重建实际坐标。"""
    root = ET.parse(path).getroot()
    group = root.find(".//XSIL[@Name='moment_group_1']")
    if group is None:
        raise ValueError("未找到ASCII XSIL数据组")
    arrays = {a.attrib.get("Name"): a for a in group.findall("Array")}
    variables = "".join(arrays["variables"].find("Stream").itertext()).split()
    data = arrays["data"]
    dims = tuple(int(d.text) for d in data.findall("Dim"))
    stream = data.find("Stream")
    if stream is None or stream.find("Metalink") is None or stream.find("Metalink").attrib.get("Format") != "Text":
        raise ValueError("仅支持内嵌ASCII数据；二进制/HDF5请使用相应读取器")
    numbers = np.fromstring("".join(stream.itertext()), sep=" ")
    if numbers.size != int(np.prod(dims)):
        raise ValueError("XSIL数据长度与维度声明不一致")
    matrix = numbers.reshape(-1, len(variables))
    columns = {name: matrix[:, i] for i, name in enumerate(variables)}
    required = {"t", "z", "E_real", "E_imag", "alpha_real", "alpha_imag"}
    if not required <= columns.keys():
        raise ValueError("缺少本项目规定的坐标或复场分量")
    z, t = np.unique(columns["z"]), np.unique(columns["t"])
    if len(z)*len(t) != len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("XSIL必须是有限的完整矩形网格")
    zi, ti = np.searchsorted(z, columns["z"]), np.searchsorted(t, columns["t"])
    if len(np.unique(zi*len(t)+ti)) != len(matrix):
        raise ValueError("XSIL含重复坐标")
    e = np.empty((len(z), len(t)), complex)
    a = np.empty_like(e)
    e[zi, ti] = columns["E_real"]+1j*columns["E_imag"]
    a[zi, ti] = columns["alpha_real"]+1j*columns["alpha_imag"]
    return GEMResult(e, a, z, t, "xmds2", float("nan"))
