"""XMDS隔离运行、错误记录和真实工具链集成。"""
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import pytest
from causalwave import GEMParameters
from causalwave import xmds
from causalwave.xmds import XMDSConfig, render_gem, run_gem, run_file


def test_config_and_ascii_render():
    xml=render_gem();xml.encode('ascii')
    assert '<moments>E_real E_imag alpha_real alpha_imag</moments>' in xml
    with pytest.raises(ValueError):XMDSConfig(switch_time=3)
    with pytest.raises(ValueError):XMDSConfig(lattice=65,z_samples=32)


def test_missing_compiler(monkeypatch,tmp_path):
    monkeypatch.setattr(shutil,'which',lambda _:None)
    with pytest.raises(FileNotFoundError,match='xmds2'):run_gem(tmp_path/'run')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_command_interruption_reaps_only_its_child(monkeypatch, tmp_path, exception):
    """真实子进程必须退出并被回收，其他任务不受中断清理影响。"""
    real_popen = subprocess.Popen
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    other = real_popen(sleeper, start_new_session=(os.name == "posix"))
    children = []

    def interrupted_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        wait = process.wait
        children.append((process, wait))
        first_wait = True

        def interrupted_wait(*args, **kwargs):
            nonlocal first_wait
            if first_wait:
                first_wait = False
                raise exception("受控中断")
            return wait(*args, **kwargs)

        monkeypatch.setattr(process, "wait", interrupted_wait)
        return process

    monkeypatch.setattr(xmds.subprocess, "Popen", interrupted_popen)
    try:
        with pytest.raises(exception, match="受控中断"):
            xmds._command(sleeper, tmp_path, "interrupted.log", 1)
        assert children[0][0].returncode is not None
        assert other.poll() is None
        assert (tmp_path / "interrupted.log").exists()
    finally:
        for process, wait in children + [(other, other.wait)]:
            if process.poll() is None:
                process.kill()
            wait(timeout=5)


def test_command_timeout_preserves_log(tmp_path):
    with pytest.raises(TimeoutError, match="命令超时"):
        xmds._command([sys.executable, "-c", "import time; time.sleep(30)"],
                      tmp_path, "timeout.log", .05)
    assert (tmp_path / "timeout.log").exists()


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("stage", ["compile.log", "simulation.log"])
def test_interrupted_run_records_terminal_state(monkeypatch, tmp_path, exception, stage):
    """编译和仿真中断均保留来源、阶段日志及可区分的结束状态。"""
    monkeypatch.setattr(shutil, "which", lambda _: sys.executable)

    def command(args, cwd, log_name, timeout):
        (cwd / log_name).write_text("已进入该阶段\n", encoding="utf-8")
        if log_name == stage:
            raise exception("受控中断")
        return .01

    monkeypatch.setattr(xmds, "_command", command)
    directory = tmp_path / "run"
    with pytest.raises(exception, match="受控中断"):
        run_gem(directory)
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == ("FAILED" if exception is RuntimeError else "INTERRUPTED")
    assert metadata["error"] == f"{exception.__name__}: 受控中断"
    assert (directory / "gem.xmds").exists()
    assert (directory / stage).read_text(encoding="utf-8") == "已进入该阶段\n"


@pytest.mark.xmds
@pytest.mark.skipif(not shutil.which('xmds2'), reason='外部XMDS2未安装')
def test_real_xmds_complex_fields_and_no_overwrite(tmp_path):
    config=XMDSConfig(params=GEMParameters(g=0,N=2,L=1,eta=4),lattice=64,z_samples=32,samples_per_stage=8,steps_per_stage=100)
    run=run_gem(tmp_path/'run',config);out=run.read()
    truth=np.exp(-((out.t-config.pulse_time)/config.pulse_width)**2)
    np.testing.assert_allclose(out.E,np.broadcast_to(truth,out.E.shape),atol=1e-8)
    np.testing.assert_allclose(out.alpha,0,atol=1e-13)
    assert out.E.shape==(32,16)
    assert out.z[-1]<config.params.L
    assert json.loads((run.directory/'run.json').read_text())['state']=='COMPLETE'
    with pytest.raises(FileExistsError):run_gem(run.directory,config)


@pytest.mark.xmds
@pytest.mark.skipif(not shutil.which('xmds2'), reason='外部XMDS2未安装')
def test_failed_input_keeps_logs(tmp_path):
    source=tmp_path/'bad.xmds';source.write_text('<simulation xmds-version="2"><name>gem</name></simulation>')
    with pytest.raises(RuntimeError):run_file(source,tmp_path/'failed')
    assert (tmp_path/'failed/compile.log').exists()
    assert json.loads((tmp_path/'failed/run.json').read_text())['state']=='FAILED'


@pytest.mark.xmds
@pytest.mark.skipif(not shutil.which('xmds2'), reason='外部XMDS2未安装')
def test_coupled_xmds_against_if2_at_actual_coordinates(tmp_path):
    from causalwave import Drive, solve_gem
    config = XMDSConfig(params=GEMParameters(g=1,N=2,L=1,eta=4,gamma=-.01))
    reference = run_gem(tmp_path/'coupled', config).read()
    nodes = np.linspace(0, config.duration, 4001)
    drive = Drive(nodes, np.exp(-((nodes-config.pulse_time)/config.pulse_width)**2),
                  np.where(nodes<config.switch_time,1.,-1.))
    result = solve_gem(config.params, reference.z, reference.t, drive,
                       method='if2', z_factor=4, t_factor=1)
    for name in ('E','alpha'):
        expected, actual = getattr(reference,name), getattr(result,name)
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.005
