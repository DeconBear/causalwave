"""统一接口的原生事件、复线性与两后端一致性。"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from gem_solvers import GEMParameters, Drive, solve_gem


@pytest.mark.parametrize('method', ['if2','marcher'])
def test_events_complex_drive_and_zero_density(method):
    p=GEMParameters(g=1,N=0,L=1,eta=4,gamma=-.1)
    dt=np.array([0.,.3,.55,1.]);u=np.array([1+1j,.2j,.4-.2j,1.]);sw=np.array([1.,-1.,.5,.5])
    drive=Drive(dt,u,sw);t=np.array([0.,.1,.5,.8,1.]);z=np.array([0.,.2,1.])
    out=solve_gem(p,z,t,drive,method=method,z_factor=1,t_factor=32)
    nodes=np.unique(np.r_[dt,t]);truth=np.zeros((len(z),len(nodes)),complex)
    for j,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
        d=p.gamma+1j*p.eta*(z-.5)*sw[np.searchsorted(dt,a,side='right')-1]
        truth[:,j+1]=solve_ivp(lambda tau,y:d*y+1j*np.interp(tau,dt,u), (a,b),truth[:,j],rtol=1e-12,atol=1e-13).y[:,-1]
    np.testing.assert_allclose(out.alpha,truth[:,np.searchsorted(nodes,t)],atol=3e-5)
    np.testing.assert_allclose(out.E,np.broadcast_to(np.interp(t,dt,u),out.E.shape),atol=1e-12)
    np.testing.assert_array_equal(out.t,t);np.testing.assert_array_equal(out.z,z)


def test_two_backends_coupled_agree_under_refinement():
    p=GEMParameters(g=1,N=2,L=1,eta=3,gamma=-.01);dt=np.linspace(0,1,41)
    drive=Drive(dt,np.exp(-((dt-.2)/.1)**2)*(1+.3j),np.where(dt<.5,1.,-1.))
    z=np.linspace(0,1,21);t=np.linspace(0,1,31)
    out=[solve_gem(p,z,t,drive,method=m,z_factor=4,t_factor=8) for m in ('if2','marcher')]
    for name in ('E','alpha'):
        a,b=[getattr(r,name) for r in out]
        assert np.linalg.norm(a-b)/np.linalg.norm(a)<2e-4


def test_invalid_inputs_and_save_no_overwrite(tmp_path):
    with pytest.raises(ValueError):Drive([.1,1],[1,1],[1,1])
    with pytest.raises(ValueError):Drive([0,1],[1,1],[1j,1])
    d=Drive([0,1],[1,1],[1,-1]);p=GEMParameters(g=0,N=0,L=1,eta=0)
    with pytest.raises(ValueError):solve_gem(p,[0,1],[0,2],d)
    with pytest.raises(ValueError):solve_gem(p,[0,1],[0,1],d,z_factor=True)
    r=solve_gem(p,[0,1],[0,1],d);path=tmp_path/'out.npz';r.save(path)
    with pytest.raises(FileExistsError):r.save(path)
    with np.load(path) as f:np.testing.assert_array_equal(f['E'],r.E)
