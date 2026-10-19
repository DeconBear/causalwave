"""缓存只节省重复计算，不改变数值结果或输入合同。"""
import numpy as np
import pytest
from causalwave._if2_reference import solve_gem
from causalwave._if2 import solve_gem_cached


@pytest.mark.parametrize('z_factor,t_factor',[(1,1),(2,1),(4,2)])
@pytest.mark.parametrize('control',['step','varying'])
def test_cache_matches_original_with_complex_irregular_inputs(z_factor,t_factor,control):
    rng=np.random.default_rng(19);drive=np.linspace(0,1,41)
    args=dict(params=dict(g=2.,N=20.,L=2.,eta=50.,gama=.001),z=np.r_[0.,np.sort(rng.uniform(0,2,18))],
              t=np.r_[0.,np.linspace(.025,1,40).astype(np.float32).astype(float)],Ein=None,sw=None,
              drive_t=drive,drive_Ein=np.exp(-((drive-.2)/.1)**2)*np.exp(10j*drive),
              drive_sw=np.where(drive<.5,1.,-1.) if control=='step' else np.cos(drive*7),
              z_factor=z_factor,t_factor=t_factor,method='if2')
    a=solve_gem(**args);b=solve_gem_cached(**args)
    for k in ('E','alpha'):np.testing.assert_array_equal(a[k],b[k])
    assert b['peak_cache_bytes']<=64*1024**2


def test_no_cross_request_cache():
    x=np.linspace(0,1,12)
    a=dict(params=dict(g=1.,N=0.,L=1.,eta=2.,gama=0.),z=x,t=x,Ein=np.ones(12),sw=np.ones(12),method='if2')
    first=solve_gem_cached(**a);a['params']['N']=3.;b=solve_gem_cached(**a);ref=solve_gem(**a)
    assert not np.array_equal(first['E'],b['E']);np.testing.assert_array_equal(b['E'],ref['E'])


@pytest.mark.parametrize('factor',[0,-1,True,1.5])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):solve_gem_cached({},[0.,1.],[0.,1.],None,None,z_factor=factor)
