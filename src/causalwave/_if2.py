"""IF2等价实现：仅缓存同一求解内重复的精确(h,sw)指数系数。"""
from collections import OrderedDict
from time import perf_counter
import numpy as np
from scipy.linalg import solve_banded
from ._if2_reference import _axis,_refine,_integral,_exponential_weights


def solve_gem_cached(params,z,t,Ein,sw,*,z_factor=1,t_factor=1,method='if2',
                     drive_t=None,drive_Ein=None,drive_sw=None):
    """与原IF2相同离散式和运算次序；每次调用独立LRU，最多64MiB。

    缓存键不舍入步长或控制值，介质与空间网格只在本次调用内固定。
    不跨请求复用，不更改驱动、网格、精度或求解器阶数。
    """
    start=perf_counter();z,t=_axis(z,'z'),_axis(t,'t')
    for factor in (z_factor,t_factor):
        if not isinstance(factor,(int,np.integer)) or isinstance(factor,bool) or factor<1:raise ValueError('加密倍数必须为正整数')
    if method!='if2':raise ValueError('缓存实现只支持if2')
    g,density,length,eta,gamma=(float(params[k]) for k in ('g','N','L','eta','gama'))
    if not np.isfinite([g,density,length,eta,gamma]).all() or length<=0:raise ValueError('介质参数必须有限且L>0')
    if z[-1]>length+1e-12*max(1.,length):raise ValueError('输出z超出L')
    supplied=(drive_t is not None,drive_Ein is not None,drive_sw is not None)
    if any(supplied) and not all(supplied):raise ValueError('必须完整提供原始驱动')
    if not any(supplied):drive_t,drive_Ein,drive_sw=t,Ein,sw
    drive_t=_axis(drive_t,'drive_t');drive_Ein=np.asarray(drive_Ein,dtype=np.complex128);raw_sw=np.asarray(drive_sw)
    if np.iscomplexobj(raw_sw) and np.any(raw_sw.imag!=0):raise ValueError('sw必须实值')
    drive_sw=np.asarray(raw_sw.real,dtype=float)
    if drive_Ein.shape!=drive_t.shape or drive_sw.shape!=drive_t.shape or not np.isfinite(drive_Ein).all() or not np.isfinite(drive_sw).all():raise ValueError('驱动形状或数值不符')
    fine_z=_refine(np.unique(np.r_[0.,z]),z_factor);zi=np.searchsorted(fine_z,z);dz=np.diff(fine_z);hz=.5*dz
    fine_t=_refine(np.unique(np.r_[0.,t,drive_t[drive_t<=t[-1]]]),t_factor);indices=np.searchsorted(fine_t,t)
    incident=np.interp(fine_t,drive_t,drive_Ein)
    controls=drive_sw[np.clip(np.searchsorted(drive_t,fine_t[:-1],side='right')-1,0,len(drive_t)-1)]
    alpha=np.zeros(len(fine_z),dtype=complex);integral=_integral(alpha,dz)
    field_e=np.empty((len(z),len(t)),complex);field_a=np.empty_like(field_e);j=0
    if indices[0]==0:field_a[:,0]=alpha[zi];field_e[:,0]=incident[0]+1j*density*integral[zi];j=1
    cache=OrderedDict();cache_bytes=0;peak_cache=0;hits=0;misses=0;limit=64*1024**2
    for k,h in enumerate(np.diff(fine_t)):
        key=(float(h),float(controls[k]))
        if key in cache:
            weights=cache[key];cache.move_to_end(key);hits+=1
        else:
            diagonal=gamma+1j*eta*(fine_z-length/2)*controls[k]
            weights=_exponential_weights(diagonal,h);misses+=1;size=sum(w.nbytes for w in weights)
            while cache and cache_bytes+size>limit:
                _,old=cache.popitem(last=False);cache_bytes-=sum(w.nbytes for w in old)
            if size<=limit:cache[key]=weights;cache_bytes+=size;peak_cache=max(peak_cache,cache_bytes)
        growth,w0,w1=weights;coupling=g*density;forcing=1j*g
        rhs=growth*alpha-coupling*w0*integral+forcing*(w0*incident[k]+w1*incident[k+1])
        q=coupling*w1
        bands=np.zeros((2,len(alpha)),dtype=complex);bands[0]=1.
        bands[0,1:]+=hz*q[1:];bands[1,:-1]=-(1.-hz*q[:-1])
        p_rhs=np.r_[0j,hz*(rhs[:-1]+rhs[1:])]
        p=solve_banded((1,0),bands,p_rhs,check_finite=False);alpha=rhs-q*p
        if not np.isfinite(alpha).all():raise FloatingPointError('IF2产生非有限状态')
        integral=_integral(alpha,dz)
        if j<len(t) and k+1==indices[j]:field_a[:,j]=alpha[zi];field_e[:,j]=incident[k+1]+1j*density*integral[zi];j+=1
    return dict(E=field_e,alpha=field_a,t=t.copy(),z=z.copy(),elapsed_seconds=perf_counter()-start,
                cache_hits=hits,cache_misses=misses,peak_cache_bytes=peak_cache)
