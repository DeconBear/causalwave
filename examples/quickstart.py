"""两个后端同输入/同输出的轻量演示；可在任意工作目录运行。"""
import numpy as np
from causalwave import GEMParameters, Drive, solve_gem

p=GEMParameters(g=1,N=5,L=1,eta=12,gamma=-.001)
dt=np.linspace(0,2,201)
drive=Drive(dt,np.exp(-((dt-.35)/.12)**2),np.where(dt<1,1.,-1.))
z=np.linspace(0,1,61);t=np.linspace(0,2,121)
for method in ('if2','marcher'):
    result=solve_gem(p,z,t,drive,method=method,z_factor=2,t_factor=2)
    print(method,result.E.shape,f'{result.elapsed_seconds:.3f}s',np.max(abs(result.alpha)))
