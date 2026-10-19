"""外部XMDS2示例：目录必须尚不存在，结果包含真实输出坐标。"""
import argparse
from causalwave import GEMParameters
from causalwave.xmds import XMDSConfig,run_gem

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory',required=True)
args=parser.parse_args()
config=XMDSConfig(params=GEMParameters(g=1,N=2,L=1,eta=4,gamma=-.001))
run=run_gem(args.directory,config)
fields=run.read()
fields.save(run.directory/'fields.npz')
print(fields.E.shape,run.compile_seconds,run.run_seconds)
