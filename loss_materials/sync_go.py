"""
Usage:
    $ python3 /home/fengwen/compare_bn/sync_go.py # 单卡
    $ python3  -m  oneflow.distributed.launch --nproc_per_node 2 /home/fengwen/compare_bn/sync_go.py # 多卡
    $ python3  -m  torch.distributed.run --nproc_per_node   2 /home/fengwen/compare_bn/sync_go.py
"""

import random
from telnetlib import WONT
import oneflow
import oneflow.backends.cudnn as cudnn
import time
import os
import numpy as np
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1)) 
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
print(LOCAL_RANK,RANK,WORLD_SIZE)


def time_sync():
    # Oneoneflow-accurate time
    if oneflow.cuda.is_available():
        print("start",end=' ')
        oneflow.cuda.synchronize()
    print("end")
    return time.time()


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pyoneflow.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible


    random.seed(seed)
    np.random.seed(seed)
    oneflow.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) 
    oneflow.cuda.manual_seed(seed)
    oneflow.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe



oneflow.cuda.set_device(LOCAL_RANK)
device = oneflow.device('cuda', LOCAL_RANK)
init_seeds(1, True)
for i in range(2):
    t0 = time_sync()
    t1 = time_sync()
    print('time : ',t1-t0,"--is ok")