'''
When you use mutil-process to dataloader which is uesd to np.random, you will reckon that random value is equal.
So, you need to deal with problem using worker_init_fn_seed.
'''
import numpy as np
def worker_init_fn_seed(worker_id):
    seed = 1234
    seed += (worker_id*1993)
    np.random.seed(seed)
