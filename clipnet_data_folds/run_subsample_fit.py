import os
import multiprocessing as mp

def run_fitting_cmd(i):
    clipnet_install = "/home2/ayh8/clipnet"
    model_root = "/home2/ayh8/subsample_models/n30_run0"
    gpu = i % 2
    cmd = f"python {clipnet_install}/fit_nn.py {model_root}/f{i + 1} --use_specific_gpu {gpu}"
    os.system(f"echo {cmd}")
    #os.system(cmd)

with mp.Pool(2) as p:
    p.map(run_fitting_cmd, range(9))
    
