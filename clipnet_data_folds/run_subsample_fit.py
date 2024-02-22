import os

clipnet_install = "/home2/ayh8/clipnet"
model_root = "/home2/ayh8/subsample_models/n5_run0"
gpu = 0
for i in range(9):
    cmd = f"python {clipnet_install}/fit_nn.py {model_root}/f{i + 1} --use_specific_gpu {gpu}"
    os.system(f"echo {cmd}")
    os.system(cmd)
