import glob
import os

model_dirs = glob.glob("models/*")
data_path = "data/merged_sequence_0.fna.gz"
for model_dir in model_dirs:
    name = os.path.split(model_dir)[-1]
    cmd = f"clipnet predict -f {data_path} -o predictions/{name}_merged_sequence_0_predictions.npz -m {model_dir} -v"
    print(cmd)
#        os.system(cmd)
