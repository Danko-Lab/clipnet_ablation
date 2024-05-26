# Train models for the subsample datasets

## Copy files to scratch space and rearrange to match calculate_dataset_params.py

Since the generator used to load data needs to know the # of batches per/epoch, we need to precalculate the total number of examples in each training dataset. This is done by the `calculate_dataset_params.py` script (see the main CLIPNET repo). This script takes as input the data directory and an output directory where it'll write json files for all of the model parameters. The data directory should just contain all the processed PRO-cap and sequence data

For example, if you run `ls -l /fs/cbsubscb17/storage/projects/CLIPNET_subsampling/subsample_data_folds_n5_run1/`, it should look like this:

```bash
total 300542
-rw-r--r-- 1 ayh8 ayh8 10838004 May 14 22:24 concat_procap_0.npz
-rw-r--r-- 1 ayh8 ayh8 10701381 May 14 22:24 concat_procap_1.npz
-rw-r--r-- 1 ayh8 ayh8 11054786 May 14 22:24 concat_procap_2.npz
-rw-r--r-- 1 ayh8 ayh8  9371678 May 14 22:24 concat_procap_3.npz
-rw-r--r-- 1 ayh8 ayh8 11382844 May 14 22:24 concat_procap_4.npz
-rw-r--r-- 1 ayh8 ayh8 10483156 May 14 22:24 concat_procap_5.npz
-rw-r--r-- 1 ayh8 ayh8 10371137 May 14 22:24 concat_procap_6.npz
-rw-r--r-- 1 ayh8 ayh8 10582969 May 14 22:24 concat_procap_7.npz
-rw-r--r-- 1 ayh8 ayh8 10872265 May 14 22:24 concat_procap_8.npz
-rw-r--r-- 1 ayh8 ayh8 10424333 May 14 22:24 concat_procap_9.npz
-rw-r--r-- 1 ayh8 ayh8 21357936 May 14 22:24 concat_sequence_0.npz
-rw-r--r-- 1 ayh8 ayh8 20989346 May 14 22:24 concat_sequence_1.npz
-rw-r--r-- 1 ayh8 ayh8 22004307 May 14 22:24 concat_sequence_2.npz
-rw-r--r-- 1 ayh8 ayh8 18879618 May 14 22:24 concat_sequence_3.npz
-rw-r--r-- 1 ayh8 ayh8 22483513 May 14 22:24 concat_sequence_4.npz
-rw-r--r-- 1 ayh8 ayh8 20479109 May 14 22:24 concat_sequence_5.npz
-rw-r--r-- 1 ayh8 ayh8 20293532 May 14 22:24 concat_sequence_6.npz
-rw-r--r-- 1 ayh8 ayh8 21249266 May 14 22:24 concat_sequence_7.npz
-rw-r--r-- 1 ayh8 ayh8 22201833 May 14 22:24 concat_sequence_8.npz
-rw-r--r-- 1 ayh8 ayh8 21188008 May 14 22:24 concat_sequence_9.npz
```

These data directories should be copied into a scratch space on either `/home2/` or the `/workdir/` on `cbsugpu01.tc.cornell.edu`. The `calculate_dataset_params.py` script can then be run to generate the parameter json files. e.g., for the above data directory, the command would be:

```bash
conda activate clipnet

cd /home2/ayh8/clipnet/
python calculate_dataset_params.py \
    /home2/ayh8/clipnet_subsampling/data/5_subsample_run1/ \
    /home2/ayh8/clipnet_subsampling/models/n5_run1/
```

Note that this script will automatically generate the output directory, which will then be structured as:

```bash
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f1
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f2
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f3
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f4
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f5
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f6
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f7
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f8
drwxr-sr-x 2 ayh8 danko      133 May 25 11:37 f9
```

Each of the `f*` directories will contain a json file with the dataset parameters. The json files will contain important parameters like paths to each of the data files, # examples per file and per epoch, etc.

For prepping a bunch of subsample runs at once, it might be useful to run this script in a for loop, e.g.:

```bash
cd /home2/ayh8/clipnet/
for n in 5 10 15 20 30; do
    python calculate_dataset_params.py \
        /home2/ayh8/clipnet_subsampling/data/subsample_data_folds_n${n}_run0/ \
        /home2/ayh8/clipnet_subsampling/models/n${n}_run0;
done
```

## Training models

The `fit_nn.py` script can be used to train the models. The `fit_nn.py` script will automatically save the model weights and training progress to the output directory, so you can check on the training progress at any time. It takes as input the `f*` directories created by the `calculate_dataset_params.py` script. For example:

```bash
cd /home2/ayh8/clipnet/
for fold in {1..9}; do
    python fit_nn.py /home2/ayh8/clipnet_subsampling/models/n5_run0/f${fold} --gpu 0;
done
```

This will run a for loop to train the models on each of the 9 folds for the n=5 subsample run 0. Please note that you will need to invoke `--gpu n` to specify which GPU to use. Needless to say, this will train extremely slowly on a CPU, so you should check which GPUs are available using `nvidia-smi`, then select the appropriate GPU to train on. Obviously, this will take longer for larger n values, so you might want to run this in a tmux session or screen session so that it doesn't fail when you DC.

There are two GPUs available on our hosted GPU server (cbsugpu01.tc.cornell.edu), so you can run two of these training loops at once (assuming no one else is running stuff). There are two additional GPUs available on the GPU cluster, which can be requested through the SLURM scheduler (see documentation [here](https://biohpc.cornell.edu/lab/cbsubscb_SLURM.htm)).
