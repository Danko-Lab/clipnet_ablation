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
cd /home2/ayh8/clipnet/
python calculate_dataset_params.py \
    /home2/ayh8/data/clipnet_subsampling/subsample_data_folds_n5_run1/ \
    /home2/ayh8/clipnet_subsampling/models/n5_run1/
```

## Calculate dataset parameters

```bash
cd /home2/ayh8/clipnet/
for n in 5 10 15 20 30; do
    python calculate_dataset_params.py \
        /home2/ayh8/data/clipnet_subsampling/ayh8/subsample_data_folds_n${n}_run1/ \
        /home2/ayh8/clipnet_subsampling/models/ayh8/n${n}_run1;
done

python calculate_dataset_params.py \
    /home2/ayh8/data/clipnet_subsampling/${n}_subsample_run00/ \
    /home2/ayh8/clipnet_subsampling/models/ayh8/n${n}_run00
```

## Train models

```bash
conda activate clipnet
cd /home2/ayh8/clipnet/
n=5
i=0
for fold in {1..9}; do
    python fit_nn.py \
        /home2/ayh8/clipnet_subsampling/models/${n}_subsample_run${i}/f${fold} \
        --use_specific_gpu 0;
done
```
