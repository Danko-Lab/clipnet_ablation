# Train models for the subsample datasets

## Copy files to scratch space and rearrange to match calculate_dataset_params.py

```bash
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
i=00
for fold in {1..9}; do
    python fit_nn.py \
        /home2/ayh8/clipnet_subsampling/models/ayh8/n${n}_run${i}/f${fold} \
        --use_specific_gpu 0;
done
```
