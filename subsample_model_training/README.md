# Train models for the subsample datasets

## Copy files to scratch space and rearrange to match calculate_dataset_params.py

```bash
```

## Calculate dataset parameters

```bash
cd /home2/ayh8/clipnet/
n=5
for i in {0..3}; do
    python calculate_dataset_params.py \
        /home2/ayh8/data/clipnet_subsampling/${n}_subsample_run${i}/ \
        /home2/ayh8/clipnet_subsampling/models/${n}_subsample_run${i};
done
```

## Train models

```bash
conda activate clipnet
cd /home2/ayh8/clipnet/
n=5
i=1
for fold in {1..9}; do
    python fit_nn.py /home2/ayh8/clipnet_subsampling/models/${n}_subsample_run${i}/f${fold};
done
```
