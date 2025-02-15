# These are scripts used to generate data for the two ablation studies in the personalized genomes paper

Most of these pipelines are from the original CLIPNET project. The intended run order is `download_raw`, `get_peaks`, `get_window`, `normalize_procap`, `procap`, `sequence`. This will generate the training data used for the original CLIPNET model. The scripts in `reference_sequence`, `subsample_procap`, and `subsample_sequence` will generate the data used for the two ablation studies presented in this paper (masking out variants & subsampling the number of training individuals used).

These scripts have not been cleaned up in a long time, so please let me know if you need clarification on file paths, etc. I intend to hook these up to the processed data we deposited on Zenodo for the main CLIPNET paper (https://zenodo.org/records/13771189), but have not had the time to do so.
