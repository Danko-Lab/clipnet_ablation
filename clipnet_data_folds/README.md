# These are the instructions for subsampling the data for the clipnet project

Most of these pipelines are from the original CLIPNET project. The intended run order is `download_raw`, `get_peaks`, `get_window`, `normalize_procap`, `procap`, `sequence`. This will generate the training data used for CLIPNET. The scripts in `reference_sequence`, `subsample_procap`, and `subsample_wequence` will generate the data used for the two ablation studies presented in this paper (masking out variants & subsampling the number of training individuals used).

These scripts have not been cleaned up in a long time, so please let me know if you need clarification on file paths, etc.
