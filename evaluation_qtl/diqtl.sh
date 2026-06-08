time python benchmark_epoch_checkpoints.py all ../models/clipnet/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --gpu --mode ensemble
time python benchmark_epoch_checkpoints.py all ../models/mean_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --gpu --mode ensemble
time python benchmark_epoch_checkpoints.py all ../models/ref_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --gpu --mode ensemble
time python benchmark_epoch_checkpoints.py all ../models/clipnet/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --gpu --mode folds
time python benchmark_epoch_checkpoints.py all ../models/mean_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --gpu --mode folds
time python benchmark_epoch_checkpoints.py all ../models/ref_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --gpu --mode folds

time python benchmark_epoch_checkpoints.py all ../models/clipnet/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --gpu --mode ensemble
time python benchmark_epoch_checkpoints.py all ../models/mean_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --gpu --mode ensemble
time python benchmark_epoch_checkpoints.py all ../models/ref_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --gpu --mode ensemble
time python benchmark_epoch_checkpoints.py all ../models/clipnet/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --gpu --mode folds
time python benchmark_epoch_checkpoints.py all ../models/mean_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --gpu --mode folds
time python benchmark_epoch_checkpoints.py all ../models/ref_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --gpu --mode folds


time python benchmark_best_model.py all ../models/clipnet/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --allow_row_order --gpu --mode ensemble 
time python benchmark_best_model.py all ../models/mean_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --allow_row_order --gpu --mode ensemble
time python benchmark_best_model.py all ../models/ref_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --allow_row_order --gpu --mode ensemble
time python benchmark_best_model.py all ../models/clipnet/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --allow_row_order --gpu --mode folds
time python benchmark_best_model.py all ../models/mean_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --allow_row_order --gpu --mode folds
time python benchmark_best_model.py all ../models/ref_model/ --qtl diqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/diqtl/ --allow_row_order --gpu --mode folds
time python benchmark_best_model.py all ../models/clipnet/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --allow_row_order --gpu --mode ensemble
time python benchmark_best_model.py all ../models/mean_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --allow_row_order --gpu --mode ensemble
time python benchmark_best_model.py all ../models/ref_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --allow_row_order --gpu --mode ensemble
time python benchmark_best_model.py all ../models/clipnet/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --allow_row_order --gpu --mode folds
time python benchmark_best_model.py all ../models/mean_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --allow_row_order --gpu --mode folds
time python benchmark_best_model.py all ../models/ref_model/ --qtl tiqtl --predictions_root ../predictions/ --data_root ../data/ --qtl_data_dir ../data/tiqtl/ --allow_row_order --gpu --mode folds