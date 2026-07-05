time python evaluate_best_model.py ../models/clipnet/ ../data/merged_sequence_0.fna.gz ../data/merged_procap_0.csv.gz ../predictions/clipnet_best.csv --gpu
time python evaluate_best_model.py ../models/ref_model/ ../data/merged_sequence_0.fna.gz ../data/merged_procap_0.csv.gz ../predictions/ref_best.csv --gpu
time python evaluate_best_model.py ../models/mean_model/ ../data/merged_sequence_0.fna.gz ../data/merged_procap_0.csv.gz ../predictions/mean_best.csv --gpu
time python evaluate_training_epochs.py ../models/clipnet/ ../data/merged_sequence_0.fna.gz ../data/merged_procap_0.csv.gz ../predictions/clipnet_epochs.csv --gpu
time python evaluate_training_epochs.py ../models/ref_model/ ../data/merged_sequence_0.fna.gz ../data/merged_procap_0.csv.gz ../predictions/ref_epochs.csv --gpu
time python evaluate_training_epochs.py ../models/mean_model/ ../data/merged_sequence_0.fna.gz ../data/merged_procap_0.csv.gz ../predictions/mean_epochs.csv --gpu