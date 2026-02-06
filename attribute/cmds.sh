clipnet attribute -c -v \
    -f ../data/all_tss_windows_reference_seq.fna.gz \
    -o ../attr/clipnetr_all_tss_windows_quantity.npz \
    -s ../attr/clipnetr_all_tss_windows_ohe.npz \
    -m ../models/reference_models/ \
    -a quantity
export NUMBA_NUM_THREADS=32
time modisco motifs \
    -s clipnetr_all_tss_windows_ohe.npz \
    -a clipnetr_all_tss_windows_quantity.npz \
    -o clipnetr_all_tss_windows_quantity.modisco.h5 \
    -n 2000000 -l 50 -w 500 -v
time modisco report \
    -i clipnetr_all_tss_windows_quantity.modisco.h5 \
    -o clipnetr_all_tss_windows_quantity.modisco \
    -m /home2/ayh8/data/JASPAR/JASPAR2024_CORE_non-redundant_pfms_meme.txt

clipnet attribute -v \
    -f ../data/all_tss_windows_reference_seq.fna.gz \
    -o ../attr/clipnetr_all_tss_windows_profile.npz \
    -m ../models/reference_models/ \
    -a profile
export NUMBA_NUM_THREADS=16
time modisco motifs \
    -s clipnetr_all_tss_windows_ohe.npz \
    -a clipnetr_all_tss_windows_profile.npz \
    -o clipnetr_all_tss_windows_profile.modisco.h5 \
    -n 2000000 -l 50 -w 500 -v
time modisco report \
    -i clipnetr_all_tss_windows_profile.modisco.h5 \
    -o clipnetr_all_tss_windows_profile.modisco \
    -m /home2/ayh8/data/JASPAR/JASPAR2024_CORE_non-redundant_pfms_meme.txt