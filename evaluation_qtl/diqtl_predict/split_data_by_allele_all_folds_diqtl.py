import itertools
import multiprocessing as mp
import os

import h5py
import joblib
import numpy as np
import pandas as pd
import tqdm

PREDICTDIR = "/workdir/ayh8/clipnet_subsampling/diqtl/"

# Import snps per individual
alleles_per_ind = pd.read_csv(
    os.path.join(PREDICTDIR, "diQTL_snps_per_individual.csv.gz"), index_col=0
)
prefixes = list(alleles_per_ind.index)
snps = list(alleles_per_ind.columns)

# Create outdir
outdir = os.path.join(PREDICTDIR, "fold_predict/split_by_allele")
os.makedirs(outdir, exist_ok=True)

# Define iters to run
n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

iters = list(itertools.product(n_individuals, runs))


def split_by_allele(it):
    run = f"n{it[0]}_run{it[1]}"
    # create dictionaries for output
    pred_tracks_per_snp_by_allele = {snps[i]: [[], [], []] for i in range(len(snps))}
    pred_quantity_per_snp_by_allele = {snps[i]: [[], [], []] for i in range(len(snps))}
    pred_track_out = {snps[i]: [[], [], []] for i in range(len(snps))}
    pred_quantity_out = {snps[i]: [[], [], []] for i in range(len(snps))}
    # iterate over folds
    folds = range(1, 10)
    for fold in folds:
        outpath = os.path.join(
            outdir, f"{run}_fold_{fold}_pred_per_snp_by_allele.joblib.gz"
        )
        if os.path.exists(outpath):
            print(f"{outpath} exists, skipping.")
            continue
        else:
            for pref in prefixes:
                pred_fp = os.path.join(
                    PREDICTDIR, f"fold_predict/{run}/fold_{fold}/{pref}.h5"
                )
                pred = h5py.File(pred_fp)
                pred_scaled = (
                    pd.DataFrame(pred["track"])
                    .divide(pd.DataFrame(pred["track"]).sum(axis=1) + 1e-3, axis=0)
                    .multiply(np.array(pred["quantity"]), axis=0)
                ).to_numpy()
                for i in range(len(snps)):
                    allele = alleles_per_ind[snps[i]][pref]
                    if allele == 0:
                        pred_tracks_per_snp_by_allele[snps[i]][0].append(pred_scaled[i])
                        pred_quantity_per_snp_by_allele[snps[i]][0].append(
                            pred["quantity"][i][0]
                        )
                    elif allele == 1:
                        pred_tracks_per_snp_by_allele[snps[i]][1].append(pred_scaled[i])
                        pred_quantity_per_snp_by_allele[snps[i]][1].append(
                            pred["quantity"][i][0]
                        )
                    elif allele == 0.5:
                        pred_tracks_per_snp_by_allele[snps[i]][2].append(pred_scaled[i])
                        pred_quantity_per_snp_by_allele[snps[i]][2].append(
                            pred["quantity"][i][0]
                        )
            for k in pred_tracks_per_snp_by_allele.keys():
                for i in range(len(pred_tracks_per_snp_by_allele[k])):
                    pred_track_out[k][i] = np.array(pred_tracks_per_snp_by_allele[k][i])
                    pred_quantity_out[k][i] = np.array(
                        pred_quantity_per_snp_by_allele[k][i]
                    )
            joblib.dump([pred_track_out, pred_quantity_out], outpath)


# Run in parallel
with mp.Pool(8) as pool:
    r = list(tqdm.tqdm(pool.imap(split_by_allele, iters), total=len(iters)))
