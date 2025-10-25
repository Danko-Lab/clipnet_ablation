import itertools
import multiprocessing as mp
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tqdm

sys.path.append("../")
import scores

PREDICTDIR = "/workdir/ayh8/clipnet_subsampling/diqtl/"

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

iters = list(itertools.product(n_individuals, runs))

folds = range(10)
data_splits = pd.read_csv("data_fold_assignments.csv")
holdouts = {i: list(data_splits[data_splits.fold == int(i)].chrom) for i in folds}

# Load and clean experimental data
expt_fp = "/fs/cbsubscb17/storage/projects/CLIPNET/predictions/ensemble/diqtl/ensemble_predictions/expt_tracks_per_snp_by_allele.joblib.gz"
expt = joblib.load(expt_fp)
ref_n = {k: len(expt[k][0]) for k in expt.keys()}
alt_n = {k: len(expt[k][1]) for k in expt.keys()}
expt_clean = {k: expt[k] for k in expt.keys() if ref_n[k] >= 3 and alt_n[k] >= 3}
snps = list(expt_clean.keys())
expt_ref_mean_tracks = pd.DataFrame(
    {snp: np.mean(expt_clean[snp][0], axis=0) for snp in snps}
).transpose()
expt_alt_mean_tracks = pd.DataFrame(
    {snp: np.mean(expt_clean[snp][1], axis=0) for snp in snps}
).transpose()
qtls = pd.DataFrame({"snps": expt.keys()})
qtl_coord = pd.merge(
    qtls, pd.read_csv("Table.10a.diQTL.2kb.csv.gz"), on="snps", how="left"
)
qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(".", expand=True)

outdir = os.path.join(PREDICTDIR, "fold_predict/split_by_allele/")
os.makedirs(outdir, exist_ok=True)


def calculate_scores(it):
    for fold in folds[1:]:
        l2_holdout_predictions = []
        # load data
        prefix = f"n{it[0]}_run{it[1]}_fold_{fold}"
        pred_fp = f"{prefix}_pred_per_snp_by_allele.joblib.gz"
        pred = joblib.load(os.path.join(outdir, pred_fp))
        # clean data
        pred_clean = {k: pred[0][k] for k in expt_clean.keys()}
        pred_ref_mean_tracks = pd.DataFrame(
            {snp: np.mean(pred_clean[snp][0], axis=0) for snp in snps}
        ).transpose()
        pred_alt_mean_tracks = pd.DataFrame(
            {snp: np.mean(pred_clean[snp][1], axis=0) for snp in snps}
        ).transpose()
        # calculate scores
        l2 = pd.DataFrame(
            {
                "expt": scores.l2_score(expt_ref_mean_tracks, expt_alt_mean_tracks),
                "pred": scores.l2_score(pred_ref_mean_tracks, pred_alt_mean_tracks),
            }
        )
        # Filter for holdout SNPs
        test = qtl_coord[qtl_coord.chrom.isin(holdouts[fold])]
        l2_holdout_predictions.append(
            l2.loc[[snp for snp in set(test["snps"]) if snp in l2.index]]
        )
        l2_qtls = pd.concat(l2_holdout_predictions)
        l2_qtls.to_csv(os.path.join(outdir, f"{prefix}_l2_scores.csv.gz"))


# Run in parallel
with mp.Pool(4) as pool:
    r = list(tqdm.tqdm(pool.imap(calculate_scores, iters), total=len(iters)))
