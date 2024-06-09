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

PREDICTDIR = "/workdir/ayh8/clipnet_subsampling/tiqtl/"

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

iters = list(itertools.product(n_individuals, runs))

folds = range(10)
data_splits = pd.read_csv("data_fold_assignments.csv")
holdouts = {i: list(data_splits[data_splits.fold == int(i)].chrom) for i in folds}

# Load and clean experimental data
expt_fp = "/fs/cbsubscb17/storage/projects/CLIPNET/predictions/ensemble/tiqtl/ensemble_predictions/expt_tracks_per_snp_by_allele.joblib.gz"
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
    qtls, pd.read_csv("Table.10a.tiqtl.2kb.csv.gz"), on="snps", how="left"
)
qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(".", expand=True)

outdir = os.path.join(PREDICTDIR, "ensemble_predict/split_by_allele/")
os.makedirs(outdir, exist_ok=True)


def calculate_scores(it):
    l2_holdout_predictions = []
    sum_score_holdout_predictions = []
    sum_log_score_holdout_predictions = []
    # load data
    prefix = f"n{it[0]}_run{it[1]}"
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
    sum_score = pd.DataFrame(
        {
            "expt": scores.sum_score(expt_ref_mean_tracks, expt_alt_mean_tracks),
            "pred": scores.sum_score(pred_ref_mean_tracks, pred_alt_mean_tracks),
        }
    )
    sum_log_score = pd.DataFrame(
        {
            "expt": scores.sum_log_score(expt_ref_mean_tracks, expt_alt_mean_tracks),
            "pred": scores.sum_log_score(pred_ref_mean_tracks, pred_alt_mean_tracks),
        }
    )
    # Filter for holdout SNPs
    l2_holdout_predictions.append(l2.loc[[snp for snp in snps if snp in l2.index]])
    sum_score_holdout_predictions.append(
        sum_score.loc[[snp for snp in snps if snp in sum_score.index]]
    )
    sum_log_score_holdout_predictions.append(
        sum_log_score.loc[[snp for snp in snps if snp in sum_log_score.index]]
    )
    l2_qtls = pd.concat(l2_holdout_predictions)
    sum_qtls = pd.concat(sum_score_holdout_predictions)
    sum_log_qtls = pd.concat(sum_log_score_holdout_predictions)
    l2_qtls.to_csv(os.path.join(outdir, f"{prefix}_l2_scores.csv.gz"))
    sum_qtls.to_csv(os.path.join(outdir, f"{prefix}_sum_scores.csv.gz"))
    sum_log_qtls.to_csv(os.path.join(outdir, f"{prefix}_sum_log_scores.csv.gz"))


with mp.Pool(8) as pool:
    r = list(tqdm.tqdm(pool.imap(calculate_scores, iters), total=len(iters)))
