from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scores
from scipy.stats import pearsonr

prediction_dir = Path("/home2/ayh8/predictions/ensemble/tiqtl/ensemble_predictions/")
expt_fp = "expt_tracks_per_snp_by_allele.joblib.gz"
pred_fp = "pred_tracks_per_snp_by_allele.joblib.gz"
expt = joblib.load(prediction_dir.joinpath(expt_fp))
pred = joblib.load(prediction_dir.joinpath(pred_fp))

qtls = pd.DataFrame({"snps": expt.keys()})
qtl_coord = pd.merge(
    qtls,
    pd.read_csv("Table.7c.tiQTL.2k.txt.gz", sep="\t"),
    on="snps",
    how="left",
)
qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(":", expand=True)
fold_assignments = pd.read_csv("../clipnet_data_folds/data_fold_assignments.csv")
fold_0 = list(fold_assignments[fold_assignments.fold == 0].chrom)
holdout_qtls = set(qtl_coord[qtl_coord.chrom.isin(fold_0)].snps)

ref_n = {k: len(expt[k][0]) for k in expt.keys()}
alt_n = {k: len(expt[k][1]) for k in expt.keys()}

expt_clean = {k: expt[k] for k in expt.keys() if ref_n[k] >= 3 and alt_n[k] >= 3}
pred_clean = {k: pred[k] for k in expt_clean.keys()}

snps = list(expt_clean.keys())
expt_ref_mean_tracks = pd.DataFrame(
    {snp: np.mean(expt_clean[snp][0], axis=0) for snp in snps}
).transpose()
expt_alt_mean_tracks = pd.DataFrame(
    {snp: np.mean(expt_clean[snp][1], axis=0) for snp in snps}
).transpose()
pred_ref_mean_tracks = pd.DataFrame(
    {snp: np.mean(pred_clean[snp][0], axis=0) for snp in snps}
).transpose()
pred_alt_mean_tracks = pd.DataFrame(
    {snp: np.mean(pred_clean[snp][1], axis=0) for snp in snps}
).transpose()
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

l2.to_csv(prediction_dir.joinpath("tiqtls_l2_scores.csv.gz"))
sum_score.to_csv(prediction_dir.joinpath("tiqtls_sum_scores.csv.gz"))
sum_log_score.to_csv(prediction_dir.joinpath("tiqtls_sum_log_scores.csv.gz"))

pearsonr(l2.expt, l2.pred)
