from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scores
from scipy.stats import pearsonr, spearmanr

prediction_dir = Path("/home2/ayh8/predictions/ensemble/tiqtl/individual_folds/")
folds = range(10)
data_splits = pd.read_csv("../clipnet_data_folds/data_fold_assignments.csv")
holdouts = {
    i: list(data_splits[data_splits.fold == int(i)].chrom) for i in range(len(folds))
}
expt_fp = "expt_tracks_per_snp_by_allele.joblib.gz"
pred_fp = "pred_tracks_per_snp_by_allele.joblib.gz"

l2_holdout_predictions = []
sum_score_holdout_predictions = []
sum_log_score_holdout_predictions = []

for fold in folds[1:]:
    path = prediction_dir.joinpath(f"tiqtl_{fold}")
    expt = joblib.load(path.joinpath(expt_fp))
    pred = joblib.load(path.joinpath(pred_fp))
    ref_n = {k: len(expt[k][0]) for k in expt.keys()}
    alt_n = {k: len(expt[k][1]) for k in expt.keys()}
    expt_clean = {k: expt[k] for k in expt.keys() if ref_n[k] >= 3 and alt_n[k] >= 3}
    pred_clean = {k: pred[k] for k in expt_clean.keys()}
    qtls = pd.DataFrame({"snps": expt.keys()})
    qtl_coord = pd.merge(
        qtls,
        pd.read_csv("Table.7c.tiQTL.2k.txt.gz", sep="\t"),
        on="snps",
        how="left",
    )
    qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(":", expand=True)
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
    test = qtl_coord[qtl_coord.chrom.isin(holdouts[fold])]
    l2_holdout_predictions.append(
        l2.loc[[snp for snp in set(test["snps"]) if snp in l2.index]]
    )
    sum_score_holdout_predictions.append(
        sum_score.loc[[snp for snp in set(test["snps"]) if snp in sum_score.index]]
    )
    sum_log_score_holdout_predictions.append(
        sum_log_score.loc[
            [snp for snp in set(test["snps"]) if snp in sum_log_score.index]
        ]
    )

l2_qtls = pd.concat(l2_holdout_predictions)
sum_qtls = pd.concat(sum_score_holdout_predictions)
sum_log_qtls = pd.concat(sum_log_score_holdout_predictions)

l2_qtls.to_csv(prediction_dir.joinpath("tiqtls_l2_scores.csv.gz"))
sum_qtls.to_csv(prediction_dir.joinpath("tiqtls_sum_scores.csv.gz"))
sum_log_qtls.to_csv(prediction_dir.joinpath("tiqtls_sum_log_scores.csv.gz"))

pearsonr(l2_qtls.expt, l2_qtls.pred)
spearmanr(l2_qtls.expt, l2_qtls.pred)
pearsonr(sum_qtls.expt, sum_qtls.pred)
spearmanr(sum_qtls.expt, sum_qtls.pred)
