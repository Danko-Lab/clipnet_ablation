import itertools
import multiprocessing as mp
import os

import joblib
import numpy as np
import pandas as pd
import tqdm

PREDICTDIR = "./tiqtl/"

# Load and clean experimental data
expt_fp = "/fs/cbsubscb17/storage/projects/CLIPNET/data/gse110638/tiqtl/expt_tracks_per_snp_by_allele.joblib.gz"
expt = joblib.load(expt_fp)
expt_clean = {
    k: expt[k] for k in expt.keys() if expt[k][0] is not None and expt[k][1] is not None
}
snps = list(expt_clean.keys())
expt_ref_mean_tracks = pd.DataFrame(
    {snp: expt_clean[snp][0].mean(axis=0) for snp in snps}
).transpose()
expt_alt_mean_tracks = pd.DataFrame(
    {snp: expt_clean[snp][1].mean(axis=0) for snp in snps}
).transpose()
qtls = pd.DataFrame({"snps": expt.keys()})
qtl_coord = pd.merge(
    qtls, pd.read_csv("Table.7c.tiQTL.2k.txt.gz", sep="\t"), on="snps", how="left"
)
qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(":", expand=True)

outdir = os.path.join(PREDICTDIR, "ensemble_predict/split_by_allele/")
os.makedirs(outdir, exist_ok=True)


def l2_score(x, y):
    return np.sqrt(np.sum(np.square(x - y), axis=-1))


def logfc_score(x, y):
    return np.log2((x + 1e-6) / (y + 1e-6))


def calculate_scores(it):
    # load data
    prefix = f"n{it[0]}_run{it[1]}"
    pred_fp = f"{prefix}_pred_per_snp_by_allele.joblib.gz"
    pred = joblib.load(os.path.join(outdir, pred_fp))[0]
    # clean data
    pred_ref_mean_tracks = pd.DataFrame(
        {snp: np.mean(pred[snp][0], axis=0) for snp in snps}
    ).transpose()
    pred_alt_mean_tracks = pd.DataFrame(
        {snp: np.mean(pred[snp][1], axis=0) for snp in snps}
    ).transpose()
    # calculate scores
    l2 = pd.DataFrame(
        {
            "expt": l2_score(
                expt_ref_mean_tracks.to_numpy(), expt_alt_mean_tracks.to_numpy()
            ),
            "pred": l2_score(
                pred_ref_mean_tracks.to_numpy(), pred_alt_mean_tracks.to_numpy()
            ),
        }
    )
    # Filter for holdout SNPs
    l2.index = expt_ref_mean_tracks.index
    l2.to_csv(os.path.join(outdir, f"{prefix}_l2_scores.csv.gz"))


# Calculate scores for each iteration
n_individuals = [5, 10, 15, 20, 30, 40, 50]
runs = range(5)

iters = list(itertools.product(n_individuals, runs))

with mp.Pool(10) as pool:
    r = list(tqdm.tqdm(pool.imap(calculate_scores, iters), total=len(iters)))


from scipy.stats import pearsonr, spearmanr

samp = []
corr = []
for it in iters:
    df = pd.read_csv(
        f"tiqtl/ensemble_predict/split_by_allele/n{it[0]}_run{it[1]}_l2_scores.csv.gz"
    )
    samp.append(it[0])
    corr.append(pearsonr(df["expt"], df["pred"])[0])

spearmanr(samp, corr)
