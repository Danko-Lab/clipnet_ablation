"""
Calculate deeplift contribution scores using shap.DeepExplainer.
"""

import gc
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pyfastx
import shap
import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
sys.path.append("../../clipnet/")
import clipnet
import tensorflow as tf
import utils

# This will fix an error message for running tf.__version__==2.5
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
    shap.explainers._deep.deep_tf.passthrough
)
tf.compat.v1.disable_v2_behavior()


np.random.seed(617)
n_subset = 100
FOLDS = range(1, 10)

# Load sequences
ref_fp = os.path.join(
    "/home2/ayh8/data/lcl/tfbs_sampling/",
    "random_5000/random_tss_windows_reference_seq.fna.gz",
)
ref_seqs = pyfastx.Fasta(ref_fp)

# Perform dinucleotide shuffle on 1000 random sequences
reference = [
    ref_seqs[i]
    for i in np.random.choice(
        np.array(range(len(ref_seqs))), size=n_subset, replace=False
    )
]
shuffled_ref = [utils.kshuffle(rec.seq, random_seed=617)[0] for rec in reference]

# One-hot encode shuffled sequences and select 5000 to use as reference
twohot_ref = np.array([utils.TwoHotDNA(seq).twohot for seq in shuffled_ref])

# Load target sequences
explain_fasta_fp = "/home2/ayh8/data/lcl/examples/examples_reference_seq.fna.gz"
seqs_to_explain = utils.get_twohot_fasta_sequences(explain_fasta_fp)

out_dir = Path("/home2/ayh8/attribution_scores/lcl/examples")
out_dir.mkdir(exist_ok=True, parents=True)

# Create QTL sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

qtls_to_explain = np.array(
    [
        # rs185220 (chr5:56909530)
        seqs_to_explain[51],
        seqs_to_explain[51],
        # rs8050061 (chr16:80231939)
        seqs_to_explain[19],
        seqs_to_explain[19],
        # rs1016110 (chr22:42440331)
        seqs_to_explain[47],
        seqs_to_explain[47],
    ]
)
# rs185220 (chr5:56909530) Convert minor/reference allele (A) to major allele (G)
qtls_to_explain[0, 500, :] = np.array([0, 0, 2, 0])
# rs8050061 (chr16:80231939) Convert major/reference allele (T) to minor allele (C)
qtls_to_explain[2, 500, :] = np.array([0, 2, 0, 0])
# rs1016110 (chr22:42440331) Convert major/reference allele (A) to minor allele (C)
qtls_to_explain[4, 499, :] = np.array([0, 2, 0, 0])

# Load models and create explainers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nn = clipnet.CLIPNET(n_gpus=1, use_specific_gpu=1)

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

attribution_scores = {}
for n, r in tqdm.tqdm(itertools.product(n_individuals, runs)):
    name = f"n{n}_run{r}"
    models = [
        tf.keras.models.load_model(f"../models/{name}/fold_{i}.h5", compile=False)
        for i in FOLDS
    ]
    explainers = []
    for model in models:
        explainers.append(
            shap.DeepExplainer((model.input, model.output[1]), twohot_ref)
        )
    raw_qtl_explanations = [
        explainer.shap_values(qtls_to_explain) for explainer in explainers
    ]
    scaled_qtl_explanations = np.array(
        [
            (np.sum(raw_qtl_explanations[i - 1], axis=0) * qtls_to_explain).swapaxes(
                1, 2
            )
            for i in FOLDS
        ]
    ).mean(axis=0)
    attribution_scores[name] = scaled_qtl_explanations
    tf.keras.backend.clear_session()
    gc.collect()

np.savez_compressed(
    out_dir.joinpath("subsample_qtl_examples_quantity.npz"), **attribution_scores
)
