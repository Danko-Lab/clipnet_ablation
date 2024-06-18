"""
Calculate deeplift contribution scores using shap.DeepExplainer.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyfastx
import shap
import tensorflow as tf
import ushuffle

sys.path.append("../../clipnet/")
import clipnet
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
shuffled_ref = [ushuffle.shuffle(rec.seq, 1000, 2) for rec in reference]

# One-hot encode shuffled sequences and select 5000 to use as reference
onehot_ref = np.array([utils.OneHotDNA(seq).onehot for seq in shuffled_ref])

# Load target sequences
explain_fasta_fp = "/home2/ayh8/data/lcl/examples/examples_reference_seq.fna.gz"
seqs_to_explain = utils.get_onehot_fasta_sequences(explain_fasta_fp)

# Load models and create explainers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nn = clipnet.CLIPNET(n_gpus=0)
models = [
    tf.keras.models.load_model(
        f"/home2/ayh8/ensemble_models/fold_{i}.h5", compile=False
    )
    for i in FOLDS
]
explainers = []
for model in models:
    profile_contrib = tf.reduce_mean(
        tf.stop_gradient(tf.nn.softmax(model.output[0], axis=-1)) * model.output[0],
        axis=-1,
        keepdims=True,
    )
    explainers.append(shap.DeepExplainer((model.input, profile_contrib), onehot_ref))

# Main explanations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

raw_explanations = [explainer.shap_values(seqs_to_explain) for explainer in explainers]
scaled_explanations = {
    f"fold_{i}": (np.sum(raw_explanations[i - 1], axis=0) * seqs_to_explain).swapaxes(
        1, 2
    )
    for i in FOLDS
}

out_dir = Path("/home2/ayh8/attribution_scores/").joinpath("examples")
out_dir.mkdir(exist_ok=True, parents=True)

np.savez_compressed(
    out_dir.joinpath("folds_examples_profile.npz"),
    **scaled_explanations,
)


# QTL explanations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

raw_qtl_explanations = [
    explainer.shap_values(qtls_to_explain) for explainer in explainers
]
scaled_qtl_explanations = {
    f"fold_{i}": (
        np.sum(raw_qtl_explanations[i - 1], axis=0) * qtls_to_explain
    ).swapaxes(1, 2)
    for i in FOLDS
}

out_dir = Path("/home2/ayh8/attribution_scores/").joinpath("examples")
out_dir.mkdir(exist_ok=True, parents=True)

np.savez_compressed(
    out_dir.joinpath("folds_qtl_examples_profile.npz"),
    **scaled_qtl_explanations,
)
