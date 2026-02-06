## NOT IMPLEMENTED

"""
This script calculates performance metrics given a set of predictions and ground truth.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--predictions",
        type=str,
        required=True,
        help="An npz file containing the predicted procap profiles and quantities.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="A csv(.gz), npy, or npz file containing the observed procap tracks.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="An npz file to write the performance metrics to.",
    )
    args = parser.parse_args()

    # Load predictions
    predictions = np.load(args.predictions)
    track = predictions["arr_0"]
    quantity = predictions["arr_1"]
    if len(quantity.shape) == 2:
        quantity = quantity.squeeze()

    # Load observed data
    if args.experiment.endswith(".npz") or args.experiment.endswith(".npy"):
        observed = np.load(args.experiment)
        if args.experiment.endswith(".npz"):
            observed = observed["arr_0"]
    elif args.experiment.endswith(".csv.gz") or args.experiment.endswith(".csv"):
        observed = pd.read_csv(args.experiment, header=None, index_col=0).to_numpy()
    else:
        raise ValueError(
            f"File with observed PRO-cap data ({args.experiment}) must be numpy or csv format."
        )

    # Validate dimensions
    if track.shape[0] != observed.shape[0]:
        raise ValueError(
            f"n predictions ({track.shape[0]}) and n observed ({observed.shape[0]}) do not match."
        )
    if track.shape[1] > observed.shape[1]:
        raise ValueError(
            f"Predicted tracks ({track.shape[1]}) are longer than observed ({observed.shape[1]})."
        )
    if (observed.shape[1] - track.shape[1]) % 4 != 0:
        raise ValueError(
            f"Padding around predicted tracks ({observed.shape[1] - track.shape[1]}) must be divisible by 4."
        )

    # Trim off padding for observed tracks
    start = (observed.shape[1] - track.shape[1]) // 4
    end = observed.shape[1] // 2 - start
    observed_clipped = observed[
        :,
        np.r_[start:end, observed.shape[1] // 2 + start : observed.shape[1] // 2 + end],
    ]

    # Benchmark directionality
    track_directionality = np.log1p(
        track[:, : track.shape[1] // 2].sum(axis=1)
    ) - np.log1p(track[:, track.shape[1] // 2 :].sum(axis=1))
    observed_directionality = np.log1p(
        observed_clipped[:, : observed_clipped.shape[1] // 2].sum(axis=1)
    ) - np.log1p(observed_clipped[:, observed_clipped.shape[1] // 2 :].sum(axis=1))
    directionality_pearson = pearsonr(track_directionality, observed_directionality)

    # Benchmark TSS position
    strand_break = track.shape[1] // 2
    pred_tss = np.concatenate(
        [track[:, :strand_break].argmax(axis=1), track[:, strand_break:].argmax(axis=1)]
    )
    obs_tss = np.concatenate(
        [
            observed_clipped[:, :strand_break].argmax(axis=1),
            observed_clipped[:, strand_break:].argmax(axis=1),
        ]
    )
    tss_pos_pearson = pearsonr(pred_tss, obs_tss)

    # Benchmark profile
    track_pearson = pd.DataFrame(track).corrwith(pd.DataFrame(observed_clipped), axis=1)
    track_js_distance = jensenshannon(track, observed_clipped, axis=1)

    # Benchmark quantity
    quantity_log_pearson = pearsonr(
        np.log1p(quantity), np.log1p(observed_clipped.sum(axis=1))
    )
    quantity_spearman = spearmanr(quantity, observed_clipped.sum(axis=1))

    # Print summary
    print(f"Median Track Pearson: {track_pearson.median():.4f}")
    print(
        f"Mean Track Pearson: {track_pearson.mean():.4f} "
        + f"+/- {track_pearson.std():.4f}"
    )
    print(f"Median Track JS Distance: {pd.Series(track_js_distance).median():.4f} ")
    print(
        f"Mean Track JS Distance: {pd.Series(track_js_distance).mean():.4f} "
        + f"+/- {pd.Series(track_js_distance).std():.4f}"
    )
    print(f"Track Directionality Pearson: {directionality_pearson[0]:.4f}")
    print(f"TSS Position Pearson: {tss_pos_pearson[0]:.4f}")
    print(f"Quantity Log Pearson: {quantity_log_pearson[0]:.4f}")
    print(f"Quantity Spearman: {quantity_spearman[0]:.4f}")

    # Save metrics
    if args.output is not None:
        np.savez_compressed(
            args.output,
            profile_pearson=track_pearson.numpy(),
            profile_jsd=track_js_distance,
            directionality_pearson=directionality_pearson[0],
            tss_pos_pearson=tss_pos_pearson[0],
            quantity_log_pearson=quantity_log_pearson[0],
            quantity_spearman=quantity_spearman[0],
        )


if __name__ == "__main__":
    main()
