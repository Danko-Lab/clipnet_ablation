#!/usr/bin/env python3

"""
Takes a directory of fasta files, a bed file, and an output directory. Uses bedtools getfasta to extract
corresponding regions from each fasta file. Assumes fasta files have a prefix equal matching the first segment of the
bed file prefixes.
"""

import argparse
import itertools as it
import json
import multiprocessing as mp
import os


def gsw(exp_dir, exp_prefix, ref_fna, out_dir):
    bed = os.path.join(exp_dir, "%s_window_uniq.bed" % exp_prefix)
    out = os.path.join(out_dir, f"{exp_prefix}.fna")
    cmd = f"bedtools getfasta -fi {ref_fna} -bed {bed} -fo {out}"
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bed_dir", type=str, help="input bed directory")
    parser.add_argument("ref_fna", type=str, help="reference fasta file")
    parser.add_argument("out_dir", type=str, help="output directory")
    parser.add_argument(
        "file_prefix_hash",
        type=str,
        help="a json file to convert between experiment and fasta prefix",
    )
    parser.add_argument(
        "--missing",
        type=str,
        default=None,
        help="a list of fasta files that are missing and must be excluded",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="number of threads to use",
    )
    args = parser.parse_args()

    with open(args.file_prefix_hash, "r") as handle:
        d = json.load(handle)
    if args.missing is None:
        missing = []
    else:
        with open(args.missing, "r") as handle:
            missing = handle.read().splitlines()
    exp_prefixes = list(d.keys())
    nonempty_exp_prefixes = [
        prefix for prefix in exp_prefixes if d[prefix] not in missing
    ]

    p = mp.Pool(min(args.threads, int(mp.cpu_count() / 2)))
    p.starmap(
        gsw,
        zip(
            it.repeat(args.bed_dir),
            nonempty_exp_prefixes,
            it.repeat(args.ref_fna),
            it.repeat(args.out_dir),
        ),
    )
