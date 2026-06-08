#!/usr/bin/env python3

"""
Generate expt_tracks_per_snp_by_allele.joblib.gz for QTL benchmarks.

The output matches the legacy CLIPNET QTL format consumed by the scoring
scripts: a dictionary mapping each SNP ID to three allele buckets:

    [reference_allele_tracks, alternate_allele_tracks, heterozygous_tracks]

Each bucket is a NumPy array of observed PRO-cap profiles for individuals with
that allele call, or None when no individuals have that allele call.
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

try:
    from qtl_filters import (
        build_filter_report,
        eligible_snps,
        load_prefix_map,
        print_filter_summary,
        write_filter_report,
    )
except ImportError:
    from evaluation_qtl.qtl_filters import (
        build_filter_report,
        eligible_snps,
        load_prefix_map,
        print_filter_summary,
        write_filter_report,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

QTL_CONFIG = {
    "diqtl": {
        "alleles": "diQTL_snps_per_individual.csv.gz",
        "table": "Table.10a.diQTL.2kb.csv.gz",
        "table_sep": ",",
        "output": "expt_tracks_per_snp_by_allele.joblib.gz",
    },
    "tiqtl": {
        "alleles": "tiQTL_snps_per_individual.csv.gz",
        "table": "Table.7c.tiQTL.2k.txt.gz",
        "table_sep": "\t",
        "output": "expt_tracks_per_snp_by_allele.joblib.gz",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qtl",
        choices=sorted(QTL_CONFIG),
        required=True,
        help="QTL dataset. Used to resolve default allele/output filenames.",
    )
    parser.add_argument(
        "--qtl_data_dir",
        type=Path,
        default=None,
        help="Directory containing the allele matrix and output joblib.",
    )
    parser.add_argument(
        "--allele_matrix",
        type=Path,
        default=None,
        help="CSV(.gz) of SNP allele calls, indexed by PRO-cap prefix.",
    )
    parser.add_argument(
        "--qtl_table",
        type=Path,
        default=None,
        help="QTL association table containing snps, gene, and p-value columns.",
    )
    parser.add_argument(
        "--prefix_map",
        type=Path,
        default=REPO_ROOT / "data_spec/procap_to_1k_genomes.json",
        help="JSON mapping PRO-cap library prefixes to unique individuals.",
    )
    parser.add_argument(
        "--pvalue_column",
        default=None,
        help="QTL-table p-value column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--filter_report",
        type=Path,
        default=None,
        help="Per-SNP filtering report path. Defaults to the QTL data directory.",
    )
    parser.add_argument(
        "--signals_dir",
        type=Path,
        required=True,
        help="Directory containing observed PRO-cap signal files.",
    )
    parser.add_argument(
        "--signal_template",
        default="{prefix}.csv.gz",
        help="Signal filename template relative to --signals_dir. May contain {prefix}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output joblib path. Defaults to {qtl_data_dir}/expt_tracks_per_snp_by_allele.joblib.gz.",
    )
    parser.add_argument(
        "--prefixes",
        default=None,
        help="Optional comma-separated prefixes. Defaults to allele matrix rows.",
    )
    parser.add_argument(
        "--sequence_dir",
        type=Path,
        default=None,
        help="Optional FASTA directory used to map signal rows by FASTA header IDs.",
    )
    parser.add_argument(
        "--fasta_template",
        default="{prefix}.fna.gz",
        help="FASTA filename template relative to --sequence_dir. May contain {prefix}.",
    )
    parser.add_argument(
        "--no_signal_row_index",
        action="store_true",
        help="Treat CSV signal files as numeric matrices without a leading row-ID column.",
    )
    parser.add_argument(
        "--allow_row_order",
        action="store_true",
        help="Allow row-order mapping when signal/FASTA row IDs do not match SNP IDs.",
    )
    parser.add_argument(
        "--skip_missing_prefixes",
        action="store_true",
        help="Skip prefixes with missing signal files instead of failing.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output track dtype. Use float16 to reduce size if desired.",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=3,
        help="joblib compression level.",
    )
    return parser.parse_args()


def default_qtl_data_dir(qtl):
    if qtl is None:
        return None
    return SCRIPT_DIR / qtl


def resolve_allele_matrix(args):
    if args.allele_matrix is not None:
        return args.allele_matrix
    if args.qtl is None:
        raise ValueError("--qtl or --allele_matrix is required.")
    qdir = args.qtl_data_dir or default_qtl_data_dir(args.qtl)
    return qdir / QTL_CONFIG[args.qtl]["alleles"]


def resolve_output(args):
    if args.output is not None:
        return args.output
    if args.qtl is None:
        raise ValueError("--qtl or --output is required.")
    qdir = args.qtl_data_dir or default_qtl_data_dir(args.qtl)
    return qdir / QTL_CONFIG[args.qtl]["output"]


def templated_path(root, template, prefix):
    return root / template.format(prefix=prefix)


def selected_prefixes(args, allele_matrix):
    if args.prefixes is None:
        return list(allele_matrix.index)
    return [prefix.strip() for prefix in args.prefixes.split(",") if prefix.strip()]


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def fasta_sequence_ids(path):
    ids = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def load_signal(path, no_row_index=False):
    name = path.name
    if name.endswith(".npz"):
        return np.load(path)["arr_0"], None
    if name.endswith(".npy"):
        return np.load(path), None
    if name.endswith(".h5") or name.endswith(".hdf5"):
        import h5py

        with h5py.File(path, "r") as handle:
            if "track" in handle:
                values = handle["track"][:]
            elif "arr_0" in handle:
                values = handle["arr_0"][:]
            else:
                first_key = next(iter(handle.keys()))
                values = handle[first_key][:]
        return values, None
    if name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".txt.gz"):
        if no_row_index:
            return pd.read_csv(path, header=None).to_numpy(), None
        frame = pd.read_csv(path, header=None, index_col=0)
        return frame.to_numpy(), [str(idx) for idx in frame.index]
    raise ValueError(f"Unsupported signal file format: {path}")


def match_sequence_id(sequence_id, snps):
    if sequence_id in snps:
        return sequence_id
    for field in sequence_id.replace("|", " ").split():
        if field in snps:
            return field
    return None


def row_order_for_prefix(args, prefix, signal_row_ids, n_rows, snps):
    snp_set = set(snps)
    if signal_row_ids is not None:
        matches = [
            (row_idx, match_sequence_id(row_id, snp_set))
            for row_idx, row_id in enumerate(signal_row_ids)
        ]
        matches = [(idx, snp) for idx, snp in matches if snp is not None]
        if matches:
            return matches

    if args.sequence_dir is not None:
        fasta_path = templated_path(args.sequence_dir, args.fasta_template, prefix)
        if fasta_path.exists():
            fasta_ids = fasta_sequence_ids(fasta_path)
            if len(fasta_ids) != n_rows:
                raise ValueError(
                    f"{prefix}: FASTA records ({len(fasta_ids)}) do not match "
                    f"signal rows ({n_rows})."
                )
            matches = [
                (row_idx, match_sequence_id(row_id, snp_set))
                for row_idx, row_id in enumerate(fasta_ids)
            ]
            matches = [(idx, snp) for idx, snp in matches if snp is not None]
            if matches:
                return matches

    if args.allow_row_order:
        if n_rows != len(snps):
            raise ValueError(
                f"{prefix}: cannot use row-order mapping because signal rows "
                f"({n_rows}) != SNP columns ({len(snps)})."
            )
        return list(enumerate(snps))

    raise ValueError(
        f"{prefix}: could not map signal rows to SNP IDs. Provide matching row IDs, "
        "--sequence_dir, or --allow_row_order."
    )


def allele_bucket(value):
    if pd.isna(value):
        return None
    if value == 0:
        return 0
    if value == 1:
        return 1
    if value == 0.5:
        return 2
    return None


def generate(args):
    allele_fp = resolve_allele_matrix(args)
    output_fp = resolve_output(args)
    allele_matrix = pd.read_csv(allele_fp, index_col=0)
    prefixes = selected_prefixes(args, allele_matrix)
    snps = list(allele_matrix.columns)
    config = QTL_CONFIG[args.qtl]
    qdir = args.qtl_data_dir or default_qtl_data_dir(args.qtl)
    table_fp = args.qtl_table or qdir / config["table"]
    qtl_table = pd.read_csv(table_fp, sep=config["table_sep"])
    report = build_filter_report(
        args.qtl,
        allele_matrix,
        qtl_table,
        load_prefix_map(args.prefix_map),
        pvalue_column=args.pvalue_column,
    )
    report_fp = args.filter_report or qdir / "qtl_filter_report.csv.gz"
    eligible = set(eligible_snps(report))

    tracks = {snp: [[], [], []] for snp in snps if snp in eligible}
    n_prefixes = 0

    for prefix in tqdm.tqdm(prefixes, desc="Grouping observed tracks by allele"):
        if prefix not in allele_matrix.index:
            raise ValueError(f"Prefix {prefix} is not in {allele_fp}.")
        signal_fp = templated_path(args.signals_dir, args.signal_template, prefix)
        if not signal_fp.exists():
            if args.skip_missing_prefixes:
                continue
            raise FileNotFoundError(signal_fp)

        signal, signal_row_ids = load_signal(
            signal_fp, no_row_index=args.no_signal_row_index
        )
        signal = np.asarray(signal, dtype=args.dtype)
        if signal.ndim != 2:
            raise ValueError(f"{signal_fp} must be a 2D signal matrix.")
        row_order = row_order_for_prefix(args, prefix, signal_row_ids, signal.shape[0], snps)

        for row_idx, snp in row_order:
            if snp not in eligible:
                continue
            bucket = allele_bucket(allele_matrix.at[prefix, snp])
            if bucket is None:
                continue
            track = signal[row_idx]
            tracks[snp][bucket].append(track)
        n_prefixes += 1

    if n_prefixes == 0:
        raise ValueError("No prefixes were processed.")

    tracks = {
        snp: [
            np.asarray(bucket, dtype=args.dtype) if bucket else None
            for bucket in allele_buckets
        ]
        for snp, allele_buckets in tracks.items()
    }
    available = {
        snp
        for snp, values in tracks.items()
        if values[0] is not None
        and values[1] is not None
        and np.asarray(values[0]).size
        and np.asarray(values[1]).size
    }
    missing_tracks = report.index.difference(list(available))
    report.loc[missing_tracks, "has_experimental_tracks"] = False
    newly_missing = report.index[
        report["eligible"] & ~report["has_experimental_tracks"]
    ]
    report.loc[newly_missing, "eligible"] = False
    report.loc[newly_missing, "exclusion_reasons"] = report.loc[
        newly_missing, "exclusion_reasons"
    ].map(lambda value: f"{value};missing_experimental_tracks".strip(";"))
    tracks = {snp: tracks[snp] for snp in eligible_snps(report)}
    write_filter_report(report, report_fp)
    print_filter_summary(report, args.qtl)
    print(f"Saved QTL filter report to {report_fp}")

    output_fp.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(tracks, output_fp, compress=args.compression)
    print(f"Wrote {output_fp}")
    print(f"Processed prefixes: {n_prefixes}")
    print(f"Eligible SNPs written: {len(tracks)}")


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
