#!/usr/bin/env python3

"""
Validate QTL benchmark inputs before running model scoring.

This checks that the original QTL table, allele matrix, experimental tracks,
sequence FASTAs, and chromosome fold assignments agree on SNP identities and
coordinate-derived holdout folds. It is intended as a lightweight guard against
mixing the original CLIPNET Zenodo QTL files with reprocessed variant/QTL sets.
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

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
        "gene_sep": ".",
        "expt": "expt_tracks_per_snp_by_allele.joblib.gz",
    },
    "tiqtl": {
        "alleles": "tiQTL_snps_per_individual.csv.gz",
        "table": "Table.7c.tiQTL.2k.txt.gz",
        "table_sep": "\t",
        "gene_sep": ":",
        "expt": "expt_tracks_per_snp_by_allele.joblib.gz",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtl", choices=sorted(QTL_CONFIG), required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root containing {qtl}/sequence/{prefix}.fna.gz.",
    )
    parser.add_argument(
        "--qtl_data_dir",
        type=Path,
        default=None,
        help="Directory containing the QTL table, allele matrix, and experimental tracks.",
    )
    parser.add_argument(
        "--fold_assignments",
        type=Path,
        default=REPO_ROOT / "clipnet_data_folds/data_fold_assignments.csv",
        help="Chromosome fold assignment CSV.",
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
        help="Per-SNP filter report. Defaults to qtl_filter_report.csv.gz in the QTL data directory.",
    )
    parser.add_argument(
        "--prefixes",
        default=None,
        help="Optional comma-separated FASTA prefixes to inspect. Defaults to allele rows.",
    )
    parser.add_argument(
        "--max_prefixes",
        type=int,
        default=5,
        help="Maximum FASTA files to inspect when --prefixes is not supplied.",
    )
    return parser.parse_args()


def qtl_data_dir(args):
    return args.qtl_data_dir or SCRIPT_DIR / args.qtl


def fasta_sequence_ids(fasta_fp):
    opener = gzip.open if str(fasta_fp).endswith(".gz") else open
    ids = []
    with opener(fasta_fp, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def existing_path(label, path):
    status = "ok" if path.exists() else "missing"
    print(f"{label}: {path} [{status}]")
    return path.exists()


def load_experimental_tracks(path):
    import joblib

    expt = joblib.load(path)
    if isinstance(expt, (list, tuple)):
        if len(expt) != 2 or not isinstance(expt[0], dict):
            raise TypeError(f"Unsupported experimental track format in {path}.")
        print(
            "warning: experimental file uses prediction-style [tracks, quantities] "
            "format; regenerate it with generate_expt_tracks_by_allele.py."
        )
        expt = expt[0]
    if not isinstance(expt, dict):
        raise TypeError(f"Expected a SNP-to-allele dictionary in {path}.")
    valid = {
        snp: values
        for snp, values in expt.items()
        if values[0] is not None
        and values[1] is not None
        and np.asarray(values[0]).size
        and np.asarray(values[1]).size
    }
    return expt, valid


def print_set_delta(label, left_name, left, right_name, right, limit=8):
    missing = sorted(left - right)
    extra = sorted(right - left)
    print(f"{label}: {len(left & right):,} shared")
    print(f"  {left_name} not in {right_name}: {len(missing):,}")
    if missing:
        print(f"    examples: {', '.join(missing[:limit])}")
    print(f"  {right_name} not in {left_name}: {len(extra):,}")
    if extra:
        print(f"    examples: {', '.join(extra[:limit])}")


def load_qtl_coordinates(args, expt_snps):
    config = QTL_CONFIG[args.qtl]
    table_fp = qtl_data_dir(args) / config["table"]
    qtl_table = pd.read_csv(table_fp, sep=config["table_sep"])
    duplicate_snps = int(qtl_table["snps"].duplicated().sum())
    qtl_table = qtl_table.drop_duplicates(subset="snps", keep="first")
    qtls = pd.DataFrame({"snps": list(expt_snps)})
    qtl_coord = pd.merge(qtls, qtl_table, on="snps", how="left")
    missing_gene = qtl_coord["gene"].isna()
    if missing_gene.any():
        return qtl_table, qtl_coord, duplicate_snps
    qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(
        config["gene_sep"], expand=True
    )
    return qtl_table, qtl_coord, duplicate_snps


def selected_prefixes(args, allele_matrix):
    if args.prefixes:
        return [prefix.strip() for prefix in args.prefixes.split(",") if prefix.strip()]
    return list(allele_matrix.index[: args.max_prefixes])


def validate_fastas(args, allele_matrix, expt_snps):
    allele_snps = set(allele_matrix.columns)
    prefixes = selected_prefixes(args, allele_matrix)
    if not prefixes:
        print("No FASTA prefixes selected.")
        return

    print("\nFASTA checks")
    for prefix in prefixes:
        fasta_fp = args.data_root / args.qtl / "sequence" / f"{prefix}.fna.gz"
        if not fasta_fp.exists():
            fasta_fp = fasta_fp.with_suffix("")
        if not existing_path(f"  {prefix}", fasta_fp):
            continue
        ids = fasta_sequence_ids(fasta_fp)
        id_set = set(ids)
        duplicate_ids = len(ids) - len(id_set)
        print(f"    records: {len(ids):,}; duplicate headers: {duplicate_ids:,}")
        print(f"    header IDs in allele SNPs: {len(id_set & allele_snps):,}")
        print(f"    header IDs in valid experimental SNPs: {len(id_set & expt_snps):,}")
        if len(ids) != len(allele_matrix.columns):
            print(
                "    warning: FASTA record count differs from allele-matrix SNP columns; "
                "row-order fallback would be unsafe."
            )
        if ids and not (id_set & allele_snps):
            print(
                "    warning: FASTA headers do not directly match SNP IDs; predictions "
                "must rely on exact row order."
            )


def validate_fold_assignments(args, qtl_coord):
    if qtl_coord["gene"].isna().any():
        missing = qtl_coord[qtl_coord["gene"].isna()]
        print(
            "\nFold checks skipped: "
            f"{len(missing):,} experimental SNPs are missing from the QTL coordinate table."
        )
        print(f"  examples: {', '.join(missing['snps'].astype(str).head(8))}")
        return

    folds = pd.read_csv(args.fold_assignments)
    missing_cols = {"chrom", "fold"} - set(folds.columns)
    if missing_cols:
        print(f"\nFold checks skipped: missing columns {sorted(missing_cols)}")
        return

    assigned_chroms = set(folds["chrom"])
    qtl_chroms = set(qtl_coord["chrom"])
    print("\nFold checks")
    print_set_delta("Chromosome assignment", "QTL chroms", qtl_chroms, "fold chroms", assigned_chroms)
    merged = qtl_coord.merge(folds, on="chrom", how="left")
    missing_fold = int(merged["fold"].isna().sum())
    print(f"  experimental SNPs without assigned fold: {missing_fold:,}")
    counts = merged.groupby("fold", dropna=False).size().sort_index()
    for fold, count in counts.items():
        print(f"  fold {fold}: {count:,} experimental SNPs")


def main():
    args = parse_args()
    config = QTL_CONFIG[args.qtl]
    qdir = qtl_data_dir(args)
    allele_fp = qdir / config["alleles"]
    table_fp = qdir / config["table"]
    expt_fp = qdir / config["expt"]

    print(f"QTL dataset: {args.qtl}")
    print("\nRequired files")
    paths_ok = [
        existing_path("allele matrix", allele_fp),
        existing_path("QTL table", table_fp),
        existing_path("experimental tracks", expt_fp),
        existing_path("fold assignments", args.fold_assignments),
        existing_path("prefix map", args.prefix_map),
    ]
    if not all(paths_ok):
        raise SystemExit("One or more required files are missing.")

    allele_matrix = pd.read_csv(allele_fp, index_col=0)
    _, valid_expt = load_experimental_tracks(expt_fp)
    raw_qtl_table = pd.read_csv(table_fp, sep=config["table_sep"])
    report = build_filter_report(
        args.qtl,
        allele_matrix,
        raw_qtl_table,
        load_prefix_map(args.prefix_map),
        pvalue_column=args.pvalue_column,
        available_experimental_snps=set(valid_expt),
    )
    report_fp = args.filter_report or qdir / "qtl_filter_report.csv.gz"
    write_filter_report(report, report_fp)
    print_filter_summary(report, args.qtl)
    print(f"  filter report: {report_fp}")
    filtered_snps = set(eligible_snps(report))
    qtl_table, qtl_coord, duplicate_table_snps = load_qtl_coordinates(
        args, filtered_snps
    )

    allele_snps = set(allele_matrix.columns)
    expt_snps = set(valid_expt)
    table_snps = set(qtl_table["snps"])

    print("\nSNP checks")
    print(f"allele matrix: {allele_matrix.shape[0]:,} prefixes x {allele_matrix.shape[1]:,} SNPs")
    print(f"valid experimental tracks: {len(expt_snps):,} SNPs")
    print(f"manuscript-filtered experimental tracks: {len(filtered_snps):,} SNPs")
    print(f"QTL table: {len(table_snps):,} unique SNPs ({duplicate_table_snps:,} duplicate rows dropped)")
    print_set_delta("Allele vs experimental", "allele SNPs", allele_snps, "experimental SNPs", expt_snps)
    print_set_delta("Experimental vs QTL table", "experimental SNPs", expt_snps, "QTL-table SNPs", table_snps)

    missing_gene = int(qtl_coord["gene"].isna().sum())
    print(f"experimental SNPs missing coordinate/gene annotation: {missing_gene:,}")
    if missing_gene:
        examples = qtl_coord[qtl_coord["gene"].isna()]["snps"].astype(str).head(8)
        print(f"  examples: {', '.join(examples)}")

    validate_fold_assignments(args, qtl_coord)
    validate_fastas(args, allele_matrix, filtered_snps)


if __name__ == "__main__":
    main()
