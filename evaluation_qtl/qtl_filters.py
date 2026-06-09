"""Shared manuscript QTL filtering utilities."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


P_VALUE_THRESHOLDS = {"tiqtl": 1e-6, "diqtl": 1e-3}
P_VALUE_NAMES = {
    "p",
    "pval",
    "pvalue",
    "pvalues",
    "p_value",
    "p.value",
    "nominal_pval",
    "pval_nominal",
    "nominal_pvalue",
    "nominal_p_value",
}
VALID_DOSAGES = (0.0, 0.5, 1.0)


def normalize_column_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_pvalue_column(qtl_table, override=None):
    if override is not None:
        if override not in qtl_table.columns:
            raise ValueError(
                f"Requested p-value column {override!r} is not present. "
                f"Available columns: {list(qtl_table.columns)}"
            )
        return override

    normalized_candidates = {
        normalize_column_name(name) for name in P_VALUE_NAMES
    }
    matches = [
        column
        for column in qtl_table.columns
        if normalize_column_name(column) in normalized_candidates
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "Could not identify a QTL p-value column. Pass --pvalue_column. "
            f"Available columns: {list(qtl_table.columns)}"
        )
    raise ValueError(
        "Multiple possible QTL p-value columns were found: "
        f"{matches}. Pass --pvalue_column explicitly."
    )


def load_prefix_map(path):
    with open(path) as handle:
        prefix_map = json.load(handle)
    return {str(prefix): str(individual) for prefix, individual in prefix_map.items()}


def resolve_prefixes(allele_matrix, prefix_map):
    missing = [str(prefix) for prefix in allele_matrix.index if str(prefix) not in prefix_map]
    if missing:
        raise ValueError(
            "Allele-matrix prefixes are missing from the prefix-to-individual map: "
            f"{missing[:5]} ({len(missing)} total)."
        )
    return pd.Series(
        [prefix_map[str(prefix)] for prefix in allele_matrix.index],
        index=allele_matrix.index,
        name="individual",
    )


def dosage_value(value):
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "invalid"
    for valid in VALID_DOSAGES:
        if np.isclose(numeric, valid):
            return valid
    return "invalid"


def genotype_summary(values, individuals):
    calls = {}
    invalid = False
    conflict = False
    library_calls = []
    for prefix, value in values.items():
        dosage = dosage_value(value)
        if dosage is None:
            continue
        if dosage == "invalid":
            invalid = True
            continue
        library_calls.append(dosage)
        individual = individuals.loc[prefix]
        if individual in calls and calls[individual] != dosage:
            conflict = True
        else:
            calls[individual] = dosage
    return {
        "n_ref_libraries": sum(value == 0.0 for value in library_calls),
        "n_alt_libraries": sum(value == 1.0 for value in library_calls),
        "n_het_libraries": sum(value == 0.5 for value in library_calls),
        "n_ref_individuals": sum(value == 0.0 for value in calls.values()),
        "n_alt_individuals": sum(value == 1.0 for value in calls.values()),
        "n_het_individuals": sum(value == 0.5 for value in calls.values()),
        "invalid_genotype": invalid,
        "replicate_genotype_conflict": conflict,
    }


def build_filter_report(
    qtl,
    allele_matrix,
    qtl_table,
    prefix_map,
    pvalue_column=None,
    available_experimental_snps=None,
    min_homozygous=3,
):
    if qtl not in P_VALUE_THRESHOLDS:
        raise ValueError(f"Unsupported QTL dataset: {qtl}")
    if "snps" not in qtl_table.columns:
        raise ValueError("QTL table must contain a 'snps' column.")

    pvalue_column = resolve_pvalue_column(qtl_table, pvalue_column)
    individuals = resolve_prefixes(allele_matrix, prefix_map)
    table = qtl_table.drop_duplicates(subset="snps", keep="first").set_index("snps")
    available = (
        None
        if available_experimental_snps is None
        else set(available_experimental_snps)
    )

    rows = []
    threshold = P_VALUE_THRESHOLDS[qtl]
    for snp in allele_matrix.columns:
        summary = genotype_summary(allele_matrix[snp], individuals)
        in_table = snp in table.index
        pvalue = (
            pd.to_numeric(pd.Series([table.at[snp, pvalue_column]]), errors="coerce").iloc[0]
            if in_table
            else np.nan
        )
        has_annotation = bool(
            in_table
            and "gene" in table.columns
            and pd.notna(table.at[snp, "gene"])
        )
        has_experimental = available is None or snp in available
        reasons = []
        if summary["invalid_genotype"]:
            reasons.append("unsupported_genotype")
        if summary["replicate_genotype_conflict"]:
            reasons.append("replicate_genotype_conflict")
        if summary["n_ref_libraries"] < min_homozygous:
            reasons.append("insufficient_homozygous_ref")
        if summary["n_alt_libraries"] < min_homozygous:
            reasons.append("insufficient_homozygous_alt")
        if not in_table:
            reasons.append("missing_qtl_table")
        elif not has_annotation:
            reasons.append("missing_qtl_annotation")
        if not np.isfinite(pvalue):
            reasons.append("missing_pvalue")
        elif not pvalue < threshold:
            reasons.append("pvalue_threshold")
        if not has_experimental:
            reasons.append("missing_experimental_tracks")

        rows.append(
            {
                "snp": snp,
                **summary,
                "biallelic": not summary["invalid_genotype"]
                and not summary["replicate_genotype_conflict"],
                "pvalue": pvalue,
                "pvalue_column": pvalue_column,
                "pvalue_threshold": threshold,
                "in_qtl_table": in_table,
                "has_qtl_annotation": has_annotation,
                "has_experimental_tracks": has_experimental,
                "eligible": not reasons,
                "exclusion_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).set_index("snp")


def eligible_snps(report):
    return list(report.index[report["eligible"]])


def write_filter_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path)
    return path


def print_filter_summary(report, qtl):
    print("\nQTL manuscript filters")
    print(f"  allele-matrix SNPs: {len(report):,}")
    biallelic = report["biallelic"]
    print(f"  after biallelic/replicate-consistency filter: {int(biallelic.sum()):,}")
    genotype_ok = (
        biallelic
        & (report["n_ref_libraries"] >= 3)
        & (report["n_alt_libraries"] >= 3)
    )
    print(f"  after >=3 homozygous libraries per allele: {int(genotype_ok.sum()):,}")
    annotation_ok = genotype_ok & report["in_qtl_table"] & report["has_qtl_annotation"]
    print(f"  after QTL annotation intersection: {int(annotation_ok.sum()):,}")
    pvalue_ok = annotation_ok & np.isfinite(report["pvalue"]) & (
        report["pvalue"] < report["pvalue_threshold"]
    )
    print(f"  after p-value threshold: {int(pvalue_ok.sum()):,}")
    experimental_ok = pvalue_ok & report["has_experimental_tracks"]
    print(f"  after experimental-track intersection: {int(experimental_ok.sum()):,}")
    print(f"  final eligible SNPs: {int(report['eligible'].sum()):,}")
    if qtl == "tiqtl" and int(report["eligible"].sum()) != 2057:
        print("  warning: eligible tiQTL count differs from manuscript value 2,057.")
