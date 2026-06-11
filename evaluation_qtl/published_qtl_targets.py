"""Load canonical observed QTL L2 values from the CLIPNET paper archive."""

import gzip
import io
import tarfile

import pandas as pd


ARCHIVE_MEMBERS = {
    "diqtl": "qtl_analysis/diqtls_ensemble_l2_scores.csv.gz",
    "tiqtl": "qtl_analysis/tiqtls_ensemble_l2_scores.csv.gz",
}


def load_published_experimental_l2(archive_path, qtl):
    if qtl not in ARCHIVE_MEMBERS:
        raise ValueError(f"Unsupported QTL dataset: {qtl}")
    with tarfile.open(archive_path, "r:gz") as archive:
        member_name = ARCHIVE_MEMBERS[qtl]
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(
                f"{member_name} is not present in {archive_path}."
            )
        contents = gzip.decompress(member.read())
    scores = pd.read_csv(io.BytesIO(contents), index_col=0)
    if scores.index.has_duplicates:
        raise ValueError(f"Published QTL targets contain duplicate SNPs in {archive_path}.")
    if "expt" not in scores.columns:
        raise ValueError(f"Published QTL targets in {archive_path} lack an expt column.")
    return scores["expt"].rename("expt")


def experimental_l2_targets(args):
    archive_path = getattr(args, "experimental_l2_archive", None)
    if archive_path is None:
        return None
    cache_key = (str(archive_path), args.qtl)
    if getattr(args, "_experimental_l2_cache_key", None) != cache_key:
        args._experimental_l2_targets = load_published_experimental_l2(
            archive_path, args.qtl
        )
        args._experimental_l2_cache_key = cache_key
    return args._experimental_l2_targets


def experimental_source(args):
    return (
        "published_archive"
        if getattr(args, "experimental_l2_archive", None) is not None
        else "generated_procap_tracks"
    )


def score_directory_name(args):
    return (
        "scores_published_l2"
        if getattr(args, "experimental_l2_archive", None) is not None
        else "scores"
    )


def summary_suffix(args):
    return (
        "_published_l2"
        if getattr(args, "experimental_l2_archive", None) is not None
        else ""
    )
