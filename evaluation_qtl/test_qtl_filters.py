import unittest

import numpy as np
import pandas as pd

from evaluation_qtl.qtl_filters import (
    build_filter_report,
    eligible_snps,
    resolve_pvalue_column,
)


class QTLFilterTests(unittest.TestCase):
    def setUp(self):
        self.prefixes = ["p1a", "p1b", "p2", "p3", "p4", "p5", "p6"]
        self.prefix_map = {
            "p1a": "i1",
            "p1b": "i1",
            "p2": "i2",
            "p3": "i3",
            "p4": "i4",
            "p5": "i5",
            "p6": "i6",
        }
        passing = [0, 0, 0, 0, 1, 1, 1]
        self.alleles = pd.DataFrame(
            {
                "pass": passing,
                "boundary": passing,
                "invalid": [0, 0, 0, 0, 1, 1, 2],
                "conflict": [0, 1, 0, 0, 1, 1, 1],
                "insufficient": [0, np.nan, 0, 0.5, 1, 1, 1],
                "missing_expt": passing,
            },
            index=self.prefixes,
        )
        self.table = pd.DataFrame(
            {
                "snps": list(self.alleles.columns),
                "gene": [f"chr1:{index}" for index in range(len(self.alleles.columns))],
                "pvalue": [1e-7, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7],
            }
        )

    def test_unique_individual_counts_and_filters(self):
        report = build_filter_report(
            "tiqtl",
            self.alleles,
            self.table,
            self.prefix_map,
            available_experimental_snps=set(self.alleles.columns) - {"missing_expt"},
        )
        self.assertEqual(report.at["pass", "n_ref_individuals"], 3)
        self.assertEqual(report.at["pass", "n_alt_individuals"], 3)
        self.assertEqual(report.at["pass", "n_ref_libraries"], 4)
        self.assertEqual(report.at["pass", "n_alt_libraries"], 3)
        self.assertTrue(report.at["pass", "eligible"])
        self.assertFalse(report.at["boundary", "eligible"])
        self.assertIn("pvalue_threshold", report.at["boundary", "exclusion_reasons"])
        self.assertIn("unsupported_genotype", report.at["invalid", "exclusion_reasons"])
        self.assertIn(
            "replicate_genotype_conflict",
            report.at["conflict", "exclusion_reasons"],
        )
        self.assertIn(
            "insufficient_homozygous_ref",
            report.at["insufficient", "exclusion_reasons"],
        )
        self.assertIn(
            "missing_experimental_tracks",
            report.at["missing_expt", "exclusion_reasons"],
        )
        self.assertEqual(eligible_snps(report), ["pass"])

    def test_diqtl_threshold_is_strict(self):
        table = self.table.loc[self.table["snps"].isin(["pass", "boundary"])].copy()
        table["pvalue"] = [9e-4, 1e-3]
        report = build_filter_report(
            "diqtl",
            self.alleles[["pass", "boundary"]],
            table,
            self.prefix_map,
        )
        self.assertTrue(report.at["pass", "eligible"])
        self.assertFalse(report.at["boundary", "eligible"])

    def test_pvalue_column_override_and_ambiguity(self):
        table = pd.DataFrame({"snps": ["rs1"], "pval": [0.1], "p_value": [0.1]})
        with self.assertRaises(ValueError):
            resolve_pvalue_column(table)
        self.assertEqual(resolve_pvalue_column(table, "pval"), "pval")

    def test_nan_genotypes_are_allowed(self):
        alleles = self.alleles[["pass"]].copy()
        alleles.loc["p1b", "pass"] = np.nan
        report = build_filter_report(
            "tiqtl",
            alleles,
            self.table[self.table["snps"] == "pass"],
            self.prefix_map,
        )
        self.assertTrue(report.at["pass", "eligible"])

    def test_replicate_libraries_count_toward_minimum(self):
        alleles = pd.DataFrame(
            {"replicate_pass": [0, 0, 0, 1, 1, 1]},
            index=["p1a", "p1b", "p2", "p4", "p5", "p6"],
        )
        table = pd.DataFrame(
            {
                "snps": ["replicate_pass"],
                "gene": ["chr1:1"],
                "pvalue": [1e-7],
            }
        )
        report = build_filter_report(
            "tiqtl",
            alleles,
            table,
            self.prefix_map,
        )
        self.assertEqual(report.at["replicate_pass", "n_ref_libraries"], 3)
        self.assertEqual(report.at["replicate_pass", "n_ref_individuals"], 2)
        self.assertTrue(report.at["replicate_pass", "eligible"])


if __name__ == "__main__":
    unittest.main()
