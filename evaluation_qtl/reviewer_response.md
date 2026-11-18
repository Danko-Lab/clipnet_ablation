# CLIPNET QTL reviewer response

Nov 4, 2026 · last updated Nov 18, 2026

## Response to Reviewer \[X\], comment \[Y\]

We thank the reviewer for these suggestions. We have added the requested intermediate-checkpoint analyses, examined whether CLIPNET-ref benefits from longer training, and added a discussion of hyperparameter optimization as a limitation.

**Correction to the reported QTL benchmark.** While carrying out these analyses we found an error in the script that scored the per-fold CLIPNET QTL predictions. Each held-out fold *k* was scored with the average of fold models 1 through *k* rather than with model *k* alone; the CLIPNET-ref predictions were scored correctly. With corrected scoring, the CLIPNET log-L2 Pearson correlations are 0.460 (tiQTL) and 0.531 (diQTL), compared with 0.477 and 0.542 previously reported. CLIPNET-ref is unchanged at 0.372 and 0.419. We have updated \[Fig. X / Table X\] accordingly.

The conclusion is unchanged: CLIPNET outperforms CLIPNET-ref on both tiQTLs (Δr = 0.087, 95% CI 0.044–0.132) and diQTLs (Δr = 0.112, 95% CI 0.059–0.167). These confidence intervals come from a paired bootstrap over SNPs within folds.

**Robustness of the published comparison.** The log-L2 Pearson metric is sensitive to expression level and to calibration of the count head across fold models, so we added calibration-robust alternatives. With observed coverage regressed out within each fold, CLIPNET's within-fold Spearman correlation still exceeds CLIPNET-ref's on both tiQTLs (Δ = 0.073, 95% CI 0.032–0.114) and diQTLs (Δ = 0.071, 95% CI 0.017–0.123).

Two diagnostics show that CLIPNET-ref's count head is partly the cause. On three CLIPNET-ref folds (2, 4 and 6), the count head predicts zero total signal for 43–58% of QTL loci; CLIPNET does this for fewer than 0.3% of loci. Excluding these three folds widens CLIPNET's advantage (tiQTL Δr = 0.125; diQTL Δr = 0.150), so the result is not driven by these folds.

**Intermediate checkpoints.** As suggested, we retrained both models for 100 epochs without early stopping, retaining a checkpoint every 5 epochs, and evaluated ti- and diQTL performance at each one \[Supplementary Fig. SX\]. Performance varies substantially between checkpoints of the same training run. For individual CLIPNET checkpoints between epochs 5 and 100, the tiQTL log-L2 correlation ranges from 0.22 to 0.51. This variation is comparable to or larger than the difference between models, and is largely attributable to the count head intermittently predicting zero signal at low-expression loci.

Ensembles of the earliest checkpoints (epochs 5–15) are similar between the two models (tiQTL Δr = −0.007, 95% CI −0.032 to 0.018; diQTL Δr = −0.013, 95% CI −0.041 to 0.015).

These retrained comparisons rest on a single training run per model, and the bootstrap intervals do not capture run-to-run variance, so we interpret checkpoint-level differences cautiously. \[Supplementary Fig. SX\] reports all metrics.

**Longer training.** We trained both models for 100 epochs without early stopping. CLIPNET did not improve with longer training: ensembles of late checkpoints (epochs 80–100) performed no better than ensembles of early checkpoints (epochs 5–15), at 0.47 vs 0.58 for tiQTLs and 0.52 vs 0.64 for diQTLs. CLIPNET-ref got worse. Over the first 25 epochs the two models were similar, with CLIPNET-ref slightly ahead on the benchmark metric (tiQTL Δr = −0.056; diQTL Δr = −0.073). After epoch 25, CLIPNET-ref's QTL performance declined while CLIPNET's did not. For epochs 80–100, CLIPNET outperformed CLIPNET-ref on tiQTLs (Δr = 0.087, 95% CI 0.053–0.122) and diQTLs (Δr = 0.090, 95% CI 0.037–0.150). With coverage regressed out, the advantage held for tiQTLs (Δ = 0.061, 95% CI 0.025–0.096) but was not significant for diQTLs (Δ = 0.032, 95% CI −0.016 to 0.081).

Longer training also made CLIPNET-ref's count head collapse more often. Per checkpoint, it predicted zero signal at 37% of tiQTL loci on average, versus 22% for CLIPNET, and 13 of its 180 fold checkpoints predicted zero at every tiQTL locus. Test-set profile and quantity performance were similar between the models on folds where CLIPNET-ref's count head did not collapse (median track Pearson 0.588 vs 0.587). CLIPNET-ref is therefore not undertrained; its gap reflects count-head collapse, which regularization or different hyperparameters might mitigate.

**Checkpoint selection.** Because QTL performance varies so much between checkpoints, we asked how the comparison depends on which checkpoint represents each model. Represented by its best checkpoint on the QTL benchmark itself, an optimistic choice, the two models are comparable. CLIPNET's best is higher on the benchmark metric (tiQTL Δr = 0.044, 95% CI 0.016–0.080; diQTL Δr = 0.074, 95% CI 0.035–0.111). CLIPNET-ref's best is higher on the tiQTL fold-0 ensemble (Δr = −0.049, 95% CI −0.097 to −0.008).

Selecting each fold model's checkpoint by validation loss, as for the published models, restores CLIPNET's advantage (table). CLIPNET-ref's validation loss keeps improving after epoch 25 while its QTL performance declines, so validation-loss selection picks checkpoints that are poor at QTLs.

| Metric (CLIPNET − CLIPNET-ref, 95% CI) | tiQTL | diQTL |
| --- | --- | --- |
| Benchmark log-L2 Pearson | +0.054 (0.011, 0.098) | +0.046 (−0.031, 0.132) |
| Fold-0 ensemble | +0.018 (−0.039, 0.075) | +0.056 (0.017, 0.095) |
| Within-fold Spearman | +0.123 (0.088, 0.159) | +0.109 (0.058, 0.159) |
| Within-fold Spearman, coverage regressed out | +0.047 (0.008, 0.086) | +0.055 (−0.001, 0.110) |
| Folds where CLIPNET is higher | 7 of 9 | 8 of 9 |

**Hyperparameter optimization.** We agree that separately optimizing hyperparameters for CLIPNET-ref would be the fairest comparison. Doing this at the scale of a 9-fold ensemble was computationally infeasible for this revision, so we have added it as a limitation in the Discussion (below).

## Added to Discussion

A limitation of our comparison is that the architecture and training hyperparameters were optimized for CLIPNET and applied unchanged to CLIPNET-ref. On some folds, CLIPNET-ref's count head predicted zero signal for a large fraction of QTL loci, suggesting that CLIPNET-ref may benefit from different regularization or optimization settings. Although CLIPNET's advantage held under calibration-robust metrics and when these folds were excluded, we cannot rule out that a separately optimized reference-genome model would narrow the gap. Training CLIPNET-ref for 100 epochs without early stopping widened it. QTL benchmark performance also varied substantially between training checkpoints, so comparisons between individual models should be interpreted with this variability in mind.

## Figure legends

**[Fig. X], corrected.** Predicted versus observed QTL effects for the published models. Each point is a SNP; axes show log10(L2 + 10⁻³) of the difference between homozygous allele tracks, coloured by point density. SNPs held out by fold models 1–9 are predicted by the model that held them out; the remaining SNPs are predicted by the 9-model ensemble. (a) CLIPNET, tiQTLs (n = 2,057; Pearson r = 0.460). (b) CLIPNET-ref, tiQTLs (r = 0.372). (c) CLIPNET, diQTLs (n = 1,207; r = 0.531). (d) CLIPNET-ref, diQTLs (r = 0.419). Dashed line, y = x. CLIPNET-ref SNPs at −3 are loci where its count head predicts zero signal.

**[Supplementary Fig. SX]. QTL performance across training checkpoints.** CLIPNET and CLIPNET-ref were retrained for 100 epochs without early stopping, retaining a checkpoint every 5 epochs. (a, b) Composite log-L2 Pearson correlation used in the main text, for tiQTLs (a) and diQTLs (b). (c, d) Within-fold Spearman correlation after regressing out log observed coverage. Shading, 95% paired SNP-bootstrap CIs; dashed lines, the published models. (e, f) Mean fraction of held-out SNPs whose predicted L2 is exactly zero because the ReLU count head outputs zero for every individual.

## Notes before sending

Open items before this goes out. The 20-epoch retraining runs are excluded: they had not converged and do not reflect how either model is trained.

- [ ] **Scoring correction: disclosed (decided).** At acceptance, upload the corrected `qtl_analysis.tar.gz` fold tables (evaluation_qtl/nonretraining_results/corrected_qtl/) as a new Zenodo version.
- [ ] **Check CLIPNET-ref values against the manuscript.** 0.372 / 0.419 come from the paper's own ref prediction tables, which match our rerun; confirm they match the reported text.
- [ ] **Swapped axes in the published CLIPNET-ref tiQTL scatter?** plot_reference_tiqtl.ipynb plots predicted values on the x-axis but labels it "Observed" (the diQTL notebook is correct). If that panel is in the manuscript, replace it with the corrected figure.
- [ ] **Fill placeholders.** Reviewer and comment numbers, [Fig. X / Table X], and [Supplementary Fig. SX] (response text and legends).

All numbers are from `evaluation_qtl/nonretraining_results/` (branch dev).
