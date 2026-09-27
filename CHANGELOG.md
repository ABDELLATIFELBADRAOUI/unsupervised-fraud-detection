# Changelog

## [1.3.0] — 2026-09-27

Accompanies the revised manuscript ARRAY-D-26-04414, now titled *Rethinking Unsupervised
Credit Card Fraud Detection: A Leakage-Controlled Chronological Benchmark*.

### Added

- `scripts/clock_ablation/ONE_CELL_clock_ablation.py` and
  `scripts/clock_ablation/CELL_2_paysim_no_step_no_residuals.py`: the ablation of the absolute
  clocks (`Time` on ULB, `step` on PaySim) and the clock × balance-residual design of Table 13.
  They run after `notebooks/revision_pipeline_v2_1.ipynb`, append rows to
  `canonical_results.csv` under `clock_ablation`, `clock_ablation_ref` (the ULB run with the
  clock, which reproduces the headline values; `table_clock_reproduction_check.csv`) and
  `clock_balance_ablation`, and write `table_clock_ablation.csv`, `table_clock_balance_2x2.csv`,
  `table_clock_position_corr.csv`, `table_clock_position_baseline.csv`,
  `table_clock_reproduction_check.csv` and `manifest_clock_ablation.json`. Headline rows are
  not modified.
- `results/canonical_results.csv` now holds 8,188 rows over 628 `run_id`s (7,372 over 532 in
  1.2.0).
- `scripts/select_hyperparameters.py`: rebuilds `results/phase0/table16_hyperparameter_selected.csv`
  and `results/phase0/table3_selected_block.csv` from `results/table_hyperparameter_sensitivity.csv`.
- `scripts/figure1_framework.tex` and `results/figures/figure1_framework.pdf`: Figure 1.

### Fixed

- **Tie-break on validation PR-AUC.** On the 4-decimal values, the comparison
  `val_PR_AUC >= top - 1e-4` excluded one tie through floating-point rounding (PaySim,
  label-free, Isolation Forest: 0.0022 and 0.0021). The comparison now subtracts 1e-9 in the
  three places it appears. The selected configuration of that cell becomes 100 trees
  (validation 0.0021, test 0.0618); no other selection changes. The stored outputs of those
  notebook cells predate the fix; the phase-0 files are regenerated.
- `v1_submitted/README.md`: the submitted code always fitted LOF with `novelty=True`; the
  defect was the `novelty=False` entry of the submitted Table 3.

### Notes

- `test_scores.npz`, the saved headline test scores (83 MB), from which the 200,000-row window
  comparison of Table 18 and the position statistics of Section 4.7 can be recomputed, is not
  tracked by Git (see `.gitignore`); it is archived as a separate file in the Zenodo record.
- Section 8b of the notebook (the ensemble rebuilt at the validation-selected configurations)
  is not used in the manuscript, and its output is not archived.

## [1.2.0] — 2026-09-18

Correction release. `notebooks/revision_pipeline_v2_1.ipynb` **reproduces the archived
results with its default settings**, so every number in the revised manuscript remains
traceable to this repository. `notebooks/revision_pipeline_v2.ipynb` is retained
unchanged: it is the run that produced the submitted tables.

### Fixed

- **DBSCAN silently replaced the core set with the whole reference sample.** The line
  `core = ref[db.core_sample_indices_] if len(db.core_sample_indices_) else ref` turned
  `s(z)` into a 1-NN distance to the training subsample whenever the clustering
  produced no core point, which is not DBSCAN. The substitution is now recorded
  (`n_core`, `core_fraction`, `degenerate_core`, `core_fallback`) and raises a
  `RuntimeWarning`. On PaySim three of five reference draws yield a single core point
  and the other two fall back; on ULB the core set holds 3 to 46 points.
- **Two hyperparameter selection rules coexisted and disagreed.** The sweep cell used a
  bare argmax on validation PR-AUC while the phase-0 cell used a documented tie-break;
  they named different members of a tied set in four cells. The sweep now uses the
  phase-0 rule verbatim, so the notebook, the run manifest and the manuscript name the
  same member.
- **The single-point invariance probe covered the ensemble only.** It now runs for all
  six configurations, which is what the manuscript claims.

### Added

- `results/phase0/` — the validation-selected hyperparameter configurations with their
  tie sets, the paired bootstrap comparisons of the ensemble against each member, and
  the per-fold rolling-origin values. These back Tables 9, 11 and 18 of the revised
  manuscript and were previously missing from the archive.
- `REQUIRE_NONDEGENERATE_CORE` (default `False`) and `MIN_CORE_POINTS`: widen `eps`
  until the DBSCAN core set is usable. Off by default so the notebook matches the
  archived results; switching it on changes every table with a DBSCAN column.
- Section 8b: refits each ensemble member at its validation-selected configuration and
  rebuilds the ECDF ensemble on top, so tuned detectors and the ensemble can be ranked
  at a single setting rather than across two tables.
- The ensemble row in the operational table: latency, peak memory and alert volume,
  computed as the sum of its members plus five ECDF lookups.
- `pipeline_version`, `min_core_points`, `require_nondegenerate_core` and
  `selection_tie_break` in the run manifest.

### Changed

- Manuscript title updated in `README.md`, `CITATION.cff` and `.zenodo.json` to
  *A Chronological, Leakage-Controlled Benchmark*, and the protocol is no longer
  described as "leakage-free" — a claim the revision itself withdrew.

SHA-256 of `notebooks/revision_pipeline_v2_1.ipynb`:
`ad0955b0a8676da7cfc0d459d9a996bd2c6ad63ada8323a59123b601da6646da`

## [1.1.0] — 2026-09-13

Results directory committed, metadata completed.

## [1.0.0]

Initial archived release.
