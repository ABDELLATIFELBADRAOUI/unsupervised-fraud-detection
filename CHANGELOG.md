# Changelog

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
