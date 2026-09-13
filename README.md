# Unsupervised and Semi-Supervised Anomaly Detection for Credit Card Fraud

Reproduction package for the manuscript *Unsupervised and Semi-Supervised Anomaly
Detection for Credit Card Fraud: A Chronological Benchmark and a Hybrid Bridge to
Supervised Learning* (Array, manuscript ARRAY-D-26-04414).

Every table and figure in the paper is produced by `revision_pipeline_v2.ipynb`, and the
outputs of that run are committed under `results/`, so the tables can be checked without
re-running anything.

`results/canonical_results.csv` is the canonical frame: 7,372 rows over 532 `run_id`s,
covering the headline run, the rolling origins, the contamination sweep and the split
comparison. Every value of those four experiments is addressed there by its `run_id`. The
remaining analyses — aggregation schemes, PaySim ablation, label efficiency, block
bootstrap and operational measurements — are written by their own cells to their own
`results/table_*.csv`. The formatted views used in the manuscript are rebuilt from those
files by `scripts/make_manuscript_tables.py`; no table is typed by hand.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726142.svg)](https://doi.org/10.5281/zenodo.22726142)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Archived on Zenodo: **[10.5281/zenodo.22726142](https://doi.org/10.5281/zenodo.22726142)** (concept DOI, always resolves to the latest version).

---

## Quick start

```bash
git clone <this repository>
cd <this repository>
pip install -r requirements.txt
```

Download the two datasets (links below), place them anywhere, and edit the two path
variables at the top of cell 0:

```python
ULB_PATH    = r"/path/to/creditcard.csv"
PAYSIM_PATH = r"/path/to/paysim.csv"
```

Then run the notebook top to bottom. Cell 0 verifies that both files exist and
reports their size and header before anything else runs.

Expect several hours on a laptop. The PaySim experiments dominate; the ULB-only
cells complete in minutes.

---

## Data

Neither dataset is redistributed here. Both are public.

| Dataset | Source | Transactions | Frauds |
|---|---|---|---|
| ULB credit card | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 284,807 | 492 |
| PaySim | https://www.kaggle.com/datasets/ealaxi/paysim1 | 6,362,620 | 8,213 |

The ULB chronological partition used here (170,884 / 56,961 / 56,962, sorted by
`Time`) is identical to the one in our companion supervised study, which makes the
supervised reference points of Section 4.5 directly comparable.

---

## What the notebook enforces

The pipeline is organised around five invariants, checked at run time rather than
stated in prose. They exist because the first version of this work violated three
of them.

**1. No test-set object reaches a calibration function.** Detectors return scores
only; the `Scores` dataclass has no prediction field. Hard decisions are produced
exclusively by `ValidationCalibrator`, which is constructed from validation scores.
A probe verifies that scoring a transaction in isolation and scoring it inside the
full test batch yield identical decisions, and records the outcome in the results
frame.

**2. The contamination parameter is never derived from labels**, except in the
regime explicitly named `O` (oracle), which is reported as an unreachable upper
bound. Every row carries its regime, so an oracle result can never be mistaken for
a label-free one.

**3. One canonical results frame.** Tables and figures are generated from it, and
the plotting code asserts that each value drawn equals the corresponding tabulated
value before writing the file.

**4. One sign convention**: higher score means more anomalous, asserted for every
detector.

**5. Traceability**: every metric row carries a `run_id` hashed from the full
configuration.

---

## Notebook layout

| Cell | Contents |
|---|---|
| 0 | Configuration, preflight file check, run manifest, stale-results guard |
| 1 | Data loading, chronological / random / prevalence-matched splits, rolling origins |
| 2 | Detectors, with explicit out-of-sample scoring for LOF and DBSCAN |
| 3 | Validation calibrator, metrics, bootstrap helpers, canonical results store |
| 4 | Headline run: three regimes, five detectors, ensemble |
| 5 | Forensic reconstruction of the three candidate thresholding rules |
| 6 | Rolling-origin evaluation |
| 7 | Prevalence-controlled random versus chronological comparison |
| 8 | Sensitivity to the assumed contamination rate |
| 9 | Hyperparameter grids and repeated subsampling under both selection rules |
| 10 | Aggregation schemes compared |
| 11 | PaySim feature ablation and raw-residual probe |
| 12 | Label-efficiency curve for the hybrid bridge |
| 13 | Block bootstrap intervals and paired comparisons |
| 14 | Operational measurement: latency, memory, alert budget, scaling |
| 15 | Tables and figures from the canonical frame |
| 16 | Confusion matrices and the decomposed split-gap chart |

`scripts/make_manuscript_tables.py` rebuilds `results/manuscript_tables/` and the
`results/table_split_gap_*.csv` files from the canonical frame and the result tables. Run it
after the notebook, or on the committed `results/` directory as it stands.

Cells 5 to 12 are independent of one another and can be run selectively, provided
cells 0 to 4 have been executed. Cell 14 needs cell 13, and cell 15 needs cells 12
and 13; a `require()` guard raises a clear message if the order is wrong.

---

## Two implementation points worth reading

**Out-of-sample LOF.** `novelty=True` is used in all regimes. With `novelty=False`,
scikit-learn's LOF does not expose an out-of-sample scoring interface, so it cannot
produce scores on a held-out test set; the first version of this work documented
`novelty=False` for one protocol, which was a documentation error.

**Out-of-sample DBSCAN.** DBSCAN has no native `predict`. The rule used here is
defined explicitly: a capped subsample of the training reference set is clustered,
the core points are collected, and for a new point *z* the score is the Euclidean
distance to the nearest training core point, with *z* labelled noise when that
distance exceeds ε. The score is continuous, computed per point, and independent of
the test set. This is what the manuscript previously called "neighbour propagation"
without defining it.

---

## Outputs

Running the notebook writes to `revision_v2/`:

```
manifest.json              versions, seeds, grids, caps
canonical_results.csv      one row per (run_id, metric)
test_scores.npz            raw test scores per configuration (83 MB, not committed here;
                           cell 13 regenerates it)
table_*.csv                individual result tables
manuscript_tables/         the same tables formatted for the paper
figures/                   PR, ROC, rolling origin, label efficiency,
                           confusion matrices, split-gap decomposition
```

`save_results()` merges with what is already on disk on the key
`(run_id, metric)`, so a partial re-run after a kernel restart replaces only its
own rows. If you change a configuration value such as the contamination grid, the
guard in cell 0 will stop and ask you to archive the previous output folder first,
because new configurations mint new `run_id`s and the two generations would
otherwise coexist silently.

---

## Reproducibility notes

Results were produced with Python 3.11, numpy 1.26.4, pandas 2.2.2, scikit-learn
1.5.0, scipy 1.13.1, on Windows 11 with an Intel Core i5-1235U and 16 GB of RAM.
`requirements.txt` pins these versions.

Seeds are fixed at 42. Two sources of variation remain and are reported rather than
suppressed:

- **Training subsample.** LOF, One-Class SVM and DBSCAN are fitted on a capped
  subsample of 20,000 transactions. Cell 9 repeats this over five seeds and under
  two selection rules. The rule matters more than the seed: on ULB under the
  normal-only regime, LOF attains 0.5572 under uniform sampling and 0.0036 under a
  contiguous most-recent block.
- **Rolling origins.** The detector ordering is not stable across temporal origins.
  Any single chronological cut, including the headline one, should be read with
  that in mind.

---

## Citation

If you use this code, please cite the paper. See `CITATION.cff`.

---

## License

MIT. See `LICENSE`. The datasets are governed by their own licences at the sources
listed above.
