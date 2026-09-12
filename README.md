# Unsupervised and Semi-Supervised Anomaly Detection for Credit Card Fraud — Reproduction Package

Code accompanying the manuscript

> **Unsupervised and Semi-Supervised Anomaly Detection for Credit Card Fraud:
> A Chronological Benchmark and a Hybrid Bridge to Supervised Learning**
>
> Abdellatif Elbadraoui, Yassine Mouhssine, Abdelkader El Alaoui, Said Ouatik El Alaoui
>
> *Array* (Elsevier) — manuscript ARRAY-D-26-04414, under revision

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726142.svg)](https://doi.org/10.5281/zenodo.22726142)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Archived on Zenodo: **[10.5281/zenodo.22726142](https://doi.org/10.5281/zenodo.22726142)**
(concept DOI, always resolves to the latest version;
release `v1.0.0` is [10.5281/zenodo.22726143](https://doi.org/10.5281/zenodo.22726143)).

This repository contains the **revision pipeline (v2)**, which is the implementation
actually run for the revised manuscript, together with the **scripts used for the
original submission (v1)** so that every reported change is auditable.

---

## 1. What this package is for

The reviewers asked, among other things, for the out-of-sample scoring procedure of
LOF and DBSCAN to be stated precisely and for the implementation to be released.
`notebooks/revision_pipeline_v2.ipynb` is that implementation. It is organised so
that the structural requirements are enforced by assertions rather than by
convention:

| Invariant | Statement |
|---|---|
| 1 | No test-set object ever reaches a calibration function. |
| 2 | `rho` is never computed from labels, except in the explicitly labelled ORACLE regime. |
| 3 | One canonical results frame; every table and figure derives from it, and their agreement is asserted. |
| 4 | One sign convention — higher score = more anomalous — asserted for every detector. |
| 5 | Every number is traceable to a row of the results frame via `run_id`. |

The two fitting "protocols" of the submitted version are replaced by three named
regimes, so that no label-derived quantity hides inside a label-free claim:

| Regime | Reference set | `rho` |
|---|---|---|
| **U** — label-free | full training split | a-priori grid (default 0.005), never from labels |
| **N** — normal-only | training rows labelled legitimate | a-priori grid |
| **O** — oracle | full training split | true training prevalence — reported as an upper bound, not deployable |

---

## 2. Out-of-sample scoring of LOF and DBSCAN

These are the two points where the submitted version did not state a usable
interface. Both are now explicit (`notebooks/revision_pipeline_v2.ipynb`, section 2).

**LOF.** `novelty=True` in *all* regimes. With `novelty=False`, scikit-learn's LOF
scores only the fitted observations and exposes no method for unseen data, so it
cannot produce the held-out test scores the manuscript reports. The reference set is
a capped random subsample of the training split (`LOF_TRAIN_CAP = 20000`); test
scores are `-score_samples(X_test)`.

**DBSCAN.** DBSCAN has no native `predict`. The out-of-sample rule is defined as a
distance to the training clustering:

1. cluster a capped subsample of the training reference set (`DBSCAN_CAP = 20000`);
2. collect the core points `C = { x : |N_eps(x)| >= min_samples }`;
3. for a new point `z`, set `s(z) = dist(z, nearest core point of C)`, and
   `z` is noise iff `s(z) > eps`.

`s(z)` is continuous, computed per point, and depends on nothing but the training
clustering — so DBSCAN becomes rankable and ensemble-eligible without any
test-set dependence. `eps` is selected by a k-distance knee computed **on the
training reference set** (the submitted version used the test set).

---

## 3. Where each reviewer point is addressed

Section headers in the notebook carry `[Rx#n]` tags; the mapping is:

| Notebook section | Reviewer points |
|---|---|
| 1. Data, splits, preprocessing | R1#4, R4#4, R4#8, R4#9, R6#3 |
| 2. Detectors — `fit(train) -> score(val), score(test)` | R4#1, R4#2, **R4#3** |
| 3. Calibration, metrics, canonical results store | R1#2, R1#9, R4#2, R5#2, R6#2, R6#3 |
| 4. Main runner — one split, three regimes, five detectors + ensemble | — |
| 4bis. Forensic reconstruction of the v1 alert volumes | R6#2 |
| 5. Rolling-origin evaluation | R1#4, R4#4, R6#3 |
| 6. Prevalence-controlled random vs chronological | R4#4 |
| 7. Sensitivity to `rho` | R1#1, R6#4 |
| 8. Hyperparameter sensitivity (selection on validation) | R1#3, R4#7, R4#8, R5#3 |
| 8bis. Aggregation schemes | R1#5 |
| 8ter. PaySim feature ablation | R4#9 |
| 9. Label-efficiency curve (the hybrid bridge) | R1#6, R4#6, R6#5 |
| 10. Uncertainty — block bootstrap over the test stream | R5#2, R6#3 |
| 11. Operational measurement — inference, not fitting | R4#10, R5#4 |
| 12. Tables and figures — all derived from the canonical frame | R4#5 |

Thresholds are fitted on the **validation** split and applied unchanged to the test
stream; the transductive rule of the submitted Eq. (13) is not used for the headline
numbers. Section 4bis reconstructs, side by side, the alert volumes produced by
`predict()`, by the submitted Eq. (13), and by the validation rule — which is how the
960,000 false positives reported for PaySim in the submitted version are accounted
for.

---

## 4. Data

Neither dataset is redistributed here. Both are public:

| Dataset | Source | Size | Fraud rate |
|---|---|---|---|
| ULB `creditcard.csv` | <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> | 284,807 tx | 0.172 % |
| PaySim | <https://www.kaggle.com/datasets/ealaxi/paysim1> | 6,362,620 tx | 0.129 % |

Download both, then edit the two path constants at the top of the notebook
(`ULB_PATH`, `PAYSIM_PATH`). Nothing else needs to be changed.

---

## 5. Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/revision_pipeline_v2.ipynb
```

Run the cells in order. Section 1 must run before any later section: the detectors,
the calibration module and every experiment read the split objects it builds. Outputs
are written to `revision_v2/` (results frame `canonical_results.csv`, figures in
`revision_v2/figures/`); `SEED = 42` throughout.

**Environment of the reference run** (written to the run manifest by cell 1, run of
2026-09-12):

| | |
|---|---|
| Python | 3.11.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.0 |
| scipy | 1.13.1 |

PaySim is 6.36 M rows: the full notebook is hours of compute on a laptop, dominated by
One-Class SVM and the bootstrap. Detector reference sets are capped at 20,000 rows
(LOF, OC-SVM, DBSCAN) — the caps are configuration constants at the top of the
notebook, reported in the manuscript, and swept in the sensitivity section.

---

## 6. What is in this repository, and what is not

* `notebooks/revision_pipeline_v2.ipynb` — the revision pipeline. Committed **as
  executed**, so the stored outputs of sections 1–4bis are the ones produced by the
  reference run. SHA-256 of the committed file:
  `c6d378414fcf66cd0f8fb1c9e63f21fc451d8f760e1151d62e82d4382004c752`.
* `v1_submitted/` — the three scripts that produced the numbers of the *submitted*
  version, unmodified. They are included for auditability, not as the recommended
  code path; their known defects are the subject of sections 2, 3 and 4bis of the
  notebook.
* No datasets, and no results files: `canonical_results.csv`, the tables and the
  figures are regenerated by running the notebook. Every number in the revised
  manuscript is a row of that frame, addressed by `run_id`.

---

## 7. Citation

See `CITATION.cff`. Please cite the manuscript, and this archive by its DOI
(10.5281/zenodo.22726142 for all versions, 10.5281/zenodo.22726143 for `v1.0.0`).

## 8. License

MIT — see `LICENSE`. The datasets are governed by their own licenses at the sources
listed in section 4.
