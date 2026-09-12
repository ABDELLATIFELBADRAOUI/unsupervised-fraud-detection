# Scripts of the submitted version (v1)

These three scripts are the code that produced the results of the **submitted**
manuscript (ARRAY-D-26-04414, initial submission). They are committed unmodified,
for auditability only.

| File | Role |
|---|---|
| `ccfraud_unsupervised.py` | ULB benchmark, chronological split |
| `ccfraud_unsupervised_RANDOM.py` | ULB benchmark, random stratified split |
| `ccfraud_paysim.py` | PaySim benchmark |

They are **not** the recommended code path. The defects the reviewers identified in
them — a label-derived `rho` inside a regime described as label-free, a decision
threshold taken as a quantile of the test-score distribution, `novelty=False` LOF
with no out-of-sample interface, DBSCAN clustered on the test set with a binary
score, and the mixture of `predict()` and Eq. (13) that produced the PaySim alert
volumes — are the subject of sections 2, 3 and 4bis of
`../notebooks/revision_pipeline_v2.ipynb`.

Use `notebooks/revision_pipeline_v2.ipynb` to reproduce the revised manuscript.
