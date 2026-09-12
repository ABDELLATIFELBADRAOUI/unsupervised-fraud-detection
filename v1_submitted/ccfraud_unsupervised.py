"""
Unsupervised Anomaly Detection Benchmark for Credit Card Fraud Detection
=========================================================================
Dataset : ULB creditcard.csv (284,807 transactions, 492 fraud, 0.172%)
Methods : Isolation Forest, LOF, One-Class SVM, DBSCAN, KMeans
Protocols:
    (A) Unsupervised  - fit on full training data (labels NOT used in fitting)
    (B) Semi-supervised - fit on NORMAL-only training data (novelty detection)
Splitting:
    SPLIT_MODE = "chronological" (sort by Time; earliest 60% train, next 20%
                 validation, most recent 20% test -- IDENTICAL to the SOIC
                 cost-aware DNN paper, so test-set numbers are comparable) or
                 "stratified" (random, same 60/20/20 proportions).
    Chronological is the realistic deployment setting and exposes drift.
Ensemble:
    A rank-average of the four continuous-score detectors (IF, LOF, OC-SVM,
    KMeans) is reported per protocol. DBSCAN is excluded (binary pseudo-score).
Memory:
    LOF, OC-SVM and DBSCAN do not scale to ~170k/57k points. They are fitted
    (or clustered) on capped subsamples (LOF_TRAIN_CAP, OCSVM_TRAIN_CAP,
    DBSCAN_CAP) and parallelism is bounded (N_JOBS) to avoid RAM spikes. Lower
    these caps further if memory is still tight; raise them if you have headroom.
Eval    : Precision, Recall, F1, ROC-AUC, PR-AUC (Average Precision) + runtime

Author setup notes
-------------------
- Place creditcard.csv path in DATA_PATH below.
- V1..V28 are already PCA-decorrelated; only Amount/Time need scaling.
- PR-AUC is the headline metric under 0.172% prevalence; ROC-AUC is reported
  but is optimistic for all models at this imbalance.
- OC-SVM is O(n^2)-ish: it is fit on a capped subsample of normals (documented
  as a finding, not hidden).
- DBSCAN/KMeans are not native detectors:
      DBSCAN  -> noise points (label -1) treated as anomalies
      KMeans  -> distance to nearest centroid used as anomaly score
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # safe on headless / Windows without display
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH        = r"C:\Users\LENOVO\Desktop\doctorat\DNN fraud detection\creditcard.csv"
RANDOM_STATE     = 42
TRAIN_FRAC       = 0.60               # chronological: earliest 60% -> train
VAL_FRAC         = 0.20               # next 20% -> validation (threshold/ensemble tuning)
TEST_FRAC        = 0.20               # most recent 20% -> test (matches SOIC paper)
OCSVM_TRAIN_CAP  = 20_000             # subsample cap for OC-SVM fitting
LOF_TRAIN_CAP    = 20_000             # subsample cap for LOF fitting (memory!)
DBSCAN_CAP       = 30_000             # subsample cap for DBSCAN (memory!)
N_JOBS           = 2                  # bounded parallelism; -1 can spike RAM
CONTAMINATION    = None               # None -> set to observed train fraud rate
SPLIT_MODE       = "chronological"    # "chronological" (by Time, 60/20/20) or "stratified"
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------- #
# 1. Load + preprocess (leakage-free: scalers fit on TRAIN only)
# --------------------------------------------------------------------------- #
def load_and_prepare(path):
    df = pd.read_csv(path)
    assert "Class" in df.columns, "Expected ULB schema with a 'Class' column."

    if SPLIT_MODE == "chronological":
        # Strict temporal 60/20/20 split, matching the SOIC cost-aware DNN paper:
        # earliest 60% -> train, next 20% -> validation, most recent 20% -> test.
        # No shuffling; mirrors deployment (train on past, deploy on future).
        df = df.sort_values("Time").reset_index(drop=True)
        n = len(df)
        c1 = int(n * TRAIN_FRAC)
        c2 = int(n * (TRAIN_FRAC + VAL_FRAC))
        df_tr, df_va, df_te = df.iloc[:c1], df.iloc[c1:c2], df.iloc[c2:]
    elif SPLIT_MODE == "stratified":
        # Random stratified split into the same 60/20/20 proportions.
        from sklearn.model_selection import train_test_split as _tts
        df_tmp, df_te = _tts(df, test_size=TEST_FRAC, stratify=df["Class"],
                             random_state=RANDOM_STATE)
        val_rel = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
        df_tr, df_va = _tts(df_tmp, test_size=val_rel, stratify=df_tmp["Class"],
                            random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    def split_xy(d):
        d = d.copy()
        y = d["Class"].astype(int).values
        X = d.drop(columns=["Class"])
        return X, y

    X_train, y_train = split_xy(df_tr)
    X_val,   y_val   = split_xy(df_va)
    X_test,  y_test  = split_xy(df_te)

    # Engineer hour-of-day from Time (seconds since first tx)
    for part in (X_train, X_val, X_test):
        part["Hour"] = (part["Time"] // 3600) % 24

    # Fit scalers on TRAIN ONLY (leakage-free), apply to val and test
    rob = RobustScaler()      # Amount is heavy-tailed
    std = StandardScaler()    # Time / Hour

    X_train["Amount"] = rob.fit_transform(X_train[["Amount"]])
    X_val["Amount"]   = rob.transform(X_val[["Amount"]])
    X_test["Amount"]  = rob.transform(X_test[["Amount"]])

    X_train[["Time", "Hour"]] = std.fit_transform(X_train[["Time", "Hour"]])
    X_val[["Time", "Hour"]]   = std.transform(X_val[["Time", "Hour"]])
    X_test[["Time", "Hour"]]  = std.transform(X_test[["Time", "Hour"]])

    # V1..V28 already standardized by PCA; leave as-is
    return (X_train.values, y_train,
            X_val.values,   y_val,
            X_test.values,  y_test)


# --------------------------------------------------------------------------- #
# 2. Metric helper
# --------------------------------------------------------------------------- #
def evaluate(name, protocol, y_true, y_pred, scores, fit_time):
    """scores: higher = more anomalous (used for ROC/PR-AUC). May be None."""
    row = {
        "Model": name,
        "Protocol": protocol,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": (roc_auc_score(y_true, scores) if scores is not None else np.nan),
        "PR_AUC_AP": (average_precision_score(y_true, scores)
                      if scores is not None else np.nan),
        "Fit_time_s": round(fit_time, 2),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
    return row


def threshold_by_contamination(scores, contamination):
    """Flag the top-`contamination` fraction of scores as anomalies (1)."""
    k = max(1, int(np.ceil(contamination * len(scores))))
    cutoff = np.partition(scores, -k)[-k]
    return (scores >= cutoff).astype(int)


def rank_average(score_dict):
    """Combine several anomaly-score vectors into one ensemble score.

    Each member's scores are converted to ranks (so heterogeneous scales do
    not dominate), normalised to [0, 1], then averaged across members. Higher
    ensemble score = more anomalous. `score_dict` maps member name -> score
    vector (all same length, higher = more anomalous).
    """
    from scipy.stats import rankdata
    n_members = len(score_dict)
    stacked = np.zeros(len(next(iter(score_dict.values()))))
    for sc in score_dict.values():
        r = rankdata(sc, method="average")     # ties share the mean rank
        stacked += (r - 1) / (len(r) - 1)       # normalise to [0, 1]
    return stacked / n_members


# --------------------------------------------------------------------------- #
# 3. Model runners (each returns y_pred and anomaly scores on the test set)
# --------------------------------------------------------------------------- #
def run_isolation_forest(Xtr, Xte, contam):
    m = IsolationForest(n_estimators=200, contamination=contam,
                        random_state=RANDOM_STATE, n_jobs=N_JOBS)
    t = time.time(); m.fit(Xtr); ft = time.time() - t
    scores = -m.score_samples(Xte)            # higher = more anomalous
    y_pred = (m.predict(Xte) == -1).astype(int)
    return y_pred, scores, ft


def run_lof(Xtr, Xte, contam, novelty):
    # LOF builds a neighbour graph that can exhaust RAM on large inputs.
    # In BOTH modes we now fit on a capped reference set and score the full
    # test set via novelty=True. This keeps memory bounded and, importantly,
    # yields one score per test point so the ensemble alignment holds.
    if novelty:
        ref = Xtr                          # semi-supervised: normals only
    else:
        ref = Xtr                          # unsupervised: full (contaminated) train
    if len(ref) > LOF_TRAIN_CAP:
        idx = np.random.choice(len(ref), LOF_TRAIN_CAP, replace=False)
        ref = ref[idx]
    m = LocalOutlierFactor(n_neighbors=20, contamination=contam,
                           novelty=True, n_jobs=N_JOBS)
    t = time.time(); m.fit(ref); ft = time.time() - t
    scores = -m.score_samples(Xte)
    y_pred = (m.predict(Xte) == -1).astype(int)
    return y_pred, scores, ft


def run_ocsvm(Xtr, Xte, contam):
    # Cap training size; OC-SVM is ~O(n^2)
    if len(Xtr) > OCSVM_TRAIN_CAP:
        idx = np.random.choice(len(Xtr), OCSVM_TRAIN_CAP, replace=False)
        Xfit = Xtr[idx]
    else:
        Xfit = Xtr
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=max(1e-4, float(contam)))
    t = time.time(); m.fit(Xfit); ft = time.time() - t
    scores = -m.decision_function(Xte)
    y_pred = (m.predict(Xte) == -1).astype(int)
    return y_pred, scores, ft


def estimate_eps(X, min_samples=10, sample_cap=20_000, plot_path="dbscan_kdistance.png"):
    """Estimate DBSCAN eps from the knee of the sorted k-distance curve.

    Computes the distance to the min_samples-th nearest neighbour for each
    (subsampled) point, sorts ascending, and picks the point of maximum
    curvature (the 'knee') via the largest second-difference. Saves a
    k-distance plot for the paper's appendix.
    """
    if len(X) > sample_cap:
        idx = np.random.choice(len(X), sample_cap, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=N_JOBS)
    nn.fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    kdist = np.sort(dist[:, -1])

    # Knee = index of maximum second difference (discrete curvature)
    if len(kdist) > 4:
        d2 = np.diff(kdist, 2)
        knee = int(np.argmax(d2)) + 1
    else:
        knee = len(kdist) // 2
    eps = float(kdist[knee])

    plt.figure(figsize=(6, 4))
    plt.plot(kdist, lw=1.5)
    plt.axhline(eps, color="red", ls="--", lw=1, label=f"eps = {eps:.3f}")
    plt.axvline(knee, color="grey", ls=":", lw=1)
    plt.xlabel(f"Points sorted by {min_samples}-NN distance")
    plt.ylabel(f"{min_samples}-NN distance")
    plt.title("DBSCAN k-distance knee")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path, dpi=150); plt.close()
    return eps


def run_dbscan(Xte, contam, eps):
    # DBSCAN on the full test set can exhaust RAM. If the test set exceeds the
    # cap, we cluster on a random subsample, then propagate the noise/cluster
    # status to every test point via its nearest neighbour in the subsample.
    # Noise (-1) = anomaly. DBSCAN gives no continuous score (binary pseudo-score).
    t = time.time()
    if len(Xte) > DBSCAN_CAP:
        idx = np.random.choice(len(Xte), DBSCAN_CAP, replace=False)
        Xsub = Xte[idx]
        labels_sub = DBSCAN(eps=eps, min_samples=10, n_jobs=N_JOBS).fit_predict(Xsub)
        noise_sub = (labels_sub == -1).astype(int)
        # Propagate: each test point inherits the noise flag of its nearest
        # neighbour among the clustered subsample.
        nn = NearestNeighbors(n_neighbors=1, n_jobs=N_JOBS).fit(Xsub)
        _, nbr = nn.kneighbors(Xte)
        y_pred = noise_sub[nbr[:, 0]]
    else:
        labels = DBSCAN(eps=eps, min_samples=10, n_jobs=N_JOBS).fit_predict(Xte)
        y_pred = (labels == -1).astype(int)
    ft = time.time() - t
    scores = y_pred.astype(float)
    return y_pred, scores, ft


def run_kmeans(Xtr, Xte, contam, k=8):
    m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    t = time.time(); m.fit(Xtr); ft = time.time() - t
    dists = m.transform(Xte)
    scores = dists.min(axis=1)                 # distance to nearest centroid
    y_pred = threshold_by_contamination(scores, contam)
    return y_pred, scores, ft


# --------------------------------------------------------------------------- #
# 4. Plotting
# --------------------------------------------------------------------------- #
def plot_curves(y_true, curves):
    """Overlay PR and ROC curves for all score-producing configs."""
    # Precision-Recall
    plt.figure(figsize=(7, 5))
    base = float(np.mean(y_true))
    for (name, proto), sc in curves.items():
        p, r, _ = precision_recall_curve(y_true, sc)
        ap = average_precision_score(y_true, sc)
        plt.plot(r, p, lw=1.4, label=f"{name} ({proto}) AP={ap:.3f}")
    plt.axhline(base, color="grey", ls="--", lw=1, label=f"baseline={base:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(fontsize=7, loc="upper right"); plt.tight_layout()
    plt.savefig("pr_curves.png", dpi=150); plt.close()

    # ROC
    plt.figure(figsize=(7, 5))
    for (name, proto), sc in curves.items():
        fpr, tpr, _ = roc_curve(y_true, sc)
        auc = roc_auc_score(y_true, sc)
        plt.plot(fpr, tpr, lw=1.4, label=f"{name} ({proto}) AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150); plt.close()


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    print("Loading and preprocessing...")
    Xtr, ytr, Xva, yva, Xte, yte = load_and_prepare(DATA_PATH)
    contam = CONTAMINATION or float(ytr.mean())
    print(f"  Train: {Xtr.shape} ({int(ytr.sum())} fraud)")
    print(f"  Val:   {Xva.shape} ({int(yva.sum())} fraud)")
    print(f"  Test:  {Xte.shape} ({int(yte.sum())} fraud)")
    print(f"  Fraud rate (train): {contam:.5f}\n")

    # Normal-only training set for the semi-supervised protocol
    Xtr_norm = Xtr[ytr == 0]

    # Estimate DBSCAN eps from the k-distance knee on the test features
    print("Estimating DBSCAN eps (k-distance knee)...")
    eps = estimate_eps(Xte, min_samples=10)
    print(f"  eps = {eps:.4f}  (plot -> dbscan_kdistance.png)\n")

    results = []
    curves = {}   # (model, protocol) -> scores, for PR/ROC plotting

    # ---- Protocol A: Unsupervised (fit on full train) -------------------- #
    print("Protocol A - Unsupervised")
    yp, sc, ft = run_isolation_forest(Xtr, Xte, contam)
    results.append(evaluate("IsolationForest", "Unsupervised", yte, yp, sc, ft))
    curves[("IsolationForest", "Unsupervised")] = sc

    yp, sc, ft = run_lof(Xtr, Xte, contam, novelty=False)
    results.append(evaluate("LOF", "Unsupervised", yte, yp, sc, ft))
    curves[("LOF", "Unsupervised")] = sc

    yp, sc, ft = run_ocsvm(Xtr, Xte, contam)
    results.append(evaluate("OneClassSVM", "Unsupervised", yte, yp, sc, ft))
    curves[("OneClassSVM", "Unsupervised")] = sc

    yp, sc, ft = run_dbscan(Xte, contam, eps)
    results.append(evaluate("DBSCAN", "Unsupervised", yte, yp, sc, ft))

    yp, sc, ft = run_kmeans(Xtr, Xte, contam)
    results.append(evaluate("KMeans", "Unsupervised", yte, yp, sc, ft))
    curves[("KMeans", "Unsupervised")] = sc

    # ---- Protocol B: Semi-supervised (fit on normals only) --------------- #
    print("Protocol B - Semi-supervised (fit on normals only)")
    yp, sc, ft = run_isolation_forest(Xtr_norm, Xte, contam)
    results.append(evaluate("IsolationForest", "Semi-supervised", yte, yp, sc, ft))
    curves[("IsolationForest", "Semi-supervised")] = sc

    yp, sc, ft = run_lof(Xtr_norm, Xte, contam, novelty=True)
    results.append(evaluate("LOF", "Semi-supervised", yte, yp, sc, ft))
    curves[("LOF", "Semi-supervised")] = sc

    yp, sc, ft = run_ocsvm(Xtr_norm, Xte, contam)
    results.append(evaluate("OneClassSVM", "Semi-supervised", yte, yp, sc, ft))
    curves[("OneClassSVM", "Semi-supervised")] = sc

    yp, sc, ft = run_kmeans(Xtr_norm, Xte, contam)
    results.append(evaluate("KMeans", "Semi-supervised", yte, yp, sc, ft))
    curves[("KMeans", "Semi-supervised")] = sc
    # DBSCAN excluded from B (transductive; no separate fit on normals)

    # ---- Ensemble (rank-average of continuous-score detectors) ----------- #
    # DBSCAN is excluded: its binary pseudo-score has no meaningful ranking.
    print("Ensembles (rank-average)")
    for proto in ("Unsupervised", "Semi-supervised"):
        members = {name: sc for (name, p), sc in curves.items() if p == proto}
        if len(members) < 2:
            continue
        t0 = time.time()
        ens = rank_average(members)
        ens_t = time.time() - t0
        yp = threshold_by_contamination(ens, contam)
        results.append(evaluate("Ensemble", proto, yte, yp, ens, ens_t))
        curves[("Ensemble", proto)] = ens

    df = pd.DataFrame(results)
    cols = ["Model", "Protocol", "Precision", "Recall", "F1",
            "ROC_AUC", "PR_AUC_AP", "Fit_time_s", "TN", "FP", "FN", "TP"]
    df = df[cols].round(4)

    print("\n=== Benchmark results ===")
    print(df.to_string(index=False))
    df.to_csv("benchmark_results.csv", index=False)
    print("\nSaved -> benchmark_results.csv")

    # ---- PR and ROC curves (one figure each, both protocols overlaid) ---- #
    plot_curves(yte, curves)
    print("Saved -> pr_curves.png, roc_curves.png")


if __name__ == "__main__":
    main()
