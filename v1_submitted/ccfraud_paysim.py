"""
PaySim Unsupervised Fraud Detection Benchmark
==============================================
Second dataset validation of the ULB creditcard.csv benchmark.
Same five detectors, same two protocols, same ensemble, same metrics.
Chronological 60/20/20 split on `step` (1 step = 1 hour, 743 steps total).

PaySim specifics:
  - 6,354,407 transactions, ~8,213 fraud (isFraud=1), prevalence ~0.13%
  - Features: step, type (categorical), amount, 4 balance columns
  - No PCA components -- raw financial features
  - Fraud occurs ONLY in TRANSFER and CASH_OUT transactions

Path: update DATA_PATH below before running.

Run:  python ccfraud_paysim.py
Outputs:
  paysim_benchmark_results.csv
  paysim_pr_curves.png
  paysim_roc_curves.png
  paysim_confusion_matrices.png
"""

import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

# ── Configuration ─────────────────────────────────────────────────────────
DATA_PATH       = r"C:\Users\LENOVO\Desktop\Nouveau dossier\paysim.csv"
RANDOM_STATE    = 42
TRAIN_FRAC      = 0.60
VAL_FRAC        = 0.20
TEST_FRAC       = 0.20
LOF_TRAIN_CAP   = 20_000
OCSVM_TRAIN_CAP = 20_000
DBSCAN_CAP      = 30_000
N_JOBS          = 2
CONTAMINATION   = None   # None → observed train fraud rate
np.random.seed(RANDOM_STATE)

# ── 1. Load and prepare ───────────────────────────────────────────────────
def load_and_prepare(path):
    print("  Reading CSV (this may take ~30s for 6M rows)...")
    df = pd.read_csv(path)

    # PaySim: fraud only in TRANSFER and CASH_OUT -- keep all rows but note
    print(f"  Raw shape: {df.shape}, fraud: {df['isFraud'].sum()}")

    # Sort chronologically by step (1 step = 1 hour)
    df = df.sort_values("step").reset_index(drop=True)

    # ── Feature engineering ──────────────────────────────────────────────
    # 1. Transaction type → one-hot (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
    type_dummies = pd.get_dummies(df["type"], prefix="type")

    # 2. Balance error features (difference between expected and actual balance)
    #    A common fraud signal: balance doesn't change after debit
    df["balance_orig_error"] = df["newbalanceOrig"] - (
        df["oldbalanceOrg"] - df["amount"])
    df["balance_dest_error"] = df["newbalanceDest"] - (
        df["oldbalanceDest"] + df["amount"])

    # 3. Assemble feature matrix (drop string columns + label)
    drop_cols = ["nameOrig", "nameDest", "type", "isFraud"]
    X = pd.concat([df.drop(columns=drop_cols), type_dummies], axis=1)
    y = df["isFraud"].astype(int).values

    # ── Chronological 60/20/20 split ─────────────────────────────────────
    n = len(df)
    c1 = int(n * TRAIN_FRAC)
    c2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    Xtr, ytr = X.iloc[:c1].values, y[:c1]
    Xva, yva = X.iloc[c1:c2].values, y[c1:c2]
    Xte, yte = X.iloc[c2:].values, y[c2:]

    # ── Leakage-free scaling (fit on train only) ──────────────────────────
    # Step 1: RobustScaler on the 8 numeric columns (heavy-tailed balances).
    #         This handles outlier amounts that would dominate StandardScaler.
    # Step 2: StandardScaler on ALL columns (including one-hot dummies).
    #         Without this second pass, balance columns (IQR in millions) still
    #         dominate Euclidean distance vs. one-hot columns (range 0-1), which
    #         causes DBSCAN eps estimation to degenerate (bimodal k-distance curve
    #         with ~99% of points at near-zero distance and a few at millions).
    #         Both scalers are fit on train only (leakage-free).
    n_numeric = 8   # first 8 columns are numeric; rest are dummies
    rob = RobustScaler()
    Xtr[:, :n_numeric] = rob.fit_transform(Xtr[:, :n_numeric])
    Xva[:, :n_numeric] = rob.transform(Xva[:, :n_numeric])
    Xte[:, :n_numeric] = rob.transform(Xte[:, :n_numeric])

    # Global StandardScaler across all columns (post-robust)
    std = StandardScaler()
    Xtr = std.fit_transform(Xtr)
    Xva = std.transform(Xva)
    Xte = std.transform(Xte)

    return Xtr, ytr, Xva, yva, Xte, yte


# ── 2. Detectors (same API as ULB pipeline) ───────────────────────────────
def run_isolation_forest(Xtr, Xte, contam):
    m = IsolationForest(n_estimators=200, contamination=contam,
                        random_state=RANDOM_STATE, n_jobs=N_JOBS)
    t = time.time(); m.fit(Xtr); ft = time.time() - t
    scores = -m.score_samples(Xte)
    y_pred = (m.predict(Xte) == -1).astype(int)
    return y_pred, scores, ft


def run_lof(Xtr, Xte, contam, novelty):
    ref = Xtr
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
    ref = Xtr
    if len(ref) > OCSVM_TRAIN_CAP:
        idx = np.random.choice(len(ref), OCSVM_TRAIN_CAP, replace=False)
        ref = ref[idx]
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=max(1e-4, float(contam)))
    t = time.time(); m.fit(ref); ft = time.time() - t
    scores = -m.decision_function(Xte)
    y_pred = (m.predict(Xte) == -1).astype(int)
    return y_pred, scores, ft


def estimate_eps(X, min_samples=10, cap=20_000):
    ref = X
    if len(ref) > cap:
        idx = np.random.choice(len(ref), cap, replace=False)
        ref = ref[idx]
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=N_JOBS)
    nn.fit(ref)
    dist, _ = nn.kneighbors(ref)
    k_dist = np.sort(dist[:, -1])

    # Primary: maximum second-derivative (curvature) knee detection.
    knee_curv = int(np.argmax(np.diff(np.diff(k_dist)))) + 1

    # Fallback: if the curvature knee is in the bottom 80% of the curve
    # (typical failure mode when k-dist is flat then suddenly vertical),
    # use the 90th percentile of k-distances instead — this is a robust
    # heuristic that places eps just beyond the "elbow" of dense points.
    if knee_curv < int(0.80 * len(k_dist)):
        eps = float(np.percentile(k_dist, 90))
        knee = int(np.searchsorted(k_dist, eps))
        method = "90th-pct fallback"
    else:
        eps = float(k_dist[knee_curv])
        knee = knee_curv
        method = "curvature"

    plt.figure(figsize=(8, 4))
    plt.plot(k_dist)
    plt.axvline(knee, color="grey", ls=":")
    plt.axhline(eps, color="red", ls="--",
                label=f"eps = {eps:.4f}  [{method}]")
    plt.xlabel("Points sorted by 10-NN distance")
    plt.ylabel("10-NN distance")
    plt.title("DBSCAN k-distance knee (PaySim)")
    plt.legend(); plt.tight_layout()
    plt.savefig("paysim_dbscan_kdistance.png", dpi=120); plt.close()
    return eps


def run_dbscan(Xte, contam, eps):
    t = time.time()
    if len(Xte) > DBSCAN_CAP:
        idx = np.random.choice(len(Xte), DBSCAN_CAP, replace=False)
        Xsub = Xte[idx]
        labels_sub = DBSCAN(eps=eps, min_samples=10,
                            n_jobs=N_JOBS).fit_predict(Xsub)
        noise_sub = (labels_sub == -1).astype(int)
        nn = NearestNeighbors(n_neighbors=1, n_jobs=N_JOBS).fit(Xsub)
        _, nbr = nn.kneighbors(Xte)
        y_pred = noise_sub[nbr[:, 0]]
    else:
        labels = DBSCAN(eps=eps, min_samples=10,
                        n_jobs=N_JOBS).fit_predict(Xte)
        y_pred = (labels == -1).astype(int)
    ft = time.time() - t
    scores = y_pred.astype(float)
    return y_pred, scores, ft


def run_kmeans(Xtr, Xte, contam):
    m = KMeans(n_clusters=8, random_state=RANDOM_STATE, n_init=10)
    t = time.time(); m.fit(Xtr); ft = time.time() - t
    dist = m.transform(Xte).min(axis=1)
    thresh = np.percentile(dist, 100 * (1 - contam))
    y_pred = (dist >= thresh).astype(int)
    return y_pred, dist, ft


def rank_average(score_dict):
    n = len(next(iter(score_dict.values())))
    rank_sum = np.zeros(n)
    for scores in score_dict.values():
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n)
        rank_sum += ranks
    return rank_sum


# ── 3. Evaluate one configuration ─────────────────────────────────────────
def evaluate(y_true, y_pred, scores, name, protocol, fit_time):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": name, "Protocol": protocol,
        "Precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC":    round(roc_auc_score(y_true, scores), 4),
        "PR_AUC_AP":  round(average_precision_score(y_true, scores), 4),
        "Fit_time_s": round(fit_time, 2),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


# ── 4. Plotting ────────────────────────────────────────────────────────────
def plot_curves(results, yte):
    from sklearn.metrics import precision_recall_curve, roc_curve

    score_results = [r for r in results if r["Model"] != "DBSCAN"]

    # ── PR curves (separate file) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in score_results:
        prec, rec, _ = precision_recall_curve(yte, r["_scores"])
        label = f"{r['Model']} ({r['Protocol'][:4]}) AP={r['PR_AUC_AP']:.3f}"
        ax.plot(rec, prec, label=label, lw=1.2)
    ax.axhline(yte.mean(), color="grey", ls="--",
               label=f"baseline={yte.mean():.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (PaySim)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig("paysim_pr_curves.png", dpi=140); plt.close()
    print("  Saved -> paysim_pr_curves.png")

    # ── ROC curves (separate file) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in score_results:
        fpr, tpr, _ = roc_curve(yte, r["_scores"])
        label = f"{r['Model']} ({r['Protocol'][:4]}) AUC={r['ROC_AUC']:.3f}"
        ax.plot(fpr, tpr, label=label, lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (PaySim)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig("paysim_roc_curves.png", dpi=140); plt.close()
    print("  Saved -> paysim_roc_curves.png")


def plot_confusion(results):
    data = {f"{r['Model']} ({r['Protocol'][:1]})":
            (r["TN"], r["FP"], r["FN"], r["TP"]) for r in results}
    n = len(data)
    cols = 3; rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, rows * 3.3))
    axes = axes.ravel()
    for ax, (name, (tn, fp, fn, tp)) in zip(axes, data.items()):
        cm = np.array([[tn, fp], [fn, tp]])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (PaySim). A=unsupervised, B=semi-supervised",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("paysim_confusion_matrices.png", dpi=140); plt.close()
    print("  Saved -> paysim_confusion_matrices.png")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("Loading and preprocessing PaySim...")
    Xtr, ytr, Xva, yva, Xte, yte = load_and_prepare(DATA_PATH)
    contam = CONTAMINATION or float(ytr.mean())
    print(f"  Train: {Xtr.shape} ({int(ytr.sum())} fraud)")
    print(f"  Val:   {Xva.shape} ({int(yva.sum())} fraud)")
    print(f"  Test:  {Xte.shape} ({int(yte.sum())} fraud)")
    print(f"  Fraud rate (train): {contam:.5f}\n")

    Xtr_norm = Xtr[ytr == 0]   # normals-only reference for Protocol B

    print("Estimating DBSCAN eps (k-distance knee)...")
    eps = estimate_eps(Xtr)
    print(f"  eps = {eps:.4f}  (plot -> paysim_dbscan_kdistance.png)\n")

    results = []

    # ── Protocol A: Unsupervised (fit on full train) ──────────────────────
    print("Protocol A - Unsupervised")
    for name, (y_pred, scores, ft) in [
        ("IsolationForest", run_isolation_forest(Xtr, Xte, contam)),
        ("LOF",             run_lof(Xtr, Xte, contam, novelty=False)),
        ("OneClassSVM",     run_ocsvm(Xtr, Xte, contam)),
        ("DBSCAN",          run_dbscan(Xte, contam, eps)),
        ("KMeans",          run_kmeans(Xtr, Xte, contam)),
    ]:
        r = evaluate(yte, y_pred, scores, name, "Unsupervised", ft)
        r["_scores"] = scores
        results.append(r)
        print(f"  {name:20s} PR-AUC={r['PR_AUC_AP']:.4f}  F1={r['F1']:.4f}")

    # ── Protocol B: Semi-supervised (fit on normals only) ─────────────────
    print("\nProtocol B - Semi-supervised (fit on normals only)")
    for name, (y_pred, scores, ft) in [
        ("IsolationForest", run_isolation_forest(Xtr_norm, Xte, contam)),
        ("LOF",             run_lof(Xtr_norm, Xte, contam, novelty=True)),
        ("OneClassSVM",     run_ocsvm(Xtr_norm, Xte, contam)),
        ("KMeans",          run_kmeans(Xtr_norm, Xte, contam)),
    ]:
        r = evaluate(yte, y_pred, scores, name, "Semi-supervised", ft)
        r["_scores"] = scores
        results.append(r)
        print(f"  {name:20s} PR-AUC={r['PR_AUC_AP']:.4f}  F1={r['F1']:.4f}")

    # ── Ensemble: rank-average of continuous-score detectors ─────────────
    print("\nEnsembles (rank-average)")
    for protocol, tag in [("Unsupervised", "A"), ("Semi-supervised", "B")]:
        score_dict = {
            r["Model"]: r["_scores"] for r in results
            if r["Protocol"] == protocol and r["Model"] != "DBSCAN"
        }
        ens_scores = rank_average(score_dict)
        thresh = np.percentile(ens_scores, 100 * (1 - contam))
        ens_pred = (ens_scores >= thresh).astype(int)
        r = evaluate(yte, ens_pred, ens_scores, "Ensemble", protocol, 0.0)
        r["_scores"] = ens_scores
        results.append(r)
        print(f"  Ensemble ({tag})          "
              f"PR-AUC={r['PR_AUC_AP']:.4f}  F1={r['F1']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    df_out = pd.DataFrame([{k: v for k, v in r.items() if k != "_scores"}
                           for r in results])
    print("\n=== PaySim Benchmark results ===")
    print(df_out.to_string(index=False))
    df_out.to_csv("paysim_benchmark_results.csv", index=False)
    print("\nSaved -> paysim_benchmark_results.csv")

    plot_curves(results, yte)
    plot_confusion(results)
    print("Done.")


if __name__ == "__main__":
    main()
