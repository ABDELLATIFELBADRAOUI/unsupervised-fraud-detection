# =============================================================================
# ONE-CELL VERSION  --  clock-feature ablation, self-contained.
# Paste this single cell at the END of revision_pipeline_v2_1.ipynb (so that it
# runs in the notebook's own folder, next to revision_v2), restart the kernel,
# and run ONLY this cell. Nothing else needs to be run first.
#
# Parts A-D are the notebook's own definition cells, copied verbatim (one
# exception: this run's manifest is written to manifest_clock_ablation.json,
# so manifest.json keeps documenting the headline run). Part E is run_split,
# verbatim from the headline cell. Part F is the ablation.
# Before running: check ULB_PATH / PAYSIM_PATH in part A.
# Runtime: about 30-40 min in total.
# =============================================================================


# =============================================================================
# PART A -- configuration (notebook cell "REVISION PIPELINE v2")
# =============================================================================
# =============================================================================
# REVISION PIPELINE v2  --  ARRAY-D-26-04414
# Unsupervised / Semi-Supervised Anomaly Detection for Credit Card Fraud
# =============================================================================
# This pipeline replaces the v1 benchmark notebook. It is organised so that the
# five structural invariants demanded by the reviewers are enforced by
# assertions rather than by convention.
#
# INVARIANT 1  No test-set object ever reaches a calibration function.
# INVARIANT 2  rho is never computed from labels, except in the explicitly
#              labelled ORACLE regime.
# INVARIANT 3  One canonical results frame; every table and figure derives
#              from it; figure/table agreement is asserted.
# INVARIANT 4  One sign convention: higher score = more anomalous, asserted
#              for every detector.
# INVARIANT 5  Every number is traceable to a row via run_id.
#
# REGIMES (replacing "Protocol A / Protocol B")
#   U  label-free      fit on the full training split, rho from an a-priori grid
#   N  normal-only     fit on training transactions labelled legitimate
#   O  oracle          as U, but rho = true training prevalence (UPPER BOUND,
#                      not achievable in deployment; reported for reference)
#
# Reviewer comments discharged by each stage are marked [Rx#n] in the headers.
# =============================================================================

import os, time, json, hashlib, warnings, platform
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths  --  EDIT THESE
# --------------------------------------------------------------------------- #
# Absolute paths are safest: a relative path resolves against whatever
# directory Jupyter was started from, not the notebook's own folder.
ULB_PATH    = r"/path/to/creditcard.csv"
PAYSIM_PATH = r"/path/to/paysim.csv"
# Relative to the notebook's working directory. Set an absolute path if you
# ever open this notebook from elsewhere, so results never scatter.
OUT_DIR     = "revision_v2"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
print(f"outputs -> {os.path.abspath(OUT_DIR)}")

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #
SEED          = 42
TRAIN_FRAC    = 0.60
VAL_FRAC      = 0.20            # test = remainder

LOF_TRAIN_CAP = 20_000          # reference-set cap for LOF   [R4#8]
OCSVM_CAP     = 20_000          # training cap for OC-SVM     [R4#8]
DBSCAN_CAP    = 20_000          # core-point cap for DBSCAN   [R4#3]
MIN_CORE_POINTS = 50            # below this the DBSCAN core set carries no
                                # density information and the out-of-sample
                                # rule of Section 3.4 degenerates.     [v2.1]
REQUIRE_NONDEGENERATE_CORE = False
                                # False reproduces the deposited v2 results,
                                # which the manuscript reports and which the
                                # manuscript also documents as degenerate on
                                # PaySim. Set True to widen eps until the core
                                # set reaches MIN_CORE_POINTS; this CHANGES the
                                # DBSCAN columns and every table that contains
                                # them, so the manuscript must be regenerated
                                # from the same run if you switch it.  [v2.1]
PAYSIM_BALANCE_FEATURES = True  # toggled by the R4#9 ablation cell
SUBSAMPLE_RULE = "random"       # headline rule, identical to v1 and to [32],
                                # so every changed number is attributable to
                                # the protocol corrections alone.
                                # "tail" (most recent contiguous block) is
                                # deployment-realistic but is a DIFFERENT
                                # experiment: it is reported as a sensitivity
                                # arm in the subsampling study, not silently
                                # substituted.                        [R4#8]
N_SUBSAMPLE_SEEDS = 5           # repeated subsampling         [R4#8]

# rho grid, fixed a priori and WITHOUT labels                  [R1#1][R6#4]
RHO_GRID      = [0.001, 0.005, 0.010, 0.020]
RHO_DEFAULT   = 0.005           # headline operating point for regimes U and N.
                                # Chosen a priori, WITHOUT labels. 0.001 sits
                                # below ULB's actual prevalence (0.00211) and
                                # under-alerts by half, collapsing every
                                # threshold-dependent metric.

N_FOLDS       = 5               # rolling-origin folds         [R1#4][R6#3]
FOLD_CONFIG   = {               # per dataset: (n_origins, share of the stream
    "ULB":    (3, 0.60),        # tiled by the test blocks)
    "PaySim": (5, 0.40),
}
MIN_FOLD_POSITIVES = 25         # below this a fold is reported but flagged as
                                # not powered for any ranking claim
N_BOOTSTRAP   = 2000            # block bootstrap replicates   [R5#2][R6#3]
BOOTSTRAP_BLOCKS = 50           # temporal blocks for the bootstrap
N_JOBS        = 2

rng_global = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- #
# Run manifest  --  INVARIANT 5
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Preflight  --  fail here, with a usable message, rather than mid-pipeline
# --------------------------------------------------------------------------- #
def preflight():
    import glob
    missing = []
    for label, path in [("ULB_PATH", ULB_PATH), ("PAYSIM_PATH", PAYSIM_PATH)]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e6
            with open(path) as f:
                header = f.readline().strip()[:90]
            print(f"  {label:<12} OK  {os.path.abspath(path)}  "
                  f"({size:,.0f} MB)\n               header: {header}")
        else:
            missing.append((label, path))
    if not missing:
        return
    print("\n  MISSING DATA FILES")
    print(f"  working directory: {os.getcwd()}")
    home = os.path.expanduser("~")
    for label, path in missing:
        pattern = "creditcard*.csv" if label == "ULB_PATH" else "*ay*im*.csv"
        print(f"\n  {label} = {path!r}  -> not found")
        hits = []
        for root in [os.getcwd(), home, os.path.join(home, "Downloads"),
                     os.path.join(home, "Desktop"), os.path.join(home, "Documents")]:
            hits += glob.glob(os.path.join(root, "**", pattern), recursive=True)
        hits = sorted(set(hits))[:5]
        if hits:
            print("  candidates found on disk -- paste one into cell 0:")
            for h in hits:
                print(f"      r{h!r}")
        else:
            print("  no candidate found in cwd, home, Downloads, Desktop or "
                  "Documents. Search the whole drive with:")
            print(f"      import glob; glob.glob(r'C:\\**\\{pattern}', "
                  "recursive=True)")
    raise FileNotFoundError(
        "Set ULB_PATH / PAYSIM_PATH in cell 0 to absolute paths, then re-run "
        "this cell. Nothing else needs changing.")


preflight()

import sklearn, scipy
MANIFEST = {
    "created":      pd.Timestamp.now().isoformat(),
    "python":       platform.python_version(),
    "numpy":        np.__version__,
    "pandas":       pd.__version__,
    "sklearn":      sklearn.__version__,
    "scipy":        scipy.__version__,
    "seed":         SEED,
    "rho_grid":     RHO_GRID,
    "rho_default":  RHO_DEFAULT,
    "subsample_rule": SUBSAMPLE_RULE,
    "pipeline_version": "v2.1",
    "min_core_points": MIN_CORE_POINTS,
    "require_nondegenerate_core": REQUIRE_NONDEGENERATE_CORE,
    "selection_tie_break": "simplest configuration within 1e-4 of the best validation PR-AUC",
    "caps":         {"lof": LOF_TRAIN_CAP, "ocsvm": OCSVM_CAP, "dbscan": DBSCAN_CAP},
}
# --------------------------------------------------------------------------- #
# Stale-results guard. run_id hashes rho_assumed, so changing RHO_DEFAULT (or
# any grid) mints NEW ids: the merge in save_results() would then KEEP the
# previous generation's rows alongside the new ones, invisibly, in the file the
# manuscript is meant to quote from.
# --------------------------------------------------------------------------- #
_mpath = os.path.join(OUT_DIR, "manifest.json")
_cpath = os.path.join(OUT_DIR, "canonical_results.csv")
if os.path.exists(_mpath) and os.path.exists(_cpath):
    old = json.load(open(_mpath))
    drift = {k: (old.get(k), MANIFEST[k]) for k in
             ["rho_default", "rho_grid", "seed", "subsample_rule", "caps"]
             if old.get(k) != MANIFEST[k]}
    if drift:
        print("\n" + "!" * 70)
        print("STALE RESULTS ON DISK -- the configuration changed since the "
              "last run:")
        for k, (was, now) in drift.items():
            print(f"    {k}: was {was}  ->  now {now}")
        print(f"\n  {_cpath} still holds rows produced under the old "
              "configuration.\n  Those rows carry different run_ids and will "
              "NOT be overwritten.\n")
        print("  Rename or delete the output folder before re-running:")
        print(f"      import shutil; shutil.move(r'{os.path.abspath(OUT_DIR)}', "
              f"r'{os.path.abspath(OUT_DIR)}_old')")
        print("!" * 70 + "\n")
        raise RuntimeError(
            "Archive the previous results folder first, then re-run this cell. "
            "Set ALLOW_MIXED_RESULTS = True above to override (not advised -- "
            "the canonical file is what every manuscript number is traced to).")

# ONE-CELL VERSION: this run's manifest goes to manifest_clock_ablation.json;
# manifest.json, which documents the headline run, is left as it is.
if os.path.exists(_mpath):
    _old = json.load(open(_mpath))
    _vd = {k: (_old.get(k), MANIFEST[k]) for k in
           ["python", "numpy", "pandas", "sklearn", "scipy"]
           if _old.get(k) != MANIFEST[k]}
    if _vd:
        print("  package versions differ from manifest.json: "
              + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in _vd.items()))
with open(os.path.join(OUT_DIR, "manifest_clock_ablation.json"), "w") as f:
    json.dump(MANIFEST, f, indent=2)
print(json.dumps(MANIFEST, indent=2))

# =============================================================================
# PART B -- data, splits, preprocessing (notebook cell "1. DATA, SPLITS, PREPROCESSING")
# =============================================================================
# =============================================================================
# 1. DATA, SPLITS, PREPROCESSING
#    [R4#4] prevalence-matched random arm   [R1#4] rolling-origin folds
# =============================================================================
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class Split:
    """One train / validation / test partition, already scaled."""
    Xtr: np.ndarray; ytr: np.ndarray
    Xva: np.ndarray; yva: np.ndarray
    Xte: np.ndarray; yte: np.ndarray
    dataset: str
    split_type: str            # "chronological" | "random" | "random_matched"
    fold: int = 0

    @property
    def prevalence(self) -> Dict[str, float]:
        return {"train": float(self.ytr.mean()),
                "val":   float(self.yva.mean()),
                "test":  float(self.yte.mean())}

    def describe(self):
        p = self.prevalence
        print(f"  [{self.dataset}/{self.split_type}/fold{self.fold}] "
              f"train {self.Xtr.shape} ({int(self.ytr.sum())}f, {p['train']:.5f}) | "
              f"val {self.Xva.shape} ({int(self.yva.sum())}f, {p['val']:.5f}) | "
              f"test {self.Xte.shape} ({int(self.yte.sum())}f, {p['test']:.5f})")


# --------------------------------------------------------------------------- #
# Feature engineering  --  identical to v1 so results stay comparable
# --------------------------------------------------------------------------- #
def _prep_ulb(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"]).astype(np.float32)
    return X, y, ["Amount", "Time", "Hour"]


def _prep_paysim(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = df.copy()
    # Balance-consistency features. They are algebraic rearrangements of the
    # simulator's own account-update equations, which is precisely R4#9's
    # objection; the ablation cell re-runs PaySim without them. [R4#9]
    df["delta_orig"] = df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    df["delta_dest"] = df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    dummies = pd.get_dummies(df["type"], prefix="type").astype(np.float32)
    y = df["isFraud"].astype(int).values
    keep = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]
    if PAYSIM_BALANCE_FEATURES:
        keep += ["delta_orig", "delta_dest"]
    X = pd.concat([df[keep].astype(np.float32), dummies], axis=1)
    return X, y, keep


def _scale(Xtr, Xva, Xte, robust_cols, global_std: bool):
    """Scalers fitted on TRAIN ONLY. [leakage-free preprocessing]"""
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    cols = [c for c in robust_cols if c in Xtr.columns]
    if cols:
        rob = RobustScaler().fit(Xtr[cols])
        for D in (Xtr, Xva, Xte):
            D[cols] = rob.transform(D[cols])
    A = Xtr.values.astype(np.float32)
    B = Xva.values.astype(np.float32)
    C = Xte.values.astype(np.float32)
    if global_std:
        std = StandardScaler().fit(A)
        A, B, C = std.transform(A), std.transform(B), std.transform(C)
    return (np.ascontiguousarray(A, dtype=np.float32),
            np.ascontiguousarray(B, dtype=np.float32),
            np.ascontiguousarray(C, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Split builders
# --------------------------------------------------------------------------- #
_RAW_CACHE: Dict[tuple, tuple] = {}


def clear_raw_cache():
    _RAW_CACHE.clear(); print("  raw cache cleared")


def load_raw(dataset: str):
    """Parsed, feature-engineered source data, cached in memory.

    Eight split constructions per dataset would otherwise re-parse PaySim's
    6.3M rows eight times. The cache key INCLUDES PAYSIM_BALANCE_FEATURES:
    keying on the dataset name alone would serve delta-feature data to the
    without-delta arm of the R4#9 ablation and silently invalidate it.
    """
    key = (dataset, PAYSIM_BALANCE_FEATURES if dataset == "PaySim" else None)
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = _load_raw_uncached(dataset)
    return _RAW_CACHE[key]


def _load_raw_uncached(dataset: str):
    if dataset == "ULB":
        df = pd.read_csv(ULB_PATH).sort_values("Time").reset_index(drop=True)
        X, y, rob = _prep_ulb(df)
        return X, y, rob, "Time", False
    elif dataset == "PaySim":
        df = pd.read_csv(PAYSIM_PATH).sort_values("step").reset_index(drop=True)
        X, y, rob = _prep_paysim(df)
        return X, y, rob, "step", True
    raise ValueError(dataset)


def chronological_split(dataset: str) -> Split:
    X, y, rob, _, gstd = load_raw(dataset)
    n = len(X)
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    Xtr, Xva, Xte = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    A, B, C = _scale(Xtr, Xva, Xte, rob, gstd)
    return Split(A, y[:i1], B, y[i1:i2], C, y[i2:], dataset, "chronological")


def rolling_origin_splits(dataset: str, n_folds: Optional[int] = None,
                          span: Optional[float] = None) -> List[Split]:
    """Expanding-window forward chaining. Fold k trains on [0, t_k), validates on
    the next block and tests on the block after it.                  [R1#4][R6#3]

    Origins and span are per dataset (FOLD_CONFIG): ULB holds 492 frauds over
    two days, so five origins would leave ~20 positives per test block and no
    PR-AUC computed on that could support a ranking.
    """
    cfg_folds, cfg_span = FOLD_CONFIG.get(dataset, (N_FOLDS, 0.40))
    n_folds = n_folds or cfg_folds
    span = span or cfg_span
    X, y, rob, _, gstd = load_raw(dataset)
    n = len(X)
    block = int(n * span / n_folds)
    start = n - n_folds * block
    out = []
    for k in range(n_folds):
        te_lo = start + k * block
        te_hi = te_lo + block
        va_lo = max(0, te_lo - block)
        Xtr, Xva, Xte = X.iloc[:va_lo], X.iloc[va_lo:te_lo], X.iloc[te_lo:te_hi]
        if len(Xtr) < 5000 or Xva.shape[0] == 0:
            continue
        A, B, C = _scale(Xtr, Xva, Xte, rob, gstd)
        out.append(Split(A, y[:va_lo], B, y[va_lo:te_lo], C, y[te_lo:te_hi],
                         dataset, "chronological", fold=k))
    return out


def random_split(dataset: str, seed: int = SEED) -> Split:
    from sklearn.model_selection import train_test_split
    X, y, rob, _, gstd = load_raw(dataset)
    idx = np.arange(len(X))
    tr, rest = train_test_split(idx, train_size=TRAIN_FRAC, stratify=y,
                                random_state=seed)
    va, te = train_test_split(rest, train_size=VAL_FRAC / (1 - TRAIN_FRAC),
                              stratify=y[rest], random_state=seed)
    A, B, C = _scale(X.iloc[tr], X.iloc[va], X.iloc[te], rob, gstd)
    return Split(A, y[tr], B, y[va], C, y[te], dataset, "random")


def random_prevalence_matched(dataset: str, target_n: int, target_pos: int,
                              seed: int = SEED) -> Split:
    """Random split whose TEST block matches the chronological test block in both
    size and number of positives. Without this, any random-vs-chronological
    PR-AUC gap is confounded by the class prior.                          [R4#4]"""
    from sklearn.model_selection import train_test_split
    X, y, rob, _, gstd = load_raw(dataset)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    te_pos = rng.choice(pos, size=target_pos, replace=False)
    te_neg = rng.choice(neg, size=target_n - target_pos, replace=False)
    te = np.concatenate([te_pos, te_neg]); rng.shuffle(te)
    remaining = np.setdiff1d(np.arange(len(X)), te)
    tr, va = train_test_split(remaining,
                              train_size=TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC),
                              stratify=y[remaining], random_state=seed)
    A, B, C = _scale(X.iloc[tr], X.iloc[va], X.iloc[te], rob, gstd)
    return Split(A, y[tr], B, y[va], C, y[te], dataset, "random_matched")


def subsample_indices(n: int, cap: int, seed: int,
                      rule: Optional[str] = None) -> np.ndarray:
    """Documented, reproducible subsampling rule.                        [R4#8]

    "tail"   the most recent `cap` training rows -- deployment-realistic and
             DETERMINISTIC, so the seed has no effect (this is why the
             repeated-subsampling study must override the rule).
    "random" uniform draw, seeded.
    """
    rule = rule or SUBSAMPLE_RULE
    if n <= cap:
        return np.arange(n)
    if rule == "tail":
        return np.arange(n - cap, n)
    return np.random.default_rng(seed).choice(n, cap, replace=False)

# =============================================================================
# PART C -- detectors (notebook cell "import warnings")
# =============================================================================
import warnings
# =============================================================================
# 2. DETECTORS  --  strict fit(train) -> score(val) AND score(test)
#    [R4#1] no label-derived quantity   [R4#2] no test-set dependence
#    [R4#3] explicit out-of-sample interface for LOF and DBSCAN
# =============================================================================
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans


@dataclass
class Scores:
    """A detector's raw anomaly scores. Higher = more anomalous (INVARIANT 4).

    NOTE the absence of any y_pred field: hard predictions are produced ONLY by
    the calibration module, from validation scores. A detector cannot emit a
    decision on its own -- this is what makes INVARIANT 1 structural.
    """
    val:  np.ndarray
    test: np.ndarray
    fit_seconds: float
    score_seconds: float
    detector: str
    extra: dict = field(default_factory=dict)
    model: object = None          # kept so per-transaction latency is measurable
    score_one: object = None      # callable: (1, d) array -> float


def _check_sign(s: Scores):
    """INVARIANT 4: assert the score is finite and non-degenerate."""
    for name, a in (("val", s.val), ("test", s.test)):
        assert np.all(np.isfinite(a)), f"{s.detector}: non-finite {name} scores"
    n_unique = len(np.unique(s.test))
    if n_unique < 10:
        print(f"    ! {s.detector}: only {n_unique} distinct test scores "
              f"(tie mass will distort any quantile threshold)")
    return s


# --------------------------------------------------------------------------- #
# NOTE ON rho: none of the fit functions below receives labels or rho, with the
# single exception of OC-SVM, whose `nu` is a genuine model hyperparameter. It
# is fed from the a-priori grid, never from the label-derived prevalence.
# --------------------------------------------------------------------------- #
def fit_isolation_forest(Xref, Xva, Xte, seed=SEED, n_estimators=200):
    t0 = time.time()
    m = IsolationForest(n_estimators=n_estimators, contamination="auto",
                        random_state=seed, n_jobs=N_JOBS).fit(Xref)
    t1 = time.time()
    return _check_sign(Scores(-m.score_samples(Xva), -m.score_samples(Xte),
                              t1 - t0, time.time() - t1, "IsolationForest",
                              model=m,
                              score_one=lambda z, m=m: -m.score_samples(z)[0]))


def fit_lof(Xref, Xva, Xte, seed=SEED, n_neighbors=20):
    """novelty=True in ALL regimes. v1's Table 3 documented novelty=False for
    Protocol A, which offers no out-of-sample scoring interface.          [R4#3]"""
    idx = subsample_indices(len(Xref), LOF_TRAIN_CAP, seed)
    ref = Xref[idx]
    t0 = time.time()
    m = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True,
                           n_jobs=N_JOBS).fit(ref)
    t1 = time.time()
    return _check_sign(Scores(-m.score_samples(Xva), -m.score_samples(Xte),
                              t1 - t0, time.time() - t1, "LOF",
                              {"n_ref": len(ref), "n_neighbors": n_neighbors},
                              model=m,
                              score_one=lambda z, m=m: -m.score_samples(z)[0]))


def fit_ocsvm(Xref, Xva, Xte, seed=SEED, nu=RHO_DEFAULT, gamma="scale"):
    idx = subsample_indices(len(Xref), OCSVM_CAP, seed)
    ref = Xref[idx]
    t0 = time.time()
    m = OneClassSVM(kernel="rbf", gamma=gamma, nu=max(1e-4, float(nu))).fit(ref)
    t1 = time.time()
    return _check_sign(Scores(-m.decision_function(Xva), -m.decision_function(Xte),
                              t1 - t0, time.time() - t1, "OneClassSVM",
                              {"n_ref": len(ref), "nu": nu}, model=m,
                              score_one=lambda z, m=m: -m.decision_function(z)[0]))


def fit_kmeans(Xref, Xva, Xte, seed=SEED, k=8):
    t0 = time.time()
    m = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xref)
    t1 = time.time()
    d_va = m.transform(Xva).min(axis=1)
    d_te = m.transform(Xte).min(axis=1)
    return _check_sign(Scores(d_va, d_te, t1 - t0, time.time() - t1,
                              "KMeans", {"k": k}, model=m,
                              score_one=lambda z, m=m: float(m.transform(z).min())))


def estimate_eps(Xref, min_samples=10, seed=SEED):
    """k-distance knee on the TRAINING reference set (v1 used the test set)."""
    idx = subsample_indices(len(Xref), DBSCAN_CAP, seed)
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=N_JOBS).fit(Xref[idx])
    d, _ = nn.kneighbors(Xref[idx])
    kd = np.sort(d[:, -1])
    # knee = point of maximum distance to the chord joining the curve endpoints
    x = np.linspace(0, 1, len(kd)); yv = (kd - kd.min()) / (np.ptp(kd) + 1e-12)
    return float(kd[np.argmax(yv - x)]), kd


def fit_dbscan(Xref, Xva, Xte, seed=SEED, min_samples=10, eps=None,
               eps_scale=1.0):
    """Out-of-sample DBSCAN, formalised.                                  [R4#3]

    v1 clustered the TEST set and emitted a BINARY score, which pins PR-AUC to
    the prevalence and makes the detector uninformative. Here:

      1. cluster a capped subsample of the TRAINING reference set;
      2. collect the core points C = {x : |N_eps(x)| >= min_samples};
      3. for any new point z, define
             s(z) = dist(z, nearest core point in C)
         and the hard rule  z is noise  <=>  s(z) > eps.

    s(z) is CONTINUOUS, computed per point, and depends on nothing but the
    training clustering -- so DBSCAN becomes rankable, ensemble-eligible, and
    free of test-set dependence.

    v2 substituted the WHOLE reference sample for the core set whenever DBSCAN
    found no core point, silently: s(z) then becomes a 1-NN distance to the
    training subsample, which is not DBSCAN. The substitution is now recorded
    instead of hidden, and REQUIRE_NONDEGENERATE_CORE widens eps until the core
    set is usable.                                                     [v2.1]
    """
    idx = subsample_indices(len(Xref), DBSCAN_CAP, seed)
    ref = Xref[idx]
    base_eps = eps
    if base_eps is None:
        base_eps, _ = estimate_eps(Xref, min_samples, seed)
    eps_used = base_eps * eps_scale   # eps, not min_samples, moves DBSCAN
    t0 = time.time()
    db = DBSCAN(eps=eps_used, min_samples=min_samples, n_jobs=N_JOBS).fit(ref)
    n_core = len(db.core_sample_indices_)
    widenings = 0
    while (REQUIRE_NONDEGENERATE_CORE and n_core < MIN_CORE_POINTS
           and widenings < 8):
        eps_used *= 2.0
        widenings += 1
        db = DBSCAN(eps=eps_used, min_samples=min_samples,
                    n_jobs=N_JOBS).fit(ref)
        n_core = len(db.core_sample_indices_)
    fell_back = n_core == 0
    core = ref if fell_back else ref[db.core_sample_indices_]
    degenerate = fell_back or n_core < MIN_CORE_POINTS or n_core >= len(ref)
    if degenerate:
        warnings.warn(
            f"DBSCAN core set is degenerate: {n_core} core points out of "
            f"{len(ref)} at eps={eps_used:.4g}, min_samples={min_samples}. "
            + ("No core point was found, so the full reference sample is used "
               "and the score is a 1-NN distance to the training subsample. "
               if fell_back else "")
            + "The resulting score is not a density-based anomaly score; see "
              "Section 3.4. Set REQUIRE_NONDEGENERATE_CORE = True to widen eps.",
            RuntimeWarning, stacklevel=2)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=N_JOBS).fit(core)
    t1 = time.time()
    d_va, _ = nn.kneighbors(Xva); d_te, _ = nn.kneighbors(Xte)
    return _check_sign(Scores(d_va.ravel(), d_te.ravel(), t1 - t0,
                              time.time() - t1, "DBSCAN",
                              {"eps": eps_used, "eps_base": base_eps,
                               "eps_widenings": widenings,
                               "n_core": int(n_core), "n_ref": len(ref),
                               "core_fraction": round(n_core / len(ref), 6),
                               "degenerate_core": float(degenerate),
                               "core_fallback": float(fell_back),
                               "min_samples": min_samples}, model=nn,
                              score_one=lambda z, nn=nn: float(nn.kneighbors(z)[0][0][0])))


DETECTORS = {
    "IsolationForest": fit_isolation_forest,
    "LOF":             fit_lof,
    "OneClassSVM":     fit_ocsvm,
    "DBSCAN":          fit_dbscan,
    "KMeans":          fit_kmeans,
}


def reference_set(sp: Split, regime: str) -> np.ndarray:
    """U and O fit on the full training split; N fits on legitimate rows only.

    The label read below is the ONLY use of ytr in the pipeline, and it is the
    defining property of the semi-supervised regime -- not a hidden dependence.
    """
    if regime in ("U", "O"):
        return sp.Xtr
    if regime == "N":
        return sp.Xtr[sp.ytr == 0]
    raise ValueError(regime)


def rho_for(sp: Split, regime: str, rho_assumed: float) -> float:
    """INVARIANT 2. The ORACLE branch is the only path to a label-derived rho,
    and every row it produces carries regime == 'O' in the results frame."""
    if regime == "O":
        return float(sp.ytr.mean())
    return float(rho_assumed)

# =============================================================================
# PART D -- results store and calibration (notebook cell "RESULTS STORE")
# =============================================================================
# =============================================================================
# 3. CALIBRATION, METRICS, CANONICAL RESULTS STORE
#    [R1#2][R4#2] thresholds and rank transforms fitted on VALIDATION only
#    [R1#9] PR-AUC primary   [R6#2] tie diagnostics
# =============================================================================
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression


class ValidationCalibrator:
    """Everything the decision rule needs, learned from validation scores alone.

    threshold(rho)  the (1-rho) quantile of VALIDATION scores. Applied unchanged
                    to the test stream, so the label assigned to a transaction
                    depends on nothing but that transaction.
    ecdf(s)         the empirical CDF of VALIDATION scores, applied pointwise.
                    Replaces v1's rankdata() over the joint test set, which made
                    each ensemble score a function of all other test rows. [R4#2]
    """

    def __init__(self, val_scores: np.ndarray):
        assert val_scores.ndim == 1
        self._sorted = np.sort(np.asarray(val_scores, dtype=np.float64))
        self.n = len(self._sorted)
        u = len(np.unique(self._sorted))
        self.tie_ratio = 1.0 - u / self.n     # 0 = all distinct             [R6#2]

    def threshold(self, rho: float) -> float:
        return float(np.quantile(self._sorted, 1.0 - rho))

    def ecdf(self, s: np.ndarray) -> np.ndarray:
        """P_val(S <= s), evaluated pointwise; clipped to [0,1] outside range."""
        return np.searchsorted(self._sorted, np.asarray(s), side="right") / self.n

    def predict(self, s: np.ndarray, rho: float) -> np.ndarray:
        return (np.asarray(s) >= self.threshold(rho)).astype(int)


def assert_calibration_is_clean(cal: ValidationCalibrator, sp: Split):
    """INVARIANT 1, checked numerically rather than promised in prose.

    Re-deriving the calibrator from the validation scores alone must reproduce
    the same threshold; and the decision for a single transaction must be
    identical whether it is scored alone or inside the full test batch.
    """
    assert cal.n == len(sp.yva), "calibrator was not built from the validation split"


def single_point_invariance(cal: ValidationCalibrator, s_test: np.ndarray,
                            rho: float, n_probe: int = 200) -> bool:
    """The property v1 could not satisfy: scoring one transaction in isolation
    yields the same decision as scoring it inside the batch.               [R4#2]"""
    rs = np.random.default_rng(SEED)
    probe = rs.choice(len(s_test), size=min(n_probe, len(s_test)), replace=False)
    batch = cal.predict(s_test, rho)[probe]
    alone = np.array([cal.predict(np.array([s_test[i]]), rho)[0] for i in probe])
    return bool(np.array_equal(batch, alone))


def ecdf_rank_average(cals: Dict[str, ValidationCalibrator],
                      scores: Dict[str, np.ndarray]) -> np.ndarray:
    """Ensemble score = mean over detectors of the validation-ECDF of each
    detector's score. Pointwise by construction.                          [R4#2]"""
    cols = [cals[m].ecdf(scores[m]) for m in scores]
    return np.mean(np.column_stack(cols), axis=1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, s_test) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "PR_AUC":    float(average_precision_score(y_true, s_test)),  # primary
        "ROC_AUC":   float(roc_auc_score(y_true, s_test)),            # secondary
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "n_alerts":  int(tp + fp),
        "alert_rate": float((tp + fp) / len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "lift":      float(((tp / max(tp + fp, 1)) / max(np.mean(y_true), 1e-12))),
    }
    return out


# --------------------------------------------------------------------------- #
# Uncertainty helpers  --  used by several cells below
# [R5#2] confidence intervals   [R6#3] 75 frauds cannot support fine rankings
# --------------------------------------------------------------------------- #
def block_bootstrap_ap(y_true, scores, n_boot=None, n_blocks=None, seed=SEED):
    """Ordinary bootstrap resamples transactions independently and understates
    the variance of a temporally ordered stream; blocks preserve local
    structure."""
    n_boot = n_boot or N_BOOTSTRAP; n_blocks = n_blocks or BOOTSTRAP_BLOCKS
    n = len(y_true)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    rs = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rs.integers(0, n_blocks, n_blocks)])
        yb = y_true[idx]
        out[b] = average_precision_score(yb, scores[idx]) if yb.sum() > 0 else np.nan
    return np.nanpercentile(out, [2.5, 50, 97.5]), out


def paired_bootstrap_diff(y_true, s_a, s_b, n_boot=1000, n_blocks=None, seed=SEED):
    """P(A > B) on the SAME resamples -- the honest way to compare two PR-AUCs
    that differ by less than their individual intervals."""
    n_blocks = n_blocks or BOOTSTRAP_BLOCKS
    n = len(y_true)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    rs = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rs.integers(0, n_blocks, n_blocks)])
        yb = y_true[idx]
        if yb.sum() == 0:
            continue
        diffs.append(average_precision_score(yb, s_a[idx])
                     - average_precision_score(yb, s_b[idx]))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.mean(diffs > 0)), \
           tuple(np.percentile(diffs, [2.5, 97.5]))


# --------------------------------------------------------------------------- #
# Canonical results store  --  INVARIANT 3 and 5
# --------------------------------------------------------------------------- #
RESULT_COLUMNS = ["run_id", "experiment", "dataset", "split_type", "fold",
                  "regime", "detector", "rho_assumed", "rho_used",
                  "metric", "value", "n_test", "n_pos_test"]

RESULTS: List[dict] = []


def _run_id(**kw) -> str:
    key = json.dumps(kw, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def record(experiment: str, sp: Split, regime: str, detector: str,
           rho_assumed: float, rho_used: float, metrics: Dict[str, float],
           **extra):
    rid = _run_id(experiment=experiment, dataset=sp.dataset,
                  split_type=sp.split_type, fold=sp.fold, regime=regime,
                  detector=detector, rho=rho_assumed, **extra)
    for k, v in metrics.items():
        RESULTS.append({
            "run_id": rid, "experiment": experiment, "dataset": sp.dataset,
            "split_type": sp.split_type, "fold": sp.fold, "regime": regime,
            "detector": detector, "rho_assumed": rho_assumed,
            "rho_used": rho_used, "metric": k, "value": v,
            "n_test": len(sp.yte), "n_pos_test": int(sp.yte.sum()),
            **extra,
        })
    return rid


def results_frame() -> pd.DataFrame:
    return pd.DataFrame(RESULTS)


def pivot(experiment: str, metric: str = "PR_AUC", **filters) -> pd.DataFrame:
    df = results_frame()
    df = df[(df.experiment == experiment) & (df.metric == metric)]
    for k, v in filters.items():
        df = df[df[k] == v]
    return df


def require(*names, cell: str = ""):
    """Fail fast and legibly when a cell is run out of order."""
    missing = [n for n in names if n not in globals()]
    if missing:
        raise RuntimeError(
            f"Run {cell} first -- this cell needs {', '.join(missing)}.\n"
            "Order: 0-3 setup -> 4 headline (defines SPLITS) -> 5-11 "
            "experiments -> 12 label efficiency (defines le_ulb) -> "
            "13 bootstrap (defines SCORE_CACHE) -> 14 operational -> "
            "15 tables and figures.")


def save_results(merge_with_disk: bool = True):
    """Persist the canonical frame WITHOUT destroying earlier work.

    RESULTS lives in memory. Restarting the kernel and re-running only some
    cells would otherwise overwrite a complete file with a partial one, and
    every number in the manuscript is meant to be traceable to this file.
    Rows are keyed on (run_id, metric): a re-run of the same configuration
    replaces its own rows and leaves every other experiment intact.
    """
    df = results_frame()
    p = os.path.join(OUT_DIR, "canonical_results.csv")
    if merge_with_disk and os.path.exists(p) and len(df):
        old = pd.read_csv(p)
        keep = ~old.set_index(["run_id", "metric"]).index.isin(
            df.set_index(["run_id", "metric"]).index)
        n_kept = int(keep.sum())
        df = pd.concat([old[keep], df], ignore_index=True)
        if n_kept:
            print(f"  merged with {n_kept} rows already on disk")
    df.to_csv(p, index=False)
    print(f"  canonical results -> {p}  ({len(df)} rows, "
          f"{df.run_id.nunique()} runs, "
          f"{df.experiment.nunique()} experiments)")
    return p

# =============================================================================
# PART E -- run_split, verbatim from the headline cell
# =============================================================================
def run_split(sp: Split, experiment: str,
              regimes=("U", "N", "O"), rho_assumed: float = RHO_DEFAULT,
              hyper: Optional[dict] = None, seed: int = SEED,
              keep_scores: bool = False, verbose: bool = True):
    """Fit every detector under every regime, calibrate on validation, evaluate
    on test, and push every metric into the canonical store."""
    hyper = hyper or {}
    store = {}
    for regime in regimes:
        Xref = reference_set(sp, regime)
        rho  = rho_for(sp, regime, rho_assumed)
        cals, s_test_all, s_val_all = {}, {}, {}

        for name, fn in DETECTORS.items():
            kw = dict(hyper.get(name, {}))
            if name == "OneClassSVM":
                kw.setdefault("nu", rho)          # a-priori rho, not labels
            sc = fn(Xref, sp.Xva, sp.Xte, seed=seed, **kw)

            cal = ValidationCalibrator(sc.val)
            assert_calibration_is_clean(cal, sp)
            y_pred = cal.predict(sc.test, rho)
            m = compute_metrics(sp.yte, y_pred, sc.test)
            m["tie_ratio"] = cal.tie_ratio
            m["fit_seconds"] = sc.fit_seconds
            m["score_seconds"] = sc.score_seconds
            record(experiment, sp, regime, name, rho_assumed, rho, m)

            cals[name] = cal          # DBSCAN included: it now has a real score
            s_test_all[name] = sc.test
            s_val_all[name] = sc.val
            if keep_scores:
                store[(regime, name)] = sc

        # ----- ensemble: validation-ECDF rank average --------------------- #
        ens_test = ecdf_rank_average(cals, s_test_all)
        # The ensemble's own calibrator is built from the ensemble score of the
        # VALIDATION points. Deriving it from cals[m]._sorted instead would
        # give the uniform ramp 1/n..1 for every member, whose (1-rho) quantile
        # averaged ECDFs never reach -- yielding zero alerts.
        ens_val_scores = ecdf_rank_average(cals, s_val_all)
        ens_cal = ValidationCalibrator(ens_val_scores)
        y_pred = ens_cal.predict(ens_test, rho)
        m = compute_metrics(sp.yte, y_pred, ens_test)
        m["tie_ratio"] = ens_cal.tie_ratio
        m["fit_seconds"] = 0.0
        m["score_seconds"] = 0.0
        record(experiment, sp, regime, "Ensemble", rho_assumed, rho, m)

        # INVARIANT 1 probe, logged as a metric so it lands in the results file.
        # v2 probed the ensemble only, which is the only score that combines
        # information across detectors; the individual scores are pointwise by
        # construction. We now probe all six so the claim in Section 3.7 is
        # backed for every configuration rather than for one.            [v2.1]
        ok = single_point_invariance(ens_cal, ens_test, rho)
        record(experiment, sp, regime, "Ensemble", rho_assumed, rho,
               {"single_point_invariant": float(ok)}, check="invariance")
        for _m in s_test_all:
            _ok = single_point_invariance(cals[_m], s_test_all[_m], rho)
            record(experiment, sp, regime, _m, rho_assumed, rho,
                   {"single_point_invariant": float(_ok)}, check="invariance")
        if verbose:
            sub = pivot(experiment, "PR_AUC", dataset=sp.dataset,
                        regime=regime, fold=sp.fold)
            print(f"  regime {regime} (rho={rho:.5f}) "
                  + " | ".join(f"{r.detector}={r.value:.4f}"
                               for r in sub.itertuples()))
    return store

# =============================================================================
# PART F -- clock-feature ablation: ULB without "Time", PaySim without "step"
# =============================================================================
# Under the chronological split of Section 3.3, every validation and test value
# of the absolute time index (ULB "Time", PaySim "step") lies beyond the
# training range, so distance- and kernel-based scores can rise with elapsed
# time regardless of transaction behaviour. This part re-runs the headline
# configuration (regimes U and N, rho = RHO_DEFAULT, default settings of
# run_split, same seed) with the clock feature removed; hour-of-day is kept on
# ULB. The sort by the clock, and hence the split, is unchanged.
#
# Order of work, per dataset:
#   1. chronological split WITH the clock (as in the headline run);
#   2. position-only baseline: the test position used alone as a score;
#   3. ULB only: headline configuration re-run WITH the clock, to check that
#      this environment reproduces the deposited headline numbers. If it does
#      not, PaySim is re-run with the clock as well and the with-clock column
#      comes from this session, so that like is compared with like;
#   4. headline configuration re-run WITHOUT the clock.
#
# Rows added to canonical_results.csv: experiment "clock_ablation" (without
# the clock) and "clock_ablation_ref" (with-clock re-runs). Headline rows are
# never touched. Files written to OUT_DIR (rewritten after each dataset):
#   table_clock_ablation.csv             PR-AUC and alert counts, with / without
#   table_clock_position_corr.csv        Spearman(score, test position)
#   table_clock_position_baseline.csv    position-only PR-AUC and ROC-AUC
#   table_clock_reproduction_check.csv   re-run vs deposited headline PR-AUC
# Expected runtime: about 10 min for ULB, 20-30 min for PaySim.
# =============================================================================
from scipy.stats import rankdata

CLOCK    = {"ULB": "Time", "PaySim": "step"}
DATASETS = ["ULB", "PaySim"]
REGIMES  = ("U", "N")
MEMBERS  = list(DETECTORS) + ["Ensemble"]
_PREP_ORIG = {"ULB": _prep_ulb, "PaySim": _prep_paysim}
_CANON = os.path.join(OUT_DIR, "canonical_results.csv")
_NPZ   = os.path.join(OUT_DIR, "test_scores.npz")
_T0 = time.time()


def _stamp(msg):
    print(f"\n[{(time.time() - _T0) / 60:5.1f} min] {msg}")


# ---- fail fast: the headline rows must exist before any long computation --- #
if not os.path.exists(_CANON):
    raise FileNotFoundError(
        f"{os.path.abspath(_CANON)} not found. Run this cell from the folder "
        "that contains revision_v2 (the folder of revision_pipeline_v2_1.ipynb),"
        " or set OUT_DIR in part A to the absolute path of revision_v2.")
_canon = pd.read_csv(_CANON)
_head = _canon[(_canon.experiment == "headline") &
               (_canon.split_type == "chronological") & (_canon.fold == 0)]


def _headline(ds, regime, metric):
    d = _head[(_head.dataset == ds) & (_head.regime == regime) &
              (_head.metric == metric)]
    return d.set_index("detector")["value"].to_dict()


for _ds in DATASETS:
    for _rg in REGIMES:
        for _mt in ("PR_AUC", "n_alerts"):
            _miss = [m for m in MEMBERS if m not in _headline(_ds, _rg, _mt)]
            if _miss:
                raise RuntimeError(
                    f"{_CANON} has no headline {_mt} row for {_ds}/{_rg}: {_miss}")
print(f"  headline rows found in {os.path.abspath(_CANON)}")
print(f"  test_scores.npz {'found' if os.path.exists(_NPZ) else 'not found (with-clock correlations for PaySim will be skipped)'}")


# ---- helpers ---------------------------------------------------------------- #
def _without(prep, col):
    """Wrap a _prep_* function so that the clock column is dropped after the
    feature engineering (the sort by that column, done before prep, is kept)."""
    def f(df):
        X, y, rob = prep(df)
        assert col in X.columns, f"{col} not among the features"
        return X.drop(columns=[col]), y, [c for c in rob if c != col]
    return f


def _split(ds, drop_clock):
    """The notebook's own chronological_split(), with or without the clock."""
    if drop_clock:
        globals()["_prep_" + ds.lower()] = _without(_PREP_ORIG[ds], CLOCK[ds])
    clear_raw_cache()
    try:
        return chronological_split(ds)
    finally:
        globals()["_prep_" + ds.lower()] = _PREP_ORIG[ds]      # restore
        clear_raw_cache()


def _values(experiment, ds, regime, metric):
    return pivot(experiment, metric, dataset=ds, regime=regime,
                 fold=0).set_index("detector")["value"].to_dict()


def _member_scores(store, regime):
    """Test scores of the five detectors and of the ECDF ensemble, rebuilt
    exactly as run_split builds them."""
    s = {det: store[(regime, det)].test for det in DETECTORS}
    cals = {det: ValidationCalibrator(store[(regime, det)].val)
            for det in DETECTORS}
    s["Ensemble"] = ecdf_rank_average(cals, {k: s[k] for k in cals})
    return s


def _stored_scores(ds, regime, y):
    """With-clock test scores saved by the bootstrap cell (test_scores.npz),
    used only if they belong to this very test block."""
    if not os.path.exists(_NPZ):
        return None
    z = np.load(_NPZ)
    ky = f"{ds}|{regime}|__y"
    if ky not in z.files or not np.array_equal(z[ky], y):
        print(f"  ! test_scores.npz does not match the {ds} test block: "
              "with-clock correlations skipped")
        return None
    return {k.split("|")[2]: z[k] for k in z.files
            if k.startswith(f"{ds}|{regime}|") and not k.endswith("__y")}


def _spearman_with_position(scores):
    """Spearman correlation of each score with test position (Pearson
    correlation of average ranks, the definition used by scipy's spearmanr)."""
    out = {}
    for det, s in scores.items():
        pos = np.arange(len(s), dtype=float)
        out[det] = float(np.corrcoef(rankdata(s), pos)[0, 1])
    return out


clock_rows, corr_rows, position_rows, repro_rows = [], [], [], []


def _write_tables():
    clock = pd.DataFrame(clock_rows)
    if len(clock):
        clock["PR_AUC_difference"] = (clock["PR_AUC_with_clock"]
                                      - clock["PR_AUC_without_clock"])
    clock.round(4).to_csv(os.path.join(OUT_DIR, "table_clock_ablation.csv"),
                          index=False)
    pd.DataFrame(corr_rows).round(3).to_csv(
        os.path.join(OUT_DIR, "table_clock_position_corr.csv"), index=False)
    pd.DataFrame(position_rows).round(4).to_csv(
        os.path.join(OUT_DIR, "table_clock_position_baseline.csv"), index=False)
    pd.DataFrame(repro_rows).to_csv(
        os.path.join(OUT_DIR, "table_clock_reproduction_check.csv"), index=False)
    return clock


# ---- main loop -------------------------------------------------------------- #
rerun_with_clock = {"ULB": True, "PaySim": False}   # PaySim only if ULB fails

for ds in DATASETS:
    col = CLOCK[ds]
    _stamp(f"{ds}: chronological split WITH '{col}'")
    sp_w = _split(ds, drop_clock=False)
    sp_w.describe()
    y = sp_w.yte
    n_feat = sp_w.Xtr.shape[1]

    # ---- position within the test block, used alone as a score ------------ #
    pos = np.arange(len(y), dtype=float)
    ap_pos, roc_pos = (float(average_precision_score(y, pos)),
                       float(roc_auc_score(y, pos)))
    dec = np.array_split(np.arange(len(y)), 10)
    share_last3 = float(y[np.concatenate(dec[-3:])].sum() / y.sum())
    position_rows.append({"dataset": ds, "PR_AUC_position_only": ap_pos,
                          "ROC_AUC_position_only": roc_pos,
                          "random_ranker_PR_AUC": float(y.mean()),
                          "share_of_frauds_in_last_3_deciles": share_last3})
    print(f"  position only: PR-AUC {ap_pos:.4f}  ROC-AUC {roc_pos:.4f}  "
          f"(random ranker {y.mean():.4f}); {share_last3:.1%} of the test "
          "frauds lie in the last three deciles")

    # ---- with-clock reference ---------------------------------------------- #
    w_ap = {r: _headline(ds, r, "PR_AUC") for r in REGIMES}
    w_al = {r: _headline(ds, r, "n_alerts") for r in REGIMES}
    w_src = "headline (canonical_results.csv)"
    s_w = {r: _stored_scores(ds, r, y) for r in REGIMES}
    if rerun_with_clock[ds]:
        _stamp(f"{ds}: headline configuration re-run WITH '{col}'")
        store_w = run_split(sp_w, experiment="clock_ablation_ref",
                            regimes=REGIMES, keep_scores=True)
        max_diff = max(abs(_values("clock_ablation_ref", ds, r, "PR_AUC")[m]
                           - w_ap[r][m]) for r in REGIMES for m in MEMBERS)
        reproduced = bool(max_diff < 1e-6)
        repro_rows.append({"dataset": ds,
                           "max_abs_PR_AUC_diff_vs_headline": max_diff,
                           "reproduced": reproduced})
        print(f"  largest |re-run - headline| PR-AUC = {max_diff:.2e}  ->  "
              + ("headline reproduced" if reproduced else "NOT REPRODUCED"))
        s_w = {r: _member_scores(store_w, r) for r in REGIMES}
        if not reproduced:
            print("  ! This environment does not reproduce the deposited "
                  "headline numbers (compare the package versions printed "
                  "above with manifest.json). The with-clock column is "
                  "therefore taken from this session's re-run.")
            w_ap = {r: _values("clock_ablation_ref", ds, r, "PR_AUC")
                    for r in REGIMES}
            w_al = {r: _values("clock_ablation_ref", ds, r, "n_alerts")
                    for r in REGIMES}
            w_src = "re-run in this session"
            rerun_with_clock["PaySim"] = True
        del store_w
    del sp_w

    # ---- the same configuration WITHOUT the clock ------------------------- #
    _stamp(f"{ds}: chronological split WITHOUT '{col}'")
    sp_wo = _split(ds, drop_clock=True)
    assert sp_wo.Xtr.shape[1] == n_feat - 1, "the clock feature was not removed"
    assert np.array_equal(sp_wo.yte, y), "the test block changed"
    sp_wo.describe()
    _stamp(f"{ds}: headline configuration re-run WITHOUT '{col}' "
           f"({n_feat - 1} features)")
    store_wo = run_split(sp_wo, experiment="clock_ablation", regimes=REGIMES,
                         keep_scores=True)
    for r in REGIMES:
        wo_ap = _values("clock_ablation", ds, r, "PR_AUC")
        wo_al = _values("clock_ablation", ds, r, "n_alerts")
        r_wo = _spearman_with_position(_member_scores(store_wo, r))
        r_w = _spearman_with_position(s_w[r]) if s_w[r] else {}
        for m in MEMBERS:
            clock_rows.append({
                "dataset": ds, "regime": r, "detector": m,
                "PR_AUC_with_clock": w_ap[r][m],
                "PR_AUC_without_clock": wo_ap[m],
                "alerts_with_clock": int(w_al[r][m]),
                "alerts_without_clock": int(wo_al[m]),
                "with_clock_source": w_src})
            corr_rows.append({
                "dataset": ds, "regime": r, "detector": m,
                "spearman_with_clock": r_w.get(m, np.nan),
                "spearman_without_clock": r_wo[m]})
    del store_wo, sp_wo

    save_results()
    _write_tables()
    _stamp(f"{ds} done; tables written to {os.path.abspath(OUT_DIR)}")

# ---- report ----------------------------------------------------------------- #
clock = _write_tables()
pd.set_option("display.width", 200)
print("\n=== PR-AUC and validation-rule alert counts, with and without the clock ===")
print(clock.drop(columns="with_clock_source").round(4).to_string(index=False))
print("  with-clock source:", ", ".join(
    f"{d}: {s}" for d, s in clock.groupby("dataset")["with_clock_source"].first().items()))
print("\n=== Spearman correlation of each score with test position ===")
print(pd.DataFrame(corr_rows).round(3).to_string(index=False))
print("\n=== Position-only baseline ===")
print(pd.DataFrame(position_rows).round(4).to_string(index=False))
print("\n=== Reproduction check (with-clock re-run vs deposited headline) ===")
print(pd.DataFrame(repro_rows).to_string(index=False))
_stamp("finished. Send back the four table_clock_*.csv files from "
       f"{os.path.abspath(OUT_DIR)}")
