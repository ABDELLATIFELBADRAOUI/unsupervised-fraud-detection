#!/usr/bin/env python3
"""Validation-based hyperparameter selection of Table 18 (and the selected block of Table 3).

Reads  results/table_hyperparameter_sensitivity.csv  (validation and test PR-AUC, 4 decimals)
Writes results/phase0/table16_hyperparameter_selected.csv and results/phase0/table3_selected_block.csv

Rule (run manifest): maximise validation PR-AUC; configurations within 1e-4 of the best are tied,
and the simplest tied configuration is reported. The comparison subtracts 1e-9 so that a gap of
exactly 1e-4 between two 4-decimal values is treated as a tie whatever the floating-point rounding
(v1.2.0 excluded one such tie: PaySim, label-free, Isolation Forest, 0.0022 against 0.0021).
Usage: python scripts/select_hyperparameters.py   (from the repository root)
"""
import os
import numpy as np
import pandas as pd

TIE, EPS = 1e-4, 1e-9
sens = pd.read_csv(os.path.join("results", "table_hyperparameter_sensitivity.csv"))


def _num(v, default):
    try:
        return float(v) if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


def _complexity(r):
    """Lower = simpler; identical to the notebook's definition."""
    det = r["detector"]
    if det == "LOF":             return _num(r.get("n_neighbors"), 20)
    if det == "KMeans":          return _num(r.get("k"), 8)
    if det == "IsolationForest": return _num(r.get("n_estimators"), 200)
    if det == "OneClassSVM":
        g = r.get("gamma")
        return 0.0 if (pd.isna(g) or str(g) == "scale") else 1.0 + _num(g, 0.0)
    if det == "DBSCAN":
        e = _num(r.get("eps_scale"), 1.0); m = _num(r.get("min_samples"), 10)
        return abs(np.log(e)) + abs(m - 10) / 100.0
    return 0.0


rows = []
for key, g in sens.groupby(["dataset", "regime", "detector"]):
    g = g.copy(); g["complexity"] = g.apply(_complexity, axis=1)
    top = g.val_PR_AUC.max()
    tied = g[g.val_PR_AUC >= top - TIE - EPS].sort_values(["complexity", "config"])
    chosen = tied.iloc[0]
    rows.append({"dataset": key[0], "regime": key[1], "detector": key[2],
                 "selected_config": chosen.config,
                 "val_PR_AUC": chosen.val_PR_AUC, "test_PR_AUC": chosen.test_PR_AUC,
                 "test_min": g.test_PR_AUC.min(), "test_max": g.test_PR_AUC.max(),
                 "test_spread": g.test_PR_AUC.max() - g.test_PR_AUC.min(),
                 "n_tied_on_validation": len(tied),
                 "tied_configs": " | ".join(tied.config.astype(str).tolist())})
sel = pd.DataFrame(rows).round(4)
out = os.path.join("results", "phase0"); os.makedirs(out, exist_ok=True)
sel.to_csv(os.path.join(out, "table16_hyperparameter_selected.csv"), index=False)
sel.pivot_table(index="detector", columns=["dataset", "regime"], values="selected_config",
                aggfunc="first").to_csv(os.path.join(out, "table3_selected_block.csv"))
print(sel.drop(columns=["tied_configs"]).to_string(index=False))
