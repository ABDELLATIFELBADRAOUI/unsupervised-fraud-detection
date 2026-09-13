#!/usr/bin/env python3
"""Rebuild the formatted views used in the manuscript from the canonical frame.

Every file this script writes is derived from results/canonical_results.csv or
from a results/table_*.csv produced by the notebook; nothing is entered by hand.

    python scripts/make_manuscript_tables.py [--results results]

Outputs
    results/table_split_gap_<dataset>_<regime>.csv
    results/manuscript_tables/<name>.csv and .txt   (tab-separated, caption first)
"""
import argparse, os
import pandas as pd

REG = {"U": "U (label-free)", "N": "N (normal-only)", "O": "O (oracle, upper bound)"}


def write(out_dir, name, caption, df):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8", newline="") as f:
        f.write(caption + "\r\n\r\n")
        df.to_csv(f, sep="\t", index=False, lineterminator="\r\n")


def main(res):
    mt = os.path.join(res, "manuscript_tables")
    canon = pd.read_csv(os.path.join(res, "canonical_results.csv"))

    # ---- main results, one block per regime ------------------------------
    for ds in ("ULB", "PaySim"):
        t = pd.read_csv(os.path.join(res, f"table_main_{ds}.csv"))
        t["regime"] = t.regime.map(REG)
        for c in ("TP", "FP", "n_alerts"):
            t[c] = t[c].astype(float).round().astype(int)
        write(mt, f"table_main_{ds}",
              f"{ds}, chronological test set. All configurations share one decision rule: the "
              f"(1-rho) quantile of validation scores, rho=0.005. PR-AUC is the primary metric; "
              f"ROC-AUC is reported only as a complementary measure of ranking.", t)

    # ---- bootstrap intervals ---------------------------------------------
    ci = pd.read_csv(os.path.join(res, "table_bootstrap_ci.csv"))
    ci["regime"] = ci.regime.map(REG)
    ci["PR-AUC [95% CI]"] = [f"{r.PR_AUC:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]" for r in ci.itertuples()]
    write(mt, "table_bootstrap_ci_formatted",
          "Block bootstrap over temporal blocks of the chronological test set, 2000 replicates. "
          "Overlapping intervals do not support a ranking.",
          ci[["dataset", "regime", "detector", "PR-AUC [95% CI]", "ci_width"]])

    # ---- rolling origins --------------------------------------------------
    ro = pd.read_csv(os.path.join(res, "table_rolling_origin.csv"))
    ro["regime"] = ro.regime.map(REG)
    write(mt, "table_rolling_origin_formatted",
          "PR-AUC across rolling origins. Fold prevalence varies, so lift over each fold's own "
          "random baseline is reported alongside the raw value. cv above 0.5 indicates that the "
          "detector ordering is not stable.",
          ro.loc[ro.regime != REG["O"],
                 ["dataset", "regime", "detector", "pr_mean", "pr_std", "lift_mean", "cv", "n"]])

    # ---- reference-set selection rule -------------------------------------
    su = pd.read_csv(os.path.join(res, "table_subsampling_rule_effect.csv"))
    su["regime"] = su.regime.map(REG)
    su["random"] = su["random"].round(4); su["tail"] = su["tail"].round(4)
    write(mt, "table_subsampling_rule_effect_formatted",
          "Effect of the reference-set selection rule alone, at fixed hyperparameters, regime and "
          "split. Compare the magnitude of rule_effect with every other effect reported in the paper.",
          su)

    # ---- split decomposition ---------------------------------------------
    sd = pd.read_csv(os.path.join(res, "table_split_decomposition.csv"))
    sd["regime"] = sd.regime.map(REG)
    write(mt, "table_split_decomposition_formatted",
          "Test PR-AUC under three evaluation arms and the decomposition of the "
          "random-versus-chronological gap. Only the temporal component admits a reading as "
          "temporal leakage; the prevalence-matched arm is the mean of three replicate draws.",
          sd.sort_values(["dataset", "regime", "detector"]))

    # ---- the three thresholding rules -------------------------------------
    fz = pd.read_csv(os.path.join(res, "table_forensic_thresholds.csv"))
    write(mt, "table_forensic_thresholds_formatted",
          "The three candidate thresholding rules applied to the same scores. An alert count is "
          "meaningful only together with the rule that produced it; the tie ratio is reported "
          "separately, since a large tie mass distorts any quantile threshold.", fz)

    # ---- calibration drift -------------------------------------------------
    h = canon[(canon.experiment == "headline") & (canon.metric == "n_alerts")].copy()
    h["budget"] = (h.rho_used * h.n_test).round().astype(int)
    cd = pd.DataFrame({
        "dataset": h.dataset, "regime": h.regime.map(REG), "detector": h.detector,
        "rho": h.rho_used.round(5), "budgeted_alerts": h.budget,
        "observed_alerts": h.value.astype(int),
        "observed_over_budgeted": (h.value / h.budget).round(2),
    }).sort_values(["dataset", "regime", "detector"])
    write(mt, "table_calibration_drift",
          "Alert budget against realised alert volume on the chronological test set. The threshold "
          "is the (1-rho) quantile of the validation scores, so the budgeted count is rho x N_test; "
          "the ratio measures how far the score distribution has moved between the calibration "
          "window and the test window.", cd)

    # ---- split-gap tables behind the split_gap figures ---------------------
    s = canon[(canon.experiment == "split_comparison") & (canon.metric == "PR_AUC")]
    for ds in ("ULB", "PaySim"):
        for rg in ("N",):
            sub = s[(s.dataset == ds) & (s.regime == rg)]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="detector", columns="split_type",
                                  values="value", aggfunc="mean")
            piv = piv.sort_values("chronological", ascending=False)
            out = pd.DataFrame({
                "detector": piv.index,
                "random": piv["random"].round(4),
                "random_matched": piv["random_matched"].round(4),
                "chronological": piv["chronological"].round(4),
                "gap_prevalence": (piv["random"] - piv["random_matched"]).round(4),
                "gap_temporal": (piv["random_matched"] - piv["chronological"]).round(4),
            })
            out.to_csv(os.path.join(res, f"table_split_gap_{ds}_{rg}.csv"), index=False)

    print("written under", os.path.abspath(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    main(ap.parse_args().results)
