# =============================================================================
# CELL 2  --  PaySim without the clock AND without the balance residuals
# Run in the SAME kernel, right after ONE_CELL_clock_ablation (no restart).
# Completes the 2 x 2 design clock x residuals of Table 13:
#   step + residuals (headline) | residuals removed (Table 13) |
#   step removed (clock ablation) | both removed (this cell)
# Runtime: about 15-25 min. Rows go to canonical_results.csv under
# experiment == "clock_balance_ablation"; headline rows are not touched.
# =============================================================================
for _n in ["_split", "run_split", "_values", "_headline", "MEMBERS", "REGIMES",
           "save_results", "clear_raw_cache", "_stamp"]:
    if _n not in globals():
        raise RuntimeError("Run ONE_CELL_clock_ablation first, in this same "
                           "kernel (no restart), then this cell.")
if pivot("clock_ablation", "PR_AUC", dataset="PaySim").empty:
    raise RuntimeError("ONE_CELL_clock_ablation did not finish PaySim in this "
                       "kernel; let it finish, then run this cell.")

_T0 = time.time()
PAYSIM_BALANCE_FEATURES = False              # read by _prep_paysim at call time
try:
    _stamp("PaySim: chronological split WITHOUT 'step' and WITHOUT the "
           "balance residuals")
    sp_nn = _split("PaySim", drop_clock=True)
    assert sp_nn.Xtr.shape[1] == 10, f"expected 10 features, got {sp_nn.Xtr.shape[1]}"
    sp_nn.describe()
    _stamp("PaySim: headline configuration re-run with 10 features")
    run_split(sp_nn, experiment="clock_balance_ablation", regimes=REGIMES,
              keep_scores=False)
    del sp_nn
finally:
    PAYSIM_BALANCE_FEATURES = True           # restore the default
    clear_raw_cache()
save_results()

# ---- the 2 x 2 table --------------------------------------------------------- #
_abl_path = os.path.join(OUT_DIR, "table_paysim_ablation.csv")
_abl = pd.read_csv(_abl_path) if os.path.exists(_abl_path) else None
if _abl is None:
    print(f"  ! {_abl_path} not found: the 'residuals removed' column is left empty")
grid = []
for r in REGIMES:
    head = _headline("PaySim", r, "PR_AUC")
    no_clock = _values("clock_ablation", "PaySim", r, "PR_AUC")
    no_both = _values("clock_balance_ablation", "PaySim", r, "PR_AUC")
    no_res = ({} if _abl is None else
              _abl[_abl.regime == r].set_index("detector")["without_delta"].to_dict())
    for m in MEMBERS:
        grid.append({"regime": r, "detector": m,
                     "step_and_residuals_headline": head[m],
                     "residuals_removed": no_res.get(m, np.nan),
                     "step_removed": no_clock[m],
                     "both_removed": no_both[m]})
grid = pd.DataFrame(grid).round(4)
grid.to_csv(os.path.join(OUT_DIR, "table_clock_balance_2x2.csv"), index=False)
print("\n=== PaySim test PR-AUC: clock x balance residuals ===")
print(grid.to_string(index=False))
_stamp("finished. Send back table_clock_balance_2x2.csv from "
       f"{os.path.abspath(OUT_DIR)}")
