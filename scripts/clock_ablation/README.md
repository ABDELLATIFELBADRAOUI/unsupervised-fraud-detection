# Clock ablation (Table 13)

Two cells, run after `notebooks/revision_pipeline_v2_1.ipynb`:

1. `ONE_CELL_clock_ablation.py`: paste it as the last cell of the notebook, restart the
   kernel and run only this cell (it is self-contained; about 30–40 min). It re-runs the
   headline configuration without the absolute clock (`Time` on ULB, `step` on PaySim), and
   re-runs ULB with the clock as a reproduction check.
2. `CELL_2_paysim_no_step_no_residuals.py`: run it next, in the same kernel, without
   restarting (about 15–25 min). It removes both `step` and the balance residuals on PaySim,
   which completes the 2 × 2 design of Table 13.

Set `ULB_PATH` and `PAYSIM_PATH` in part A of the first cell. Rows are appended to
`canonical_results.csv` under `clock_ablation`, `clock_ablation_ref` and
`clock_balance_ablation`; headline rows are not modified. The cells also write
`table_clock_*.csv` and `manifest_clock_ablation.json`. They write to the notebook's output
folder (`revision_v2/`), committed in this repository as `results/`.
