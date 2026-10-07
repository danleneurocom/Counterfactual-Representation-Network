# Smoke-test outputs (SYNTHETIC data — numbers are meaningless)

Produced by `scripts/isbi/smoke_synthetic.sh` style run on tiny synthetic volumes (48³, MedNeXt with
base_channels=8, 10 epochs, 2 seeds, CPU). They only demonstrate that every step of the pipeline runs
end to end and what each output looks like. Do not cite any value here.

* `check_data.png` – output of `trace_seg3d.check_data` (modality / orientation check)
* `table_final.md` – `trace_seg3d.summarize` main table
* `audit_utsw_to_brats_trace.md` – `trace_seg3d.audit` table
* `fig_cct.png` – `trace_seg3d.figures` (note the instability map is non-zero and concentrated on the lesion,
  unlike the old FiLM-based CCT)
* `example_summary_*.json`, `example_calib.json` – per-run evaluation summary and calibration file
