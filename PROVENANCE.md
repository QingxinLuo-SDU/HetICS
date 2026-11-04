# Code and result provenance

This repository is a release copy of the code behind the simulations reported in the paper. It contains the arms and estimators reported in the paper. Result archives and result tables are not included.

## Output versions

Unit records carry the implementation that produced them.

| Experiment | Output version of the reported runs | Output version of this release |
|---|---|---|
| Continuous simulation, ablation and pooled comparators | earlier versions of `RA_VERSION` | `cont-ra-v2` (`continuous/ra_core.py`) |
| Continuous hypothesis tests | earlier versions of `CAL_RA_VERSION` | `cont-cal-ra-v3` (`continuous/calibration_ra.py`) |
| Binary round 1 | `binary-formal-v5-impl-20260927-perf-sigy1-icp1e3` | `binary-formal-r2-impl-20261016-perf-sigy1-icp1e3-sgamma` |
| Binary round 2 | `binary-formal-r2-impl-20261016-perf-sigy1-icp1e3-sgamma` | same |
| Binary ablation | `binary-ablation-v6-20261028` | same |
| Binary pooled comparators | `binary-fully-pooled-v2-20261028` and, for `hdps_fp`, `binary-fully-pooled-fpx-v1-20261031` | `binary-pooled-v3-20261104` (all arms in one run) |

`binary/merge/merge_ablation.py` and `merge_pooled.py` accept the reported versions and the current one.

## Differences from the code of the reported runs

* The arms of the run tables (`execution_methods`) and the code are restricted to the arms reported in the paper, so the tables and their checksums differ from those of the runs; all other columns, and hence the structures and seeds, are unchanged. Each arm uses its own deterministic or locally seeded streams, so the remaining arms give the same results; see [VALIDATION.md](VALIDATION.md).
* The partial Chow test and BH adjustment are collected in `chow_test.py` in both branches, with the same computations. In the binary branch `step6_methods.knockoff_family_cluster` now returns the candidates, statistics and p-values directly.
* The hdPS proxy of the binary pooled comparators runs with the other pooled comparators in `run_pooled.py`; in the reported runs its selections came from the stored formal units.
* The binary diagnostics count selections before the ICP from records read back from JSON (`n_pre_icp` in `summarize_formal.py`).
* Binary runners count a failed base group as a failed execution even when the group has no unit records.
* The binary formal runner has an explicit `--stage r1/r2` option (default r2); round 1 uses its own design table, structure names and seeds.
* Machine-specific paths and server identities are removed from defaults, examples, metadata and the source identifiers of the frozen labels.

## Frozen simulation labels

`binary/config/labels/labels_r1.csv.gz` and `labels_r2.csv.gz` originate from the formal units of rounds 1 and 2 and hold replication identities, parameter and data seeds, learned center assignments, availability status and check values. The ablation and pooled runners reuse these assignments instead of rerunning the grouping stage. `tools/extract_frozen_labels.py` reads the unit records from result archives, which are not included. The tables are not clinical data or Monte Carlo summaries.

Selection caches and reference units are not included. With `--allow-recompute` the runners compute missing selections with the same functions; `--gate` compares with the original references and fails without them.

`binary/config/gamma_small_frozen.json` is the frozen record of the small gamma (0.045) chosen by `src/calibrate_gamma_small.py`; its `run_table_sha256` field refers to the run table at the time of calibration.

## Licensing and method references

The project code, including `seam_clustering.py` and `binary/vendor/bniseam_test/`, is distributed under the MIT license. `binary/vendor/patches/` records the changes of the label package made for the full-sample mode and its performance. Python dependencies are installed separately and keep their own licenses. The sEAM reference in the main README is a methodological citation; this repository does not contain the authors' software.
