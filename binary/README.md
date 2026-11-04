# Binary simulations

Use Python 3.10.6 with `requirements.txt`, in an environment separate from the continuous branch. NumPy 1.26.4 is needed to reproduce the reported random streams; passing tests under another version does not reproduce the paper's values.

From this directory:

```sh
python -m pip install -r requirements.txt
python smoke.py --out-dir runs_smoke
python -m pytest tests -q
```

The smoke example uses a small design (about 7,200 observations, 100 covariates, nine centers, `SIG_Y=2`; the formal design uses 1). It learns center labels, runs the dual screening and fits RA, AIPW and PSM; the oracle arm is a reference. It is not the paper design.

## Main simulation, sensitivity analyses and hypothesis tests

Round 1 (`--stage r1`) holds the 40 main design points; round 2 (`--stage r2`, the default) holds the hypothesis-test points (C001-C012 at gamma = 0, S01-S13 at a small gamma) and the sensitivity points (P041-P054 and U001-U020). Each point has 500 parameter replicates and one data replicate.

```sh
python src/run_formal.py --stage r1 --out-dir runs/r1 --workers 4
python src/run_formal.py --stage r1 --out-dir runs/r1 --verify-complete
python src/summarize_formal.py --out-dir runs/r1 --planned-reps 500
```

Before round 2, copy `config/gamma_small_frozen.json` to `results/gamma_small_frozen.json`; it holds the small gamma (0.045) of the S points chosen by `src/calibrate_gamma_small.py`. Without it the S points do not run.

```sh
mkdir -p results && cp config/gamma_small_frozen.json results/
python src/run_formal.py --stage r2 --out-dir runs/r2 --workers 4
python src/run_formal.py --stage r2 --out-dir runs/r2 --verify-complete
python src/summarize_formal.py --out-dir runs/r2 --planned-reps 500
```

These are server workloads. `--points`, `--param-reps` and `--data-reps` partition a run; structure identities and seeds come from the full design tables and do not change for a subset. The arms of every point are listed in the `execution_methods` column of the run tables under `config/`.

## Ablation and pooled comparators

Both runners regenerate each unit from its design row and recorded seeds and reuse the frozen center labels of the formal units (`config/labels/labels_<stage>.csv.gz`) instead of learning them again.

```sh
python src/run_ablation.py --stage r1 --out-dir runs_ablation/r1 --workers 4 --allow-recompute
python src/run_pooled.py   --stage r1 --out-dir runs_pooled/r1   --workers 4 --allow-recompute
```

Repeat with `--stage r2` for the sensitivity points. The ablation runs `dual_icp`, `dual_no_icp` and `dual_global_icp`; the pooled runner runs `pooled_lasso_fp`, `pooled_pds_fp`, `pooled_oal_fp`, `hdps_fp` and `oracle_safe_pooled`, each selected and estimated over all centers. Selection caches are not distributed, so `--allow-recompute` computes the selections with the same functions. `--gate` compares with the original reference units, which are not distributed, and fails without them. A data replicate whose regenerated `ate_do` differs from the stored value is recorded in `regeneration_checks`; this happens with another NumPy version.

## Merging

`src/summarize_formal.py` summarizes one run directory. For runs split over several machines:

| Output | Tool |
|---|---|
| formal runs (rounds 1 and 2) | `merge/r1/` and `merge/r2/`: `scan_units.py`, then `merge_tables.py` (gates for missing or duplicate units, design revisions and denominators; shard identifiers through `--machines`) |
| ablation | `python -I merge/merge_ablation.py --source ablation --inputs runs_ablation --out merged_ablation` |
| exposure-side and outcome-side variants | `python -I merge/merge_ablation.py --source formal --inputs runs --out merged_ablation` |
| pooled comparators | `python -I merge/merge_pooled.py --inputs runs_pooled --out merged_pooled` |

The merge tools read run directories, `.tar.gz` archives or `.zip` files containing them. `tools/` holds the extraction of frozen labels and caches from stored unit archives, the round-2 table builder and a data fingerprint for comparing environments.
