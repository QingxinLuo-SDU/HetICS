# HetICS

Simulation code for **Heterogeneity-Guided Cross-Center Invariant Covariate Selection for Causal Effect Estimation in Multi-Center Studies**.

**Qingxin Luo · Shandong University**

HetICS groups centers into clusters, selects covariates within each cluster by fixed-X knockoff screening on the exposure and the outcome followed by a partial Chow test of cross-center invariance, and estimates the treatment effect with the selected adjustment sets. The primary estimators are **HetICS-RA** for continuous outcomes and **HetICS-AIPW** for binary outcomes. The supplementary estimators are Robinson's estimator and generalized propensity score weighting (with and without truncation) for continuous outcomes, and regression standardization and propensity score matching for binary outcomes.

## Installation and use

```sh
pip install git+https://github.com/QingxinLuo-SDU/HetICS
```

or `pip install .` from a checkout. The package needs Python 3.10 or later and installs NumPy, SciPy, scikit-learn and statsmodels.

```python
import hetics

# continuous exposure and outcome: HetICS-RA
result = hetics.fit_ra(F, X, Y, centers, K=3)
result["clusters"][0]["selected"], result["clusters"][0]["theta"], result["overall"]["mu"]

# binary exposure and outcome: HetICS-AIPW, on the risk-difference scale
result = hetics.fit_aipw(F, X, Y, centers, K=3, analysis_id="my-study")
result["estimate"], result["se"], result["ci95"], result["clusters"][0]["selected"]
```

`F` is the array of baseline covariates (rows are individuals; no intercept or center dummies), `X`, `Y` and `centers` are vectors, and `K` is the prespecified number of center clusters. Selected covariates are column indices of `F`; pass `feature_names` to get names as well. Both functions check their inputs and raise an error when a cluster is too small for fixed-X knockoffs (more than 2p + 1 observations are needed); `fit_aipw` also raises one when the label algorithm abstains. Knockoff+ at the default `q=0.10` selects no covariate or at least 10, so a study with few covariates or weak signals can end with an empty adjustment set; the functions warn when this happens. Limit numerical libraries to one thread when running many analyses in parallel. See the docstrings for all options.

To reproduce the paper's simulations, use the branch code and pinned environments below; the package is for applying the method and does not pin the package versions of the reported runs.

## Layout

```text
src/hetics/          installable package for applying HetICS to other data
continuous/          continuous DGP, clustering, screening, estimators, runners and tests
binary/
  src/               binary runners, estimators and summaries
  vendor/            label package and screening modules used by the binary runners
  config/            run tables, configurations and frozen center labels
  merge/             merging of run outputs
  tools/             extraction and design utilities
  tests/             binary tests
formal/              Lean 4 formalization of the core proof steps
.agents/skills/      agent skill for applying HetICS to user data
```

No patient data, application files, simulation results, per-replication output or figure scripts are included. The two frozen-label files under `binary/config/labels/` are simulation-derived inputs of the binary ablation and pooled-comparator runs.

## Environments

The two branches use separate environments.

| | Continuous | Binary |
|---|---|---|
| Python | 3.11 | 3.10 |
| NumPy | 1.26.4 | 1.26.4 |
| SciPy | 1.17.1 | 1.14.1 |
| pandas | 3.0.2 | 2.2.3 |
| scikit-learn | 1.8.0 | 1.6.1 |
| statsmodels | not required | 0.14.4 |

Install each branch's `requirements.txt` in its own environment, for example

```sh
python -m venv continuous/.venv
python -m pip install -r continuous/requirements.txt
```

and the same with Python 3.10 and `binary/requirements.txt`.

**The NumPy version changes the generated data for a given seed.** Use NumPy 1.26.4 to reproduce the reported values; other versions give different realizations.

## Small examples

```sh
cd continuous && python smoke.py
cd binary && python smoke.py
```

Each example runs the simulation and estimation code on a fixed small synthetic design in one process and checks the outputs. The continuous example covers the DGP, center clustering, screening, RA, Robinson and REML aggregation; the binary example covers center labels, screening and RA, AIPW and PSM. Neither writes paper results.

## Paper-to-code map

| Component | Code |
|---|---|
| Continuous DGP and design | `continuous/latent_cluster_dgp_formal.py`, `latent_cluster_dgp_specx.py`, `latent_cluster_dgp_specx_u.py`, `step6x_config.py`, `step6x_config_v4.py`, `v4_seeds.py` |
| Continuous clustering and selection | `continuous/seam_clustering.py`, `step6_methods.py`, `step6x_methods.py`, `seam_support_methods.py`, `chow_test.py` |
| Continuous estimation (RA, Robinson, GPS weighting, REML aggregation) | `continuous/ra_core.py`, `whole_center_plr.py`, `random_effects_gls.py`, `plr_transfer.py` |
| Continuous simulation, ablation and pooled comparators | `continuous/run_ra_cont.py`, `step6x_core_v4.py`, `pooled_selection.py` |
| Continuous hypothesis tests | `continuous/run_cal_ra.py`, `calibration_ra.py`, `calibration_selection_pipeline.py`, `hypothesis_calibration_config.py` |
| Binary DGP and design | `binary/src/formal_dgp.py`, `binary_with_u.py`, `design_table.py`, `binary/vendor/step6_pkg/` |
| Binary center labels and selection | `binary/vendor/bniseam_test/`, `binary/src/label_stage.py`, `screening_comparison.py` |
| Binary estimation (AIPW, RA, PSM) | `binary/src/aipw_sandwich.py`, `regression_adjustment.py`, `psm_ate.py` |
| Binary main simulation, sensitivity and hypothesis tests | `binary/src/run_formal.py`, `calibrate_gamma_small.py`, `summarize_formal.py`, `binary/merge/r1/`, `binary/merge/r2/` |
| Binary ablation | `binary/src/run_ablation.py`, `ablation_arms.py`, `binary/merge/merge_ablation.py` |
| Binary pooled comparators | `binary/src/run_pooled.py`, `pooled_arms.py`, `pooled_selection.py`, `binary/merge/merge_pooled.py` |

In the code the exposure, outcome and covariates are `X` (or `X_treat`), `Y` and `F`; the target and safe adjustment sets are `A_core_by_cluster` and `A_safe` (`observed_core_by_cluster` and `a_safe` in the binary branch). HetICS is the arm `dual_icp`, the exposure-side and outcome-side variants are `xside_icp` and `yside_icp`, and the ablation variants are `dual_no_icp` (no invariance selection) and `dual_global_icp` (one screening group of all centers). The pooled comparators select and estimate over all centers: `pooled_lasso_fp`, `pooled_pds_fp`, `pooled_oal_fp` and `hdps_fp`; `oracle_safe_pooled` is the pooled estimate with the safe set.

## Seeds and full-scale runs

Both branches use a parameter seed, which fixes a parameter realization, and a data seed, which draws a sample given those parameters; separate streams isolate fold and algorithm randomness. Continuous seeds derive from `numpy.random.SeedSequence` in `continuous/v4_seeds.py`; binary structure identities and seeds are defined in `binary/src/design_table.py` and `binary/vendor/bniseam_test/design.py`, and the frozen-label files keep the original replication identities and seeds.

The continuous grid and U sensitivity contain 86 design points with 1,000 replications each; the continuous hypothesis tests use 10 parameter seeds x 100 data seeds per point. The binary design uses 500 replications per point. These are server workloads; choose the number of workers from the measured memory per worker and limit numerical libraries to one thread.

### Continuous

From `continuous/`:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SCENA_BETA_ANCHOR=1.0,1.25
SCENA_EXPERIMENT=grid python run_ra_cont.py run --out runs_ra --workers 4
SCENA_EXPERIMENT=grid python run_ra_cont.py verify --out runs_ra
SCENA_EXPERIMENT=u07 python run_ra_cont.py run --out runs_ra --workers 4
python run_ra_cont.py combine --runs runs_ra --dest combined_ra
```

`--buckets lo-hi` splits a run across machines. For the hypothesis tests set `SCENA_EXPERIMENT=grid`, `SCENA_BETA_ANCHOR=2.0,3.0` and `SCENA_CAL_AXIS=m`, `n` or `p`, and use `run_cal_ra.py` with the same `run`, `verify` and `combine` subcommands. The value `2.0,3.0` selects the seed namespace of the calibration design; its coefficient band is 1.0-1.25.

### Binary

See [binary/README.md](binary/README.md).

## Tests

```sh
cd continuous && SCENA_BETA_ANCHOR=1.0,1.25 SCENA_EXPERIMENT=grid python -m pytest tests -q
cd binary && python -m pytest tests -q
pip install ".[test]" && python -m pytest tests -q   # package, from the repository root
```

Some continuous tests require `SCENA_BETA_ANCHOR=2.0,3.0` and are skipped otherwise. See [VALIDATION.md](VALIDATION.md).

## Agent skill

[.agents/skills/hetics/SKILL.md](.agents/skills/hetics/SKILL.md) tells a coding agent how to apply HetICS-RA and HetICS-AIPW to a user's own multi-center data, including the checks to make before and after fitting.

## Formal proofs

[formal/README.md](formal/README.md) gives the Lean and Mathlib versions, the build commands, the theorem map and the statistical premises of the formalized steps.

## License and citation

MIT, copyright 2026 Qingxin Luo. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff). Separately installed dependencies keep their own licenses.

The center clustering of the continuous branch builds on sEAM: Jing, K., Khalili, A., & Xu, C. (2025). Class-Specific Joint Feature Screening in Ultrahigh-Dimensional Mixture Regression. *Journal of the American Statistical Association*, **120**(552), 2473–2483. [doi:10.1080/01621459.2025.2468011](https://doi.org/10.1080/01621459.2025.2468011). This repository implements the method for its simulations and does not distribute the authors' software.
