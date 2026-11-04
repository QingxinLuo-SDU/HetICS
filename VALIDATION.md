# Validation

Date: **2026-11-04**. Local environment: Windows, Python 3.13.9, NumPy 2.3.4, SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.7.2, statsmodels 0.14.5 and pytest 8.3.5, with one numerical-library thread. The versions of the formal runs are in each branch's `requirements.txt`.

## Test suites

| Branch and setting | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Continuous, default environment | 97 | 8 | 0 |
| Continuous, `SCENA_BETA_ANCHOR=1.0,1.25`, `SCENA_EXPERIMENT=grid` | 93 | 12 | 0 |
| Continuous, `SCENA_BETA_ANCHOR=2.0,3.0`, `SCENA_EXPERIMENT=grid` | 97 | 8 | 0 |
| Binary | 153 | 0 | 0 |

Continuous tests that depend on the coefficient band or seed namespace are skipped under the other setting.

## Comparison with the code of the reported runs

On fixed small designs, in the same environment and with the same seeds, the selections and estimates of every retained arm were compared with those of the code of the reported runs; all values agree bit by bit:

- continuous: one replicate of `ra_core.fit_rep` (`dual_icp`, `xside_icp`, `yside_icp`, `oracle_min`, the ablation arms, the pooled comparators with `hdps_fp`, with RA, Robinson and both GPS weighting estimators) and one replicate of `calibration_ra.fit_rep`;
- binary: one formal unit (center labels, `dual_icp`, `xside_icp`, `yside_icp`, `oracle_min`, `oracle_reference`, with RA, AIPW and PSM), the ablation arms, and the pooled comparators with `hdps_fp`.

A formal binary unit run with the full arm set of the original run tables gave the same values for these arms as a run with the retained arms only.

## Other checks

- The binary static design checks pass for rounds 1 and 2 and the master table; `tools/build_round2_table.py` reproduces the round-2 table byte for byte.
- The tests cover round-specific design identities, frozen seeds at the first and last parameter replicates, missing or corrupt caches, reference comparisons, nonzero exit status for failed base groups, the merge tools, and the ablation and pooled runners without selection caches.
- The two code blocks of `.agents/skills/hetics/SKILL.md` run on synthetic data.

## Small examples

| Branch | Observations x covariates | Content |
|---|---:|---|
| Continuous | 1,200 x 80 | learned center clusters, screening, RA, Robinson and REML aggregation |
| Binary | about 7,400 x 100 | center labels on nine centers, dual screening and the oracle reference with RA, AIPW and PSM; `SIG_Y=2` (the formal design uses 1) |

## Limits

These checks cover execution and numerical consistency on small designs. They do not reproduce the full Monte Carlo study or replace a run under the pinned Python 3.10 and 3.11 environments; NumPy 2.3.4 gives other random streams than the NumPy 1.26.4 of the reported runs.
