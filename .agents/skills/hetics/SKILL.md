---
name: hetics
description: Apply HetICS to user-supplied multicenter tabular data for center grouping, invariant covariate selection, and causal effect estimation. Use HetICS-RA for continuous exposure and outcome, or HetICS-AIPW for binary exposure and outcome, with the existing HetICS repository implementations.
---

# Apply HetICS

Use the existing implementation in the HetICS checkout. Produce a runnable analysis script and its results in the user's analysis directory. Keep study data and results outside the code repository. Do not modify the scientific algorithms to make a dataset fit.

## Locate the code and prepare inputs

Find the checkout containing `continuous/ra_core.py` and `binary/src/aipw_sandwich.py`; when this skill is inside `.agents/skills/hetics/`, that checkout is its repository root. If the skill was copied elsewhere, locate the user's checkout or ask for its path. Read the repository `README.md` for the branch-specific Python environments. Run the two branches in separate Python processes: their module names overlap.

From the user's data dictionary and instructions, establish the exposure, outcome, center identifier, eligible baseline covariates, analysis population, and intended effect scale. Ask only for missing decisions. Confirm missing-value handling, category encoding, and any unit conversion before applying them. Do not choose a treatment cutpoint or change outcome coding silently. Survival outcomes, repeated observations per person, and mixed exposure/outcome types need a study-specific adaptation rather than automatic routing to one of these recipes.

Prepare aligned arrays, preserving the row selection, feature names, and original-center-to-integer mapping:

| Input | Required content |
|---|---|
| `F` | Finite numeric array, shape `(n, p)`, of candidate baseline covariates |
| `X` | Exposure vector, shape `(n,)`; continuous for the continuous recipe, 0/1 for binary |
| `Y` | Outcome vector, shape `(n,)`; continuous for the continuous recipe, 0/1 for binary |
| `env` | Integer center codes, shape `(n,)`, with no missing identifiers |
| `K` | Prespecified number of learned center groups; the paper uses 3, not an automatic choice for a new study |

Check shapes, numeric finiteness, binary coding where applicable, and constant/duplicate columns before fitting. Exclude identifiers, exposure, outcome, post-exposure variables, intercepts, and center dummy columns from `F`; the estimators add center effects. Resolve rank deficiency or encoding problems explicitly rather than letting a numerical fallback define preprocessing.

Choose `K` and, for continuous data, `sparsity` and `seed` from the study's analysis plan. Require integer `1 <= K < number_of_centers`; the continuous recipe additionally needs `K >= 2`, at least `2 * K` centers, and integer `1 <= sparsity <= p`. Do not derive these choices from simulated causal-variable counts; as a reference, the paper's simulations use a `sparsity` of about 0.20 p. Record them and the repository commit. Start with one numerical-library thread and measure resource use before increasing parallelism.

Both recipes use fixed-X knockoffs: each learned group needs more than `2 * p + 1` observations for the augmented design and the BIC noise-variance calculation with its intercept. Check this after grouping. Do not silently switch to Model-X, prefilter variables, or discard small groups to satisfy it. Grouping must cover every included center. Preserve empty selected sets; they are different from a failed screening calculation. With `q=0.10`, knockoff+ selects either no covariate or at least `ceil(1/q) = 10`, so one side can return an empty candidate set when few covariates carry signal; this comes from the threshold, not from an error.

In the snippets below, `repo_root` is a `pathlib.Path` to the checkout, and the arrays and analysis parameters have already been validated. For binary data, also set `analysis_id` to a stable run name such as `"CUSTOM_DATA"`. Set numerical-library thread limits before importing NumPy. Follow the relevant recipe only.

## Installed package

`pip install` of this repository provides `hetics.fit_ra(F, X, Y, centers, K=...)` and `hetics.fit_aipw(F, X, Y, centers, K=...)`, which run the two recipes below, accept arbitrary center labels and check their inputs. Prefer them when the analysis needs nothing beyond the recipe; the choices and reporting rules of this skill apply to them as well. `tests/test_package.py` checks that both give the same results as the recipes.

## Continuous data: HetICS-RA

Read `continuous/seam_clustering.py`, `step6x_methods.py`, `seam_support_methods.py`, `ra_core.py`, and `random_effects_gls.py` when preparing the analysis script. `fit_once()` generates simulations; `fit_existing_selection_pipeline()` requires simulation truth. Neither is a user-data entry point.

```python
import sys
import numpy as np

sys.path.insert(0, str(repo_root / "continuous"))
from seam_clustering import run_seam_clustering
from step6x_methods import knockoff_candidates
from seam_support_methods import icp_detail
from ra_core import extra_estimators
from random_effects_gls import fit_reml_gls

grouping = run_seam_clustering(
    F, Y, env, K=K, sparsity=sparsity, random_state=seed,
    max_iter=300, tol=1e-4, lipschitz_method="exact",
)
centers = np.unique(env)
labels = grouping["center_cluster_hat"]  # Same order as centers.
clusters = []
for k in range(K):
    members = centers[labels == k]
    take = np.isin(env, members)
    if len(members) < 2 or take.sum() <= 2 * F.shape[1] + 1:
        raise ValueError(f"Group {k} has insufficient centers or observations")
    Fk, Xk, Yk, ek = F[take], X[take], Y[take], env[take]
    sx, sy, candidate_diagnostics = knockoff_candidates(Fk, Yk, Xk, ek, q=0.10)
    screening = icp_detail(Fk, Yk, ek, sx | sy, len(members), alpha=0.001)
    if screening["status"] not in ("ok", "empty_candidate"):
        raise RuntimeError(f"Group {k}: {screening['status']}")
    selected = screening["kept_strict"]
    ra = extra_estimators(Xk, Yk, Fk[:, selected], ek, gps=False)["ra"]
    if ra["status"] != "ok" or not np.isfinite(ra["se"]) or ra["se"] <= 0:
        raise RuntimeError(f"Group {k}: {ra}")
    clusters.append(dict(group=k, centers=members, selected=selected, ra=ra,
                         candidates_x=sx, candidates_y=sy,
                         candidate_diagnostics=candidate_diagnostics,
                         screening=screening))

overall = fit_reml_gls(
    [row["ra"]["theta"] for row in clusters],
    np.diag([row["ra"]["se"] ** 2 for row in clusters]),
)
```

The dual rule takes the union of the two candidate sets, then performs one ICP calculation. Use `kept_strict`; `kept_official` adds candidates back when every candidate is rejected. The recipe explicitly uses the paper's `q=0.10` and `alpha=0.001`; the latter is a tunable threshold, not a universal causal-validity guarantee.

Report `grouping["diagnostics"]`, including convergence and iteration count. RA estimates the exposure coefficient with center fixed effects, center-clustered CR1 standard errors, and `ci_t` using the group's center count. For an overall effect, require every expected group to succeed, at least two groups, and inspect both `overall.status` and `overall.converged`. Preserve the REML weights, heterogeneity, and diagnostics. Do not substitute `ra_core.aggregate()` for arbitrary `K`: that helper assumes the simulation's three groups. Report `intervals_exploratory` as exploratory intervals. REML aggregation targets its model-based overall effect; do not relabel it as a person-weighted ATE when group effects differ.

## Binary data: HetICS-AIPW

Read `binary/src/label_stage.py`, `screening_comparison.py`, `screening_bridge.py`, and `aipw_sandwich.py`. `run_formal.run_unit()` builds and evaluates simulated outcomes; do not fabricate its design rows or truth fields for user data.

```python
import json
import sys
import numpy as np

root = repo_root / "binary"
for folder in ("src", "vendor", "vendor/step6_pkg"):
    sys.path.insert(0, str(root / folder))
from bniseam_test.multiview_types import ObservedDataView
from label_stage import algorithm_task, learn_labels
from screening_bridge import load_screening
from screening_comparison import screen_family
from aipw_sandwich import fit_partition_aipw

config = json.loads((root / "config/formal_r2.json").read_text(encoding="utf-8"))
m6 = load_screening(str(root / "vendor/step6_pkg"))
m6.ALPHA_ICP = float(config["icp_alpha"])
observed = ObservedDataView(F, X, Y, env, np.unique(env))
task = algorithm_task(analysis_id, 1, 1)
labels = learn_labels(observed, task, k_components=K)
if labels["center_map"] is None:
    raise RuntimeError(labels["label_status"])
for k in set(labels["center_map"].values()):
    members = [c for c, group in labels["center_map"].items() if group == k]
    if np.isin(env, members).sum() <= 2 * F.shape[1] + 1:
        raise ValueError(f"Group {k} is too small for fixed-X knockoffs")
screening = screen_family(
    F, X, Y, env, labels["center_map"], "fixedx_bic",
    m6=m6, task=task, config=config,
)
selected = screening["directions"]["dual_icp"]
if selected["status"] != "complete":
    raise RuntimeError(selected["errors_by_cluster"])
effect = fit_partition_aipw(
    F, X, Y, env, labels["center_map"], selected["selected_by_cluster"],
    backend=config["logistic_backend"],
)
```

Record `analysis_id`, the three seeds in `task`, and `labels["label_config"]`. The identifier need not belong to the registered simulation design. The label implementation retains its full-sample fitting, support size 25, stability guard, and eligibility rules; inspect those settings for the study. A maximum-iteration result can be eligible. Use `center_map` availability as the existing eligibility decision; do not replace it with audit-only labels when the algorithm abstains. Flag groups with only one center: the existing screening code skips ICP there, so cross-center invariance has not been assessed for that group.

Use the top-level `config["icp_alpha"]` (0.001 in the current release). `fixedx_bic` uses candidate union followed by one ICP calculation. Label-model supports are not the final dual selection sets.

`fit_partition_aipw` returns the person-weighted ATE on the **risk-difference** scale, including its joint covariance and aggregation. Do not exponentiate it or label it an odds ratio. Inspect `status`, `point_status`, and `se_status` separately: a point estimate can exist without an available SE. The current successful SE status is `candidate_fixed_center_sandwich`; retain that designation. Report separation, rank deficiency, missing treatment arms, and propensity-score failures without silently clipping scores, adding a penalty, changing models, or dropping groups.

## Deliver the application

Save the analysis script, input-column and center mappings, preprocessing decisions, parameters and seeds, grouping diagnostics, candidate and final selections with original feature names, group sizes, group effects, overall effect, SE/intervals where available, and failure states. Selection indices are zero-based columns of `F`, not original table column numbers. Keep patient-level data local; report summaries without identifying records.

Validate the selected branch with a small synthetic example before applying the script to the full dataset. The repository's `continuous/smoke.py` and `binary/smoke.py` check installation, but do not replace validation of the actual user-data adapter. A successful run checks computation; it does not establish exchangeability, positivity, appropriate covariate timing, or valid causal interpretation for a new study. For paper simulation reproduction, use the repository README instead of replacing the user's data with generated samples.
