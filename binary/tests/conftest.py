"""tests/conftest.py — paths and small fixtures.

The small dimensions serve local connectivity and numerical checks only; the formal runners always read the
run tables under config/ and accept no override.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "vendor"),
           os.path.join(_ROOT, "vendor", "step6_pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

PKG_ROOT = _ROOT

SMALL_PARAMS = {
    "n_centers": 6, "n_per_center": 300, "n_per_center_var": 0.1,
    "p_total": 60, "K": 3, "n_Z": 4, "n_causal_shared": 5,
    "n_causal_specific": 2, "n_spurious": 6,
    "rho_max": 3.0, "rho_min": 1.0, "delta_spec": 0.0, "nu_sd": 0.4,
    "alpha_ZX": 1.0, "alpha_ZY": 1.0, "sigma_Z": 1.0, "kappa": 0.0,
    "f_X": "linear", "f_Y": "tanh", "rho_block": 0.6, "rho_cross": 0.3,
    "beta_mag_lo": 2.0, "beta_mag_hi": 3.0, "signed_beta": True,
    "gamma_XY": 1.0, "centers_per_cluster": None,
}
SMALL_SEEDS = {"param_seed": 1_000_000_123, "data_seed": 2_000_000_456}

# Larger fixture: with SMALL_PARAMS the label pipeline abstains (primary_status=abstain_no_stable_candidate), so
# the end-to-end tests use dimensions that freeze labels; still far below the formal design.
LABELED_PARAMS = {**{k: v for k, v in SMALL_PARAMS.items()},
                  "n_centers": 9, "n_per_center": 800, "p_total": 100,
                  "n_Z": 8, "n_causal_shared": 10, "n_causal_specific": 3,
                  "n_spurious": 10}


@pytest.fixture(scope="session")
def sources():
    import formal_dgp
    return formal_dgp.load_sources()


@pytest.fixture(scope="session")
def small_continuous(sources):
    generate, _b6 = sources
    return generate(**SMALL_PARAMS, seed=SMALL_SEEDS["data_seed"],
                    param_seed=SMALL_SEEDS["param_seed"])


@pytest.fixture(scope="session")
def labeled_continuous(sources):
    generate, _b6 = sources
    return generate(**LABELED_PARAMS, seed=SMALL_SEEDS["data_seed"],
                    param_seed=SMALL_SEEDS["param_seed"])


@pytest.fixture(scope="session")
def small_group():
    return {
        "base_group_key": "TEST__p001__d01",
        "structure_id": "TEST", "param_rep": 1, "data_rep": 1,
        "param_seed": SMALL_SEEDS["param_seed"],
        "data_seed": SMALL_SEEDS["data_seed"],
        "u_epsilon_seed": 424242,
        "sig_x_spec": 0.6,
        "point_ids": ["TEST_P", "TEST_U"],
    }


def small_row(point_id: str, *, gamma_logor: float, has_u: bool,
              methods: list[str]) -> dict:
    """A test row with the fields of the run table."""
    row = {name: str(value) for name, value in SMALL_PARAMS.items()
           if name not in ("centers_per_cluster", "gamma_XY")}
    row.update({
        "point_id": point_id, "axis_membership": "TEST", "tier": "main",
        "scan_parameter": "test", "scan_value": "test",
        "gamma_logor": str(gamma_logor),
        "beta_mag_lo": "2.0", "beta_mag_hi": "3.0",
        "SIG_X": "1.2", "SIG_X_SPEC": "0.6", "SIG_Y": "2", "C0": "0",
        "gamma_XY_continuous": "1", "centers_per_cluster": "equal",
        "param_reps": "100", "data_reps": "10", "n_reps": "1000",
        "status": "test", "design_revision": "v4_frozen_u07_03_cal3_20260918",
        "scenario": ("unmeasured_U07_03" if has_u else "observed_confounders"),
        "has_unmeasured_U": ("true" if has_u else "false"),
        "u_shared_loading": ("0.7" if has_u else "0"),
        "u_noise_loading": ("0.3" if has_u else "0"),
        "lambda_UX": ("0.5" if has_u else "0"),
        "lambda_UY": ("0.5" if has_u else "0"),
        "calibration_member": "false",
        "report_panels": "main",
        "execution_methods": "|".join(methods),
        "n_methods": str(len(methods)),
        "estimators": "ra_standardized|aipw|psm_caliper",
        "n_estimators": "3",
        "source_v3_point_id": "", "pair_reference_point_id": "",
        "planned_estimator_attempts": str(len(methods) * 3 * 1000),
    })
    return row
