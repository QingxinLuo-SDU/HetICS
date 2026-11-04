"""formal_dgp.py — continuous base layer, binary layer and evaluation truth.

The unit of scheduling is a base group (structure x param_rep x data_rep): a process generates the
continuous base data once, runs every point of the structure in turn and releases the data.

The truth goes only to the evaluation; no clustering, screening, ICP, model or SE receives it, and U enters
no arm, including the oracle.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

import binary_with_u as bu
import design_table as dt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
VENDOR_DIR = os.path.join(_PKG_ROOT, "vendor")
STEP6_DIR = os.path.join(VENDOR_DIR, "step6_pkg")


def _ensure_paths() -> None:
    for path in (VENDOR_DIR, STEP6_DIR):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def load_sources():
    """Import the packaged continuous DGP and binary layer (modules of the same name elsewhere are refused)."""
    _ensure_paths()
    for name in ("step6b_binary", "latent_cluster_dgp_formal"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            actual = os.path.abspath(getattr(loaded, "__file__", "") or "")
            if not actual.startswith(os.path.abspath(STEP6_DIR)):
                raise RuntimeError(f"{name} is already loaded from outside the package: {actual}")
    from latent_cluster_dgp_formal import generate_latent_cluster_dgp
    import step6b_binary as b6
    return generate_latent_cluster_dgp, b6


def generate_base(group: dict, structures: dict) -> tuple[dict, dict]:
    """One continuous base realization (gamma_XY = 1); returns (continuous, base_meta)."""
    generate_latent_cluster_dgp, _b6 = load_sources()
    block = structures[group["structure_id"]]
    params = dict(block["params"])
    continuous = generate_latent_cluster_dgp(
        **params, seed=group["data_seed"], param_seed=group["param_seed"])
    env = np.asarray(continuous["env"])
    base_meta = {
        "base_group_key": group["base_group_key"],
        "structure_id": group["structure_id"],
        "param_rep": int(group["param_rep"]),
        "data_rep": int(group["data_rep"]),
        "param_seed": int(group["param_seed"]),
        "data_seed": int(group["data_seed"]),
        "u_epsilon_seed": int(group["u_epsilon_seed"]),
        "gamma_XY_continuous": dt.BASE_GAMMA_XY,
        "sig_x_spec": float(block["sig_x_spec"]),
        "dgp_params": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in params.items()},
        "n_rows": int(np.asarray(continuous["F"]).shape[0]),
        "n_features": int(np.asarray(continuous["F"]).shape[1]),
        "n_centers": int(np.unique(env).size),
        "n_per_center_actual": [int(v) for v in
                                np.asarray(continuous["n_per_center_actual"]).tolist()],
    }
    return continuous, base_meta


def unmeasured_u_for_base(continuous, group: dict, rho_block: float):
    """U drawn once from the stream of structure, param and data; the U points of a structure share it."""
    S = bu.shared_common_score(continuous["F"], continuous["true_shared_causal"],
                               rho_block)
    rng_eps = np.random.default_rng(int(group["u_epsilon_seed"]))
    U, meta = bu.draw_unmeasured_u(S, rng_eps)
    shared_idx = [int(j) for j in continuous["true_shared_causal"]]
    corr_j = [float(np.corrcoef(U, np.asarray(continuous["F"], float)[:, j])[0, 1])
              for j in shared_idx]
    meta["sample_corr_U_shared_mean"] = float(np.mean(corr_j))
    meta["sample_corr_U_shared_min"] = float(np.min(corr_j))
    meta["sample_corr_U_shared_max"] = float(np.max(corr_j))
    meta["n_shared"] = len(shared_idx)
    meta["rho_block"] = float(rho_block)
    meta["u_epsilon_seed"] = int(group["u_epsilon_seed"])
    return U, S, meta


def binary_unit(continuous, row: dict, group: dict, b6, *, u=None):
    """Observed data, evaluation truth and metadata of one analysis unit."""
    from bniseam_test.multiview_types import ObservedDataView

    gamma_logor = float(row["gamma_logor"])
    has_u = row["has_unmeasured_U"] == "true"
    lam_x = float(row["lambda_UX"]) if has_u else 0.0
    lam_y = float(row["lambda_UY"]) if has_u else 0.0
    if has_u and u is None:
        raise ValueError(f"{row['point_id']} is a U scenario but no U vector was given")

    # SIG_Y comes from the run table; SIG_X and C0 are module constants, so the run table must agree with them.
    if abs(float(row["SIG_X"]) - float(b6.SIG_X)) > 1e-12:
        raise RuntimeError("run table SIG_X=%s differs from the module constant %s"
                           % (row["SIG_X"], b6.SIG_X))
    if abs(float(row["C0"]) - float(b6.C0)) > 1e-12:
        raise RuntimeError("run table C0=%s differs from the module constant %s"
                           % (row["C0"], b6.C0))
    original = b6.SIG_X_SPEC
    original_sig_y = b6.SIG_Y
    b6.SIG_X_SPEC = float(group["sig_x_spec"])
    b6.SIG_Y = float(row["SIG_Y"])
    try:
        binary = bu.make_binary_u(
            continuous, gamma_logor,
            np.random.default_rng([group["data_seed"], 1]),
            np.random.default_rng([group["param_seed"], 2]),
            b6, u=(u if has_u else None), lambda_ux=lam_x, lambda_uy=lam_y)
    finally:
        b6.SIG_X_SPEC = original
        b6.SIG_Y = original_sig_y

    drift = abs(float(np.asarray(binary["tau_do_i"]).mean()) - float(binary["ate_do"]))
    if drift > 1e-12:
        raise RuntimeError(f"mean individual truth differs from ate_do by {drift:.3e}")
    if gamma_logor == 0.0:
        worst = float(np.max(np.abs(np.asarray(binary["tau_do_i"]))))
        if worst != 0.0:
            raise RuntimeError(f"individual risk differences must be exactly 0 at gamma_logor=0, max {worst:.3e}")

    env = np.asarray(continuous["env"])
    observed = ObservedDataView(
        features=np.asarray(continuous["F"]),
        treatment=np.asarray(binary["X_bin"]),
        outcome=np.asarray(binary["Y_bin"]),
        center_codes=env,
        centers=np.unique(env),
    )

    idx_Z = [int(j) for j in continuous["idx_Z"]]
    shared = [int(j) for j in continuous["true_shared_causal"]]
    spec = {int(k): [int(j) for j in v]
            for k, v in continuous["true_cluster_specific_causal"].items()}
    spec_all = sorted({j for v in spec.values() for j in v})
    observed_core = {k: sorted(set(idx_Z) | set(shared) | set(v))
                     for k, v in spec.items()}
    truth = {
        "gamma_logor": gamma_logor,
        "scenario": row["scenario"],
        "has_unmeasured_U": has_u,
        "ate_do": float(binary["ate_do"]),
        "ate_k": np.asarray(binary["ate_k"]).tolist(),
        "n_k": np.asarray(binary["n_k"]).tolist(),
        "ate_do_nk": float(binary["ate_do_nk"]),
        "p0_do": np.asarray(binary["p0_do"], float),
        "p1_do": np.asarray(binary["p1_do"], float),
        "tau_do_i": np.asarray(binary["tau_do_i"], float),
        "row_id": np.arange(observed.n_samples, dtype=np.int64),
        "true_cluster": np.asarray(continuous["true_cluster"]).tolist(),
        "true_cluster_sample": np.asarray(continuous["true_cluster_sample"], int),
        "idx_Z": idx_Z,
        "true_shared_causal": shared,
        "true_cluster_specific_causal": spec,
        "spec_all": spec_all,
        "spurious_vars": [int(j) for j in continuous["spurious_vars"]],
        "noise_vars": [int(j) for j in continuous["noise_vars"]],
        # observed_core = Z ∪ shared ∪ spec_k: the measured core, without U
        "observed_core_by_cluster": observed_core,
        "a_safe": sorted(set(idx_Z) | set(shared) | set(spec_all)),
        "full_causal_core_note": (
            "with U the full causal set also contains the unmeasured U, which is not in the recall denominator"
            if has_u else "no U: observed_core is the full causal core"),
        "beta_spec_X": {int(k): np.asarray(v).tolist()
                        for k, v in binary["beta_spec_X"].items()},
    }
    meta = {
        "gamma_logor": gamma_logor,
        # binary-layer strengths in effect, recorded in every unit
        "sig_y_applied": float(row["SIG_Y"]),
        "sig_x_spec_applied": float(group["sig_x_spec"]),
        "lambda_UX": lam_x, "lambda_UY": lam_y,
        "prevalence_x": float(np.mean(binary["X_bin"])),
        "prevalence_y": float(np.mean(binary["Y_bin"])),
        "n_rows": int(observed.n_samples),
        "n_features": int(np.asarray(continuous["F"]).shape[1]),
    }
    return observed, truth, meta


__all__ = ["load_sources", "generate_base", "unmeasured_u_for_base", "binary_unit"]
