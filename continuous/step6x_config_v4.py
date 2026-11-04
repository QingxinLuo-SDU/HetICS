"""step6x_config_v4.py — continuous simulation: experiments, replicate structure, seeds and design points.

The DGP parameters, design axes and method arms come from step6x_config. This module adds the
two-level replicate structure (10 parameter seeds x 100 data seeds), the seed identifiers of each design
point, the ICP threshold alpha_inv = 1e-3 and the unmeasured-confounder experiment.

The coefficient band of the confounders is set by SCENA_BETA_ANCHOR ("1.0,1.25" -> b1-125, the setting of
the reported results; "2.0,3.0" -> b2-3; "2.0,2.4" -> b2-24). build_combos_v4() checks that the requested
beta batch matches the environment.
"""
from __future__ import annotations

import copy
import os

import step6x_config as cfg

# SCENA_EXPERIMENT: grid = 65-point design (default); u07 = unmeasured confounder U, 20 U points and one
# reference point without U.
EXPERIMENTS = ("grid", "u07")
EXPERIMENT = os.environ.get("SCENA_EXPERIMENT", "grid")
if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f"SCENA_EXPERIMENT must be one of {EXPERIMENTS}, got {EXPERIMENT!r}")

OUTPUT_VERSION = (f"scenA-cont-wccf-reml-4.3-{cfg.BETA_TAG}"
                  + ("-u07" if EXPERIMENT == "u07" else ""))

ALPHA_ICP = 1e-3                   # alpha_inv of the within-cluster invariance selection

N_FOLDS = 2                        # whole-center folds of the Robinson estimator
N_PARAM_SEEDS = 10
N_DATA_SEEDS_PER_PARAM = 100
PRIMARY_METHOD = "dual_icp"
MAIN_PARTITION_MODE = "seam"
MAIN_ICP_RULE = "strict"
DELTA_SPEC_SENSITIVITY_POINTS = cfg.DELTA_SPEC_SENSITIVITY_POINTS
ANCHOR_COMBO_ID = cfg.ANCHOR_COMBO_ID          # "A05"
K_TRUE = cfg.K_TRUE

# Fixed tables for seed derivation (Python's randomized hash() is never used).
BETA_BATCH_ID = {"b2-3": 1, "b2-24": 2, "b1-125": 3}
MASTER_SEED = 20260911
STREAM_ID = {"parameter": 1, "data": 2, "fold": 3, "algorithm": 4}

# combo_id -> integer, from the position of the axis in SWEEP_ORDER and the level index.
SEED_GROUP_ID = {
    f"{sweep}{level:02d}": si * 100 + level
    for si, sweep in enumerate(cfg.SWEEP_ORDER)
    for level in range(32)
}


def build_combos_v4(beta_batch: str, kappa_frac: float = cfg.KAPPA_FRAC_DEFAULT):
    """Design points of the current experiment (SCENA_EXPERIMENT)."""
    if EXPERIMENT == "u07":
        return build_u_combos_v4(beta_batch, kappa_frac)
    return build_grid_combos_v4(beta_batch, kappa_frac)


# Unmeasured confounder U (see latent_cluster_dgp_specx_u.py).
U_SENSITIVITY = dict(lambda_ux=0.5, lambda_uy=0.5,
                     u_shared_loading=0.7, u_noise_loading=0.3)
# (U point, grid point with the same parameters and seeds). Effect axis 8 levels (-2, -1, -0.5, 0, 0.5, 1,
# 1.5, 3), rho_max 6 levels and number of centers 6 levels. Each U point shares the seed group of its
# reference, so the data differ only through U.
U_POINTS = (
    ("U001", "A00"), ("U002", None), ("U003", "A01"), ("U004", "A02"),
    ("U005", "A03"), ("U006", "A04"), ("U007", "A05"), ("U008", "A06"),
    ("U009", "B00"), ("U010", "B01"), ("U011", "B02"), ("U012", "B03"),
    ("U013", "B05"), ("U014", "B06"),
    ("U015", "E00"), ("U016", "E02"), ("U017", "E03"), ("U018", "E04"),
    ("U019", "E05"), ("U020", "E06"),
)
# The grid has no gamma = -1; this level uses an unused seed group and its own reference without U (U002N).
U_GAMMA_MINUS_ONE = -1.0
U_GAMMA_MINUS_ONE_GROUP = "A07"
U_GAMMA_MINUS_ONE_REF = "U002N"


def build_u_combos_v4(beta_batch: str, kappa_frac: float = cfg.KAPPA_FRAC_DEFAULT):
    """20 U points and the reference of U002: 21 design points."""
    grid = {c["combo_id"]: c for c in build_grid_combos_v4(beta_batch, kappa_frac)}
    combos = []
    for uid, ref in U_POINTS:
        if ref is None:
            base = copy.deepcopy(grid["A04"])           # effect-axis template; only gamma differs
            base["dgp_params"]["gamma_XY"] = U_GAMMA_MINUS_ONE
            base["true_gamma"] = U_GAMMA_MINUS_ONE
            base["scan_val"] = U_GAMMA_MINUS_ONE
            base["level_idx"] = 7
            base["seed_group_id"] = U_GAMMA_MINUS_ONE_GROUP
            ref = U_GAMMA_MINUS_ONE_REF
        else:
            base = copy.deepcopy(grid[ref])
        base.update(combo_id=uid, pair_reference_combo_id=ref,
                    unmeasured_u=dict(U_SENSITIVITY), output_version=OUTPUT_VERSION,
                    label=f"U:{base['scan_param']}={base['scan_val']}")
        combos.append(base)

    noU = copy.deepcopy(next(c for c in combos if c["combo_id"] == "U002"))
    del noU["unmeasured_u"]
    noU.update(combo_id=U_GAMMA_MINUS_ONE_REF, pair_reference_combo_id=None,
               label=f"U-ref:{noU['scan_param']}={noU['scan_val']}")
    combos.append(noU)

    with_u = [c for c in combos if "unmeasured_u" in c]
    assert len(combos) == 21 and len(with_u) == 20, "expected 20 U points and one reference"
    assert len({c["combo_id"] for c in combos}) == 21
    assert all(c["seed_group_id"] in SEED_GROUP_ID for c in combos)
    assert U_GAMMA_MINUS_ONE_GROUP not in grid, "the borrowed seed group must not be a grid point"
    gammas = sorted(c["true_gamma"] for c in with_u if c["sweep"] == "A")
    assert gammas == [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0], gammas
    for c in with_u:
        r = grid.get(c["pair_reference_combo_id"]) or noU
        assert c["dgp_params"] == r["dgp_params"] and c["seed_group_id"] == r["seed_group_id"]
    return combos


def build_grid_combos_v4(beta_batch: str, kappa_frac: float = cfg.KAPPA_FRAC_DEFAULT):
    """The 65 design points with beta_batch and seed_group_id attached.

    The delta_spec points (axis G) share the seed group of the anchor A05.
    """
    if beta_batch not in BETA_BATCH_ID:
        raise ValueError(f"unknown beta_batch {beta_batch!r}; expected one of {sorted(BETA_BATCH_ID)}")
    if beta_batch != cfg.BETA_TAG:
        raise ValueError(f"beta_batch={beta_batch!r} differs from {cfg.BETA_TAG!r} implied by SCENA_BETA_ANCHOR")

    combos = copy.deepcopy(cfg.build_combos(kappa_frac))
    _restore_strongest_c_band(combos)
    for combo in combos:
        combo["beta_batch"] = beta_batch
        combo["seed_group_id"] = (ANCHOR_COMBO_ID if combo["sweep"] == "G"
                                  else combo["combo_id"])
        combo["output_version"] = OUTPUT_VERSION
        if combo["seed_group_id"] not in SEED_GROUP_ID:
            raise ValueError(f"combo_id {combo['seed_group_id']!r} is not in the seed table")

    assert len(combos) == 65, f"expected 65 design points, found {len(combos)}"
    assert sum(c["analysis_role"] == "main" for c in combos) == 58
    assert sum(c["analysis_role"] == cfg.ANALYSIS_ROLE_DELTA_SPEC
               for c in combos) == 7
    assert all(c["dgp_params"]["delta_spec"] == 0.0
               for c in combos if c["analysis_role"] == "main"), \
        "main points must have delta_spec = 0"
    assert [c["dgp_params"]["delta_spec"] for c in combos
            if c["analysis_role"] == cfg.ANALYSIS_ROLE_DELTA_SPEC] == \
        list(DELTA_SPEC_SENSITIVITY_POINTS)
    assert all(c["seed_group_id"] == ANCHOR_COMBO_ID
               for c in combos if c["sweep"] == "G"), \
        "delta_spec points must share the seed group of the anchor"
    _assert_anchors(combos)
    return combos


C_STRONGEST_BAND = (2.0, 3.0)


def _restore_strongest_c_band(combos) -> None:
    """When the anchor band equals one of the fixed bands of axis C, C07 is set to (2.0, 3.0).

    Axis C has seven fixed bands and an eighth band equal to the anchor. With the anchor (1.0, 1.25) the
    eighth band would duplicate C03, so C07 takes the strongest band instead.
    """
    c = [x for x in combos if x["sweep"] == "C"]
    bands = [(x["dgp_params"]["beta_mag_lo"], x["dgp_params"]["beta_mag_hi"]) for x in c]
    if len(set(bands)) == len(bands):
        return
    c07 = next(x for x in c if x["combo_id"] == "C07")
    lo, hi = C_STRONGEST_BAND
    c07["dgp_params"]["beta_mag_lo"], c07["dgp_params"]["beta_mag_hi"] = lo, hi
    c07["scan_val"] = f"{lo}-{hi}"
    c07["label"] = f"C:{c07['scan_param']}={c07['scan_val']}"
    bands = [(x["dgp_params"]["beta_mag_lo"], x["dgp_params"]["beta_mag_hi"]) for x in c]
    assert len(set(bands)) == len(bands) == 8, f"axis C should have 8 distinct bands, found {bands}"
    assert cfg.ANCHOR_BETA in bands, "the anchor band must remain on axis C"


def _assert_anchors(combos):
    """Anchor: gamma 1.5, p 500, 12 centers, 2000 individuals per center, rho_max 3, delta_spec 0."""
    anchor = next(c for c in combos if c["combo_id"] == ANCHOR_COMBO_ID)
    pr = anchor["dgp_params"]
    expect = {"gamma_XY": 1.5, "p_total": 500, "n_centers": 12,
              "n_per_center": 2000, "rho_max": 3.0, "delta_spec": 0.0}
    for name, want in expect.items():
        got = float(pr[name])
        if abs(got - float(want)) > 1e-12:
            raise AssertionError(f"anchor {name} should be {want}, found {got}")
