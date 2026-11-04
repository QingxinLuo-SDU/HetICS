"""hypothesis_calibration_config.py — design of the continuous type I error and power experiment.

Separate from the 65-point simulation (own entry point, output version and result directory).

  DGP              latent_cluster_dgp_specx anchor defaults with confounder coefficient band (1.0, 1.25)
  K                3, delta_spec = 0
  centers axis     3, 4, 5, 6, 8, 10, 12 centers per cluster, 10,000 individuals per center (no size variation)
  sample-size axis 4 centers per cluster, 5,000-12,000 individuals per center
  true effect      gamma_XY in {0, 0.035}; 0 gives the type I error, 0.035 the power on the centers axis;
                   the sample-size axis uses gamma_XY = 0.055 for power
  replicates       10 parameter seeds x 100 data seeds per design point
  arms             dual_icp (HetICS) and oracle_reference (true clusters and S_k^*, the same estimators)
  ICP threshold    alpha_inv = 1e-3

The centers per cluster are fixed without size variation so that the number of centers is the only quantity
that changes along the centers axis.
"""
from __future__ import annotations

import copy

import step6x_config as cfg

BETA_MAG = (1.0, 1.25)
BETA_TAG_REAL = f"b{BETA_MAG[0]:g}-{BETA_MAG[1]:g}".replace(".", "")
OUTPUT_VERSION = f"scenA-cont-wccf-reml-testcal-2.0-{BETA_TAG_REAL}"
EXPERIMENT_ID = "hypothesis_calibration"
BETA_BATCH = "b2-3"          # seed namespace identifier; the coefficient band is BETA_MAG

K_TRUE = 3
CENTERS_PER_CLUSTER = (3, 4, 5, 6, 8, 10, 12)
GAMMA_VALUES = (0.0, 0.035)
N_PER_CENTER = 10000
N_PER_CENTER_VAR = 0.0
DELTA_SPEC = 0.0
N_PARAM_SEEDS = 10
N_DATA_SEEDS_PER_PARAM = 100

ALPHA_ICP = 1e-3

# oracle_reference = true cluster labels + S_k^* + the same estimators; a reference, not a method.
PRIMARY_ARM = "dual_icp"
DIAGNOSTIC_ARM = "oracle_reference"
ARMS = (PRIMARY_ARM, DIAGNOSTIC_ARM)
ARM_ROLES = {PRIMARY_ARM: "primary", DIAGNOSTIC_ARM: "diagnostic"}
RANKED_ARMS = (PRIMARY_ARM,)
# Candidate channel of each learned-cluster arm: dual = S_x union S_y.
ARM_CHANNEL = {PRIMARY_ARM: "dual"}


def _gamma_tag(gamma: float) -> str:
    """gamma without decimal point or sign: 0 -> '0', 0.035 -> '0035'."""
    return ('%g' % gamma).replace('.', '').replace('-', 'm')


def _point_id(m: int, gamma: float) -> str:
    """Point identifier such as M04_G0 or M04_G0035."""
    return f"M{int(m):02d}_G{_gamma_tag(gamma)}"


def build_calibration_points(beta_batch: str = BETA_BATCH,
                             kappa_frac: float = cfg.KAPPA_FRAC_DEFAULT) -> list:
    """Centers axis: 7 levels of centers per cluster x 2 true effects = 14 points.

    DGP parameters are copied from cfg.BASE_PARAMS with n_centers, n_per_center, n_per_center_var,
    gamma_XY, beta_mag_lo and beta_mag_hi replaced.
    """
    if beta_batch != BETA_BATCH:
        raise ValueError(f"the calibration design uses seed batch {BETA_BATCH!r}, got {beta_batch!r}")
    if beta_batch != cfg.BETA_TAG:
        raise ValueError(f"beta_batch={beta_batch!r} differs from {cfg.BETA_TAG!r} implied by SCENA_BETA_ANCHOR")

    points = []
    for m in CENTERS_PER_CLUSTER:
        for gamma in GAMMA_VALUES:
            params = copy.deepcopy(dict(cfg.BASE_PARAMS))
            params["n_centers"] = K_TRUE * int(m)
            params["n_per_center"] = N_PER_CENTER
            params["n_per_center_var"] = N_PER_CENTER_VAR
            params["gamma_XY"] = float(gamma)
            params["beta_mag_lo"], params["beta_mag_hi"] = BETA_MAG
            points.append(dict(
                point_id=_point_id(m, gamma),
                experiment_id=EXPERIMENT_ID,
                output_version=OUTPUT_VERSION,
                beta_batch=beta_batch,
                centers_per_cluster=int(m),
                n_centers=K_TRUE * int(m),
                gamma_true=float(gamma),
                true_gamma=float(gamma),
                dgp_params=params,
                seam_sparsity=cfg._sparsity_for(params, kappa_frac),
                alpha_icp=ALPHA_ICP,
                n_param_seeds=N_PARAM_SEEDS,
                n_data_seeds_per_param=N_DATA_SEEDS_PER_PARAM,
                arms=list(ARMS),
            ))
    _assert_design(points)
    return points


def _assert_design(points) -> None:
    assert len(points) == len(CENTERS_PER_CLUSTER) * len(GAMMA_VALUES) == 14, \
        f"expected 14 calibration points, found {len(points)}"
    assert len({p["point_id"] for p in points}) == len(points), "point_id must be unique"
    assert min(CENTERS_PER_CLUSTER) >= 3, "the Chow test needs at least three centers per cluster"
    for p in points:
        pr = p["dgp_params"]
        assert pr["K"] == K_TRUE
        assert pr["delta_spec"] == DELTA_SPEC
        assert pr["n_per_center"] == N_PER_CENTER
        assert pr["n_per_center_var"] == 0.0, "center sizes must not vary"
        assert pr["n_centers"] == K_TRUE * p["centers_per_cluster"]
        assert float(pr["gamma_XY"]) in GAMMA_VALUES
        assert (pr["beta_mag_lo"], pr["beta_mag_hi"]) == BETA_MAG
        assert p["alpha_icp"] == ALPHA_ICP
    _assert_gamma_outside_cancellation_band()


def _assert_gamma_outside_cancellation_band() -> None:
    """Nonzero gamma must lie outside 0.689 x [lo/hi, hi/lo], where an outcome-side partial coefficient
    gamma beta_X + 0.689 beta_Y can vanish."""
    lo, hi = BETA_MAG
    band_lo, band_hi = 0.689 * (lo / hi), 0.689 * (hi / lo)
    for gamma in GAMMA_VALUES:
        if gamma == 0.0:
            continue
        assert not (band_lo <= abs(gamma) <= band_hi), (
            f"gamma={gamma} lies in the cancellation band [{band_lo:.3f}, {band_hi:.3f}] of beta {BETA_MAG}")


# Sample-size axis: 4 centers per cluster (12 centers) and 5,000-12,000 individuals per center. The 10,000
# level is the centers-axis point M04. SCENA_CAL_AXIS=n runs these 12 points.
CAL_AXES = ("m", "n", "p")
N_AXIS_CENTERS_PER_CLUSTER = 4
N_AXIS_LEVELS = (5000, 6000, 7000, 8000, 9000, 12000)


def build_n_axis_points(beta_batch: str = BETA_BATCH,
                        kappa_frac: float = cfg.KAPPA_FRAC_DEFAULT) -> list:
    """6 sample sizes x 2 true effects; template M04_G0 with only n_per_center and gamma_XY changed."""
    template = point_by_id(_point_id(N_AXIS_CENTERS_PER_CLUSTER, 0.0),
                           build_calibration_points(beta_batch, kappa_frac))
    points = []
    for n in N_AXIS_LEVELS:
        for gamma in GAMMA_VALUES:
            pt = copy.deepcopy(template)
            pt["dgp_params"]["n_per_center"] = int(n)
            pt["dgp_params"]["gamma_XY"] = float(gamma)
            pt.update(point_id=f"N{int(n):05d}_G{_gamma_tag(gamma)}", cal_axis="n",
                      n_per_center=int(n), gamma_true=float(gamma), true_gamma=float(gamma))
            points.append(pt)
    _assert_n_axis(points, template)
    return points


def _assert_n_axis(points, template) -> None:
    assert len(points) == len(N_AXIS_LEVELS) * len(GAMMA_VALUES) == 12, len(points)
    assert len({p["point_id"] for p in points}) == len(points), "point_id must be unique"
    assert N_PER_CENTER not in N_AXIS_LEVELS, "the 10,000 level is the centers-axis point M04"
    for p in points:
        pr = p["dgp_params"]
        assert p["centers_per_cluster"] == N_AXIS_CENTERS_PER_CLUSTER
        assert pr["n_centers"] == K_TRUE * N_AXIS_CENTERS_PER_CLUSTER
        assert pr["n_per_center_var"] == 0.0, "center sizes must not vary"
        assert float(pr["gamma_XY"]) in GAMMA_VALUES
        diff = {k for k in pr if pr[k] != template["dgp_params"][k]}
        assert diff <= {"n_per_center", "gamma_XY"}, f"{p['point_id']} changes {diff}"
        assert p["alpha_icp"] == ALPHA_ICP and p["seam_sparsity"] == template["seam_sparsity"]


# Power along the sample-size axis: gamma = 0.055 at 7 sample sizes (10,000 included). Seeds equal those of
# the gamma = 0 points at the same sample size because gamma does not enter the seeds. SCENA_CAL_AXIS=p.
P_AXIS_GAMMA = 0.055
P_AXIS_LEVELS = (5000, 6000, 7000, 8000, 9000, 10000, 12000)


def build_p_axis_points(beta_batch: str = BETA_BATCH,
                        kappa_frac: float = cfg.KAPPA_FRAC_DEFAULT) -> list:
    """7 sample sizes x gamma = 0.055; template M04_G0."""
    template = point_by_id(_point_id(N_AXIS_CENTERS_PER_CLUSTER, 0.0),
                           build_calibration_points(beta_batch, kappa_frac))
    points = []
    for n in P_AXIS_LEVELS:
        pt = copy.deepcopy(template)
        pt["dgp_params"]["n_per_center"] = int(n)
        pt["dgp_params"]["gamma_XY"] = P_AXIS_GAMMA
        pt.update(point_id=f"N{int(n):05d}_G{_gamma_tag(P_AXIS_GAMMA)}", cal_axis="p",
                  n_per_center=int(n), gamma_true=P_AXIS_GAMMA, true_gamma=P_AXIS_GAMMA)
        points.append(pt)
    lo, hi = BETA_MAG
    assert not 0.689 * lo / hi <= P_AXIS_GAMMA <= 0.689 * hi / lo, "gamma lies in the cancellation band"
    assert len({p["point_id"] for p in points}) == len(P_AXIS_LEVELS) == 7
    for p in points:
        diff = {k for k in p["dgp_params"] if p["dgp_params"][k] != template["dgp_params"][k]}
        assert diff <= {"n_per_center", "gamma_XY"}, f"{p['point_id']} changes {diff}"
    return points


def point_by_id(point_id: str, points=None) -> dict:
    points = build_calibration_points() if points is None else points
    for p in points:
        if p["point_id"] == point_id:
            return p
    raise KeyError(f"no calibration point {point_id!r}")
