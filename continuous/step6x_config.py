"""step6x_config.py — continuous simulation design: anchor, design axes and method arms.

DGP: latent_cluster_dgp_specx with alpha_spec_X = 1.0, so the specific covariates of a cluster confound
the exposure-outcome relation in that cluster, while other clusters' specific covariates enter only the
outcome. Main design: axes A-F and H vary one parameter at a time from the anchor with delta_spec = 0
(58 points); axis G gives 7 standalone points for the other-cluster leakage strength delta_spec.
Fixed settings: knockoff q = 0.10, K = 3, sEAM max_iter = 300.
"""
import math
import os

import run_scenarioA_main as off
import step6_methods as m6

# Coefficient band of the confounders, "lo,hi" (default "2.0,3.0").
_env_beta = os.environ.get("SCENA_BETA_ANCHOR", "2.0,3.0")
try:
    _lo, _hi = (float(x) for x in _env_beta.split(","))
except Exception as exc:                       # noqa: BLE001
    raise ValueError(f'SCENA_BETA_ANCHOR must look like "lo,hi" (e.g. "2.0,2.4"), got {_env_beta!r}') from exc
if not (0 < _lo < _hi):
    raise ValueError(f"SCENA_BETA_ANCHOR requires 0 < lo < hi, got ({_lo}, {_hi})")
ANCHOR_BETA = (_lo, _hi)
BETA_TAG = f"b{_lo:g}-{_hi:g}".replace(".", "")

ANCHOR_GAMMA = 1.5
GAMMA_LEVELS = [-2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]

ALPHA_SPEC_X = 1.0
K_TRUE = off.K_TRUE
BASE_PARAMS = {**off.BASE_PARAMS, "alpha_spec_X": ALPHA_SPEC_X,
               "gamma_XY": ANCHOR_GAMMA,
               "delta_spec": 0.0,
               "beta_mag_lo": ANCHOR_BETA[0], "beta_mag_hi": ANCHOR_BETA[1]}

SEAM_MAX_ITER = 300
SEAM_TOL = 1e-4
# sEAM sparsity = max(round_half_up(KAPPA_FRAC * p), n_shared + K * n_spec + 5).
KAPPA_FRAC_DEFAULT = 0.20

Q_KNOCK = m6.Q_KNOCK             # 0.10

# Method arms of the within-cluster pipeline: three candidate rules followed by Chow-ICP, and the oracle.
METHODS = ("xside_icp", "dual_icp", "yside_icp", "oracle_min")
ICP_METHODS = ("xside_icp", "dual_icp", "yside_icp")

C_COMMON_LEVELS = [(0.3, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.25),
                   (1.25, 1.5), (1.5, 2.0), (1.7, 2.3)]
GRID = {
    "A": ("gamma_XY",     GAMMA_LEVELS),
    "B": ("rho_max",      [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0]),
    "C": ("beta_band",    C_COMMON_LEVELS + [ANCHOR_BETA]),
    "D": ("p_total",      [200, 300, 400, 500, 600, 800, 1000]),
    "E": ("n_centers",    [9, 12, 15, 18, 21, 24, 27]),
    "F": ("n_per_center", [1000, 1200, 1500, 1700, 2000, 2200, 2800, 3000]),
}
DELTA_SPEC_SENSITIVITY_POINTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70)
H_RHO_BLOCK = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]   # rho_cross fixed at 0
H_RHO_CROSS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]       # rho_block fixed at 0.6
SWEEP_ORDER = list("ABCDEFGH")
MAIN_SWEEPS = tuple("ABCDEFH")
ANALYSIS_ROLE_MAIN = "main"
ANALYSIS_ROLE_DELTA_SPEC = "delta_spec_sensitivity"
ANCHORS = {"A": ANCHOR_GAMMA, "B": 3.0, "C": ANCHOR_BETA, "D": 500, "E": 12, "F": 2000}
ANCHOR_COMBO_ID = "A%02d" % GAMMA_LEVELS.index(ANCHOR_GAMMA)


def knock_min_selections(q):
    """A non-empty knockoff+ selection contains at least ceil(1/q) variables."""
    return int(math.ceil(1.0 / q))


def knock_feasibility(params, q):
    """Whether the number of true confounders per cluster reaches the knockoff+ minimum selection size."""
    n_core = params["n_Z"] + params["n_causal_shared"] + params["n_causal_specific"]
    need = knock_min_selections(q)
    return dict(a_core_size=n_core, knock_min_sel_required=need,
                knock_feasible=bool(n_core >= need),
                knock_margin=n_core - need)


def _sparsity_for(params, frac):
    """sEAM sparsity: max(round_half_up(frac * p), n_shared + K * n_spec + 5)."""
    floor = params["n_causal_shared"] + params["K"] * params["n_causal_specific"] + 5
    return int(max(off._round_half_up(frac * params["p_total"]), floor))


def _mk(sweep, level_idx, scan_param, scan_val, params, frac, sub="",
        analysis_role=None):
    role = analysis_role or (ANALYSIS_ROLE_DELTA_SPEC if sweep == "G"
                             else ANALYSIS_ROLE_MAIN)
    return {
        "sweep": sweep, "combo_id": f"{sweep}{level_idx:02d}", "level_idx": level_idx,
        "scan_param": scan_param, "scan_val": scan_val, "sub_sweep": sub,
        "analysis_role": role,
        "label": f"{sweep}:{scan_param}={scan_val}",
        "dgp_params": params, "seam_sparsity": _sparsity_for(params, frac),
        "true_gamma": float(params["gamma_XY"]),
    }


def build_combos(kappa_frac=KAPPA_FRAC_DEFAULT):
    """All 65 design points (58 main points and 7 delta_spec points)."""
    combos = []
    for sweep in SWEEP_ORDER:
        if sweep == "G":
            for i, v in enumerate(DELTA_SPEC_SENSITIVITY_POINTS):
                pr = {**BASE_PARAMS, "delta_spec": float(v)}
                combos.append(_mk(
                    sweep, i, "delta_spec", float(v), pr, kappa_frac,
                    sub="standalone_point", analysis_role=ANALYSIS_ROLE_DELTA_SPEC))
            continue
        if sweep == "H":
            i = 0
            for rb in H_RHO_BLOCK:
                pr = {**BASE_PARAMS, "rho_block": rb, "rho_cross": 0.0}
                sub = "rho_block+rho_cross" if abs(rb - 0.6) < 1e-12 else "rho_block"
                combos.append(_mk(sweep, i, "rho_block", rb, pr, kappa_frac, sub)); i += 1
            for rc in H_RHO_CROSS:
                if abs(rc) < 1e-12:
                    continue        # (rho_block 0.6, rho_cross 0) is already on the rho_block sweep
                pr = {**BASE_PARAMS, "rho_block": 0.6, "rho_cross": rc}
                combos.append(_mk(sweep, i, "rho_cross", rc, pr, kappa_frac, "rho_cross")); i += 1
            continue
        scan_param, levels = GRID[sweep]
        for i, v in enumerate(levels):
            pr = dict(BASE_PARAMS)
            if sweep == "C":
                pr["beta_mag_lo"], pr["beta_mag_hi"] = v
                scan_val = f"{v[0]}-{v[1]}"
            elif sweep == "D":
                # _p_scaled_params scales the variable counts with p and passes the other entries through.
                pr, _ = off._p_scaled_params(int(v), pr)
                assert "alpha_spec_X" in pr, "_p_scaled_params dropped alpha_spec_X"
                scan_val = v
            else:
                pr[scan_param] = v
                scan_val = v
            combos.append(_mk(sweep, i, scan_param, scan_val, pr, kappa_frac))
    assert_grid(combos)
    return combos


C_X_LINEAR = 1.0          # E[f'(u)] for f_X = "linear"
C_Y_TANH = 0.689027       # E[f'(u)] for f_Y(u) = tanh(1.5 u), u ~ N(0, 1)
CANCEL_BAND_MIN_MARGIN = 0.02


def cancellation_band(mag_lo, mag_hi, c_x=C_X_LINEAR, c_y=C_Y_TANH):
    """Range of |gamma| in which an outcome-side partial coefficient gamma c_X beta_X + c_Y beta_Y can vanish."""
    r = c_y / c_x
    return (r * mag_lo / mag_hi, r * mag_hi / mag_lo)


def assert_grid(combos):
    """Checks of the design grid."""
    assert len(combos) == 65, f"expected 65 design points, found {len(combos)}"
    bad = []
    for lo, hi in GRID["C"][1]:
        b0, b1 = cancellation_band(lo, hi)
        g = abs(ANCHOR_GAMMA)
        if b0 <= g <= b1:
            bad.append((f"[{lo},{hi}]", round(b0, 3), round(b1, 3), "inside"))
        elif min(abs(g - b0), abs(g - b1)) / g < CANCEL_BAND_MIN_MARGIN:
            bad.append((f"[{lo},{hi}]", round(b0, 3), round(b1, 3), "margin < 2%"))
    assert not bad, f"anchor gamma={ANCHOR_GAMMA} lies in or near the cancellation band of {bad}"
    n_h = sum(1 for c in combos if c["sweep"] == "H")
    assert n_h == 14, f"axis H should have 14 points, found {n_h}"
    main = [c for c in combos if c["analysis_role"] == ANALYSIS_ROLE_MAIN]
    sens = [c for c in combos if c["analysis_role"] == ANALYSIS_ROLE_DELTA_SPEC]
    assert len(main) == 58, f"expected 58 main points, found {len(main)}"
    assert len(sens) == len(DELTA_SPEC_SENSITIVITY_POINTS), \
        f"expected {len(DELTA_SPEC_SENSITIVITY_POINTS)} delta_spec points, found {len(sens)}"
    assert all(c["sweep"] in MAIN_SWEEPS and c["dgp_params"]["delta_spec"] == 0.0
               for c in main), "main points come from axes A-F/H with delta_spec = 0"
    assert [c["dgp_params"]["delta_spec"] for c in sens] == \
        list(DELTA_SPEC_SENSITIVITY_POINTS), "delta_spec points changed"
    for c in sens:
        assert c["sweep"] == "G" and c["sub_sweep"] == "standalone_point", \
            f"{c['combo_id']} must be a standalone point of axis G"
        for key, value in BASE_PARAMS.items():
            if key != "delta_spec":
                assert c["dgp_params"][key] == value, \
                    f"{c['combo_id']} changes {key} in addition to delta_spec"
    for s, anchor in ANCHORS.items():
        assert anchor in GRID[s][1], f"axis {s} lacks the anchor {anchor}"
    assert 0.6 in H_RHO_BLOCK and 0.0 in H_RHO_BLOCK
    assert 0.3 in H_RHO_CROSS
    ids = [c["combo_id"] for c in combos]
    assert len(ids) == len(set(ids)), "duplicate combo_id"
    for c in combos:
        assert c["dgp_params"].get("alpha_spec_X") == ALPHA_SPEC_X, \
            f"{c['combo_id']} lost alpha_spec_X"
    # Fixed-X knockoffs need n_k >= 2p in every learned cluster.
    for c in combos:
        pr = c["dgp_params"]
        n_k = (pr["n_centers"] // pr["K"]) * int(pr["n_per_center"] * 0.9)
        assert n_k >= 2 * pr["p_total"], \
            f"{c['combo_id']}: worst-case n_k={n_k} < 2p={2 * pr['p_total']}"
    # Every point has at least ceil(1/q) + 5 true confounders per cluster.
    thin = [(c["combo_id"], knock_feasibility(c["dgp_params"], Q_KNOCK)["knock_margin"])
            for c in combos
            if knock_feasibility(c["dgp_params"], Q_KNOCK)["knock_margin"] < 5]
    assert not thin, f"knockoff margin below 5 at {thin}"
    assert set(ICP_METHODS) < set(METHODS)
