"""Small, single-process binary example; not a paper-result replication."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
for folder in (ROOT / "src", ROOT / "vendor", ROOT / "vendor/step6_pkg"):
    sys.path.insert(0, str(folder))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import formal_dgp
import run_formal
from screening_bridge import load_screening


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs_smoke")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    params = dict(n_centers=9, n_per_center=800, n_per_center_var=0.1,
                  p_total=100, K=3, n_Z=8, n_causal_shared=10,
                  n_causal_specific=3, n_spurious=10,
                  rho_max=3.0, rho_min=1.0, delta_spec=0.0, nu_sd=0.4,
                  alpha_ZX=1.0, alpha_ZY=1.0, sigma_Z=1.0, kappa=0.0,
                  f_X="linear", f_Y="tanh", rho_block=0.6, rho_cross=0.3,
                  beta_mag_lo=2.0, beta_mag_hi=3.0, signed_beta=True,
                  gamma_XY=1.0, centers_per_cluster=None)
    group = dict(base_group_key="SMOKE__p001__d01", structure_id="SMOKE",
                 param_rep=1, data_rep=1, param_seed=1_000_000_123,
                 data_seed=2_000_000_456, u_epsilon_seed=424242,
                 sig_x_spec=0.6, point_ids=["TEST_P"])
    row = {key: str(value) for key, value in params.items()
           if key not in ("centers_per_cluster", "gamma_XY")}
    row.update(point_id="TEST_P", tier="main", axis_membership="SMOKE",
               calibration_member="false", scenario="observed_confounders",
               design_revision="small-example-not-paper-design",
               gamma_logor="1", SIG_X="1.2", SIG_X_SPEC="0.6", SIG_Y="2", C0="0",
               gamma_XY_continuous="1", centers_per_cluster="equal",
               has_unmeasured_U="false", u_shared_loading="0", u_noise_loading="0",
               lambda_UX="0", lambda_UY="0", report_panels="smoke",
               execution_methods="dual_icp|oracle_reference", n_methods="2",
               estimators="ra_standardized|aipw|psm_caliper", n_estimators="3")
    generate, b6 = formal_dgp.load_sources()
    continuous = generate(**params, seed=group["data_seed"], param_seed=group["param_seed"])
    m6 = load_screening(formal_dgp.STEP6_DIR)
    with (ROOT / "config/formal_r2.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["save_matching_detail"] = "summary"
    m6.ALPHA_ICP = float(config["icp_alpha"])
    record, _ = run_formal.run_unit(
        continuous, row, group, {"SMOKE": {"params": params, "sig_x_spec": 0.6}},
        m6, b6, config, None, None)
    assert record["unit_complete"] and not record["errors"], record["errors"]
    assert record["design_revision"] == row["design_revision"]
    assert record["label_status"]["center_map_available"], record["label_status"]
    estimates = {}
    for name, by_estimator in record["effects"].items():
        estimates[name] = {}
        for estimator, rows in by_estimator.items():
            overall = next(result for result in rows if result["stage"] == "overall")
            assert overall["point_status"] == "complete", (name, estimator, overall)
            estimates[name][estimator] = {key: overall.get(key) for key in ("estimate", "se")}
    path = run_formal.unit_path(args.out_dir, row, 1, 1)
    run_formal.write_atomic(path, record)
    print(json.dumps({"status": "passed", "elapsed_seconds": time.perf_counter() - start,
                      "shape": record["observed_shape"], "estimates": estimates,
                      "output": path}, indent=2))


if __name__ == "__main__":
    main()
