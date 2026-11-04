"""Small real DGP -> sEAM -> screening -> Robinson/RA -> REML example.

Run: python smoke.py
Uses 12 centers, 100 observations per center and 80 features. It is an
installation example, not a reproduction of the paper's numerical results.
"""
import json
import os
import time

os.environ["SCENA_BETA_ANCHOR"] = "1.0,1.25"
os.environ["SCENA_EXPERIMENT"] = "grid"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

import step6x_config_v4 as cfg
from ra_core import aggregate, extra_estimators
from step6x_core_v4 import fit_once
from v4_seeds import seed_key_for, seeds_for_key


def main():
    started = time.perf_counter()
    combo = next(c for c in cfg.build_combos_v4("b1-125") if c["combo_id"] == "A05")
    combo["dgp_params"].update(n_centers=12, n_per_center=100, n_per_center_var=0.0,
                               p_total=80, n_Z=3, n_causal_shared=8,
                               n_causal_specific=4, n_spurious=12)
    combo["seam_sparsity"] = 15
    seeds = seeds_for_key(seed_key_for(combo, 0, 0))
    fitted = fit_once(combo, seeds)
    data = fitted.data
    env = np.asarray(data["env"])
    selected = fitted.selected_sets[("seam", "strict", "dual_icp")]
    members = fitted.diagnostics["members_by_mode"]["seam"]
    clusters = []
    for k, centers in sorted(members.items()):
        mask = np.isin(env, centers)
        cols = sorted(int(j) for j in selected[k])
        result = extra_estimators(data["X_treat"][mask], data["Y"][mask],
                                  data["F"][mask][:, cols], env[mask], gps=False)["ra"]
        robinson = fitted.plr_by_analysis[("seam", "strict", "dual_icp")][k]
        clusters.append({"cluster": int(k), "centers": len(centers),
                         "selected": cols, "ra": result,
                         "robinson_status": robinson.status})
    assert len(clusters) == 3 and any(c["ra"]["status"] == "ok" for c in clusters)
    for cluster in clusters:
        if cluster["ra"]["status"] == "ok":
            assert np.isfinite(cluster["ra"]["theta"]) and cluster["ra"]["se"] > 0
    summary = {"example": "small_nonformal_continuous", "n": len(env), "p": int(data["F"].shape[1]),
               "param_seed": seeds.param_seed, "data_seed": seeds.data_seed,
               "algorithm_seed": seeds.algorithm_seed, "fold_seed": seeds.fold_seed,
               "clusters": clusters, "ra_aggregate": aggregate([c["ra"] for c in clusters]),
               "elapsed_seconds": round(time.perf_counter() - started, 3)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
