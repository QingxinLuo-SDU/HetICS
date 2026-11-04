"""hypothesis_calibration_seeds.py — seed namespace of the calibration experiment.

Four independent streams (parameter / data / fold / algorithm) with the two-level structure of the main
simulation. CALIBRATION_NAMESPACE_ID keeps these seeds apart from the 65-point design. gamma does not enter
the seeds, so the zero-effect and nonzero-effect points of a level share their random numbers.
"""
from __future__ import annotations

import numpy as np

from hypothesis_calibration_config import EXPERIMENT_ID, N_PER_CENTER
from step6x_config_v4 import BETA_BATCH_ID, MASTER_SEED, STREAM_ID
from v4_types import SeedBundle

CALIBRATION_NAMESPACE_ID = 907001
# Sample-size axis: the individuals per center enter the seed, except at 10,000 (the centers-axis point M04).
N_AXIS_NAMESPACE_ID = 907002


def _one_seed(parts) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1, dtype=np.uint32)[0])


def calibration_seeds(point: dict, param_index: int, data_index: int,
                      master_seed: int = MASTER_SEED) -> SeedBundle:
    """Seeds from (beta_batch, centers per cluster, param_index, data_index) and, on the sample-size axis,
    the individuals per center; gamma does not enter."""
    batch_id = BETA_BATCH_ID[point["beta_batch"]]
    common = [master_seed, CALIBRATION_NAMESPACE_ID, batch_id,
              int(point["centers_per_cluster"]), int(param_index)]
    if point.get("cal_axis") in ("n", "p") and int(point["dgp_params"]["n_per_center"]) != N_PER_CENTER:
        common += [N_AXIS_NAMESPACE_ID, int(point["dgp_params"]["n_per_center"])]
    return SeedBundle(
        param_seed=_one_seed(common + [STREAM_ID["parameter"]]),
        data_seed=_one_seed(common + [int(data_index), STREAM_ID["data"]]),
        fold_seed=_one_seed(common + [int(data_index), STREAM_ID["fold"]]),
        algorithm_seed=_one_seed(common + [int(data_index), STREAM_ID["algorithm"]]),
    )


def calibration_seed_row(point: dict, param_index: int, data_index: int,
                         bundle: SeedBundle) -> dict:
    """Identity and integer seeds of one dataset."""
    return dict(
        experiment_id=EXPERIMENT_ID,
        output_version=point["output_version"],
        beta_batch=point["beta_batch"],
        point_id=point["point_id"],
        centers_per_cluster=int(point["centers_per_cluster"]),
        n_centers=int(point["n_centers"]),
        gamma_true=float(point["gamma_true"]),
        param_index=int(param_index), data_index=int(data_index),
        param_seed=int(bundle.param_seed), data_seed=int(bundle.data_seed),
        fold_seed=int(bundle.fold_seed), algorithm_seed=int(bundle.algorithm_seed),
    )
