"""v4_seeds.py — two-level seeds and common random numbers.

Integer seeds are derived with numpy.random.SeedSequence from the identity of each dataset.
  parameter stream                 no data_index: the 100 datasets of a parameter seed share the parameters
  data / fold / algorithm streams  include data_index: independent across datasets
Seeds depend on seed_group_id rather than combo_id; the delta_spec points use the anchor group "A05".
"""
from __future__ import annotations

import numpy as np

from step6x_config_v4 import BETA_BATCH_ID, MASTER_SEED, SEED_GROUP_ID, STREAM_ID
from v4_types import SeedBundle, SeedKey


def _one_seed(parts) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1, dtype=np.uint32)[0])


def seeds_for_key(key: SeedKey, master_seed: int = MASTER_SEED) -> SeedBundle:
    batch_id = BETA_BATCH_ID[key.beta_batch]
    group_id = SEED_GROUP_ID[key.seed_group_id]
    common = [master_seed, batch_id, group_id, key.param_index]
    return SeedBundle(
        param_seed=_one_seed(common + [STREAM_ID["parameter"]]),
        data_seed=_one_seed(common + [key.data_index, STREAM_ID["data"]]),
        fold_seed=_one_seed(common + [key.data_index, STREAM_ID["fold"]]),
        algorithm_seed=_one_seed(common + [key.data_index, STREAM_ID["algorithm"]]),
    )


def seed_key_for(combo: dict, param_index: int, data_index: int) -> SeedKey:
    """SeedKey of a design point with its identity fields attached."""
    return SeedKey(
        beta_batch=combo["beta_batch"], combo_id=combo["combo_id"],
        param_index=int(param_index), data_index=int(data_index),
        seed_group_id=combo["seed_group_id"],
    )
