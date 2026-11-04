"""Data-blind, within-center splitting for the multiview label test."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .multiview_types import ObservedDataView, _require_observed


_SPLIT_RULE = (
    "within-center seeded permutation; 50% label, 25% screening, "
    "25% effect; label alternates odd remainders between A and B"
)

# ── Packaged addition (the source package is unchanged) ───────────────────────
# Full-sample mode: the label, screening and effect stages each cover all N rows; A/B use the same
# seeded permutation and odd-row rule per center and partition the label pool (here the whole sample).
# The diagnostics report the overlapping stages (*_disjoint=False).
_FULL_SAMPLE_SPLIT_RULE = (
    "within-center seeded permutation; label/screening/effect each cover all rows; "
    "label pool = full sample; A/B mutually exclusive over the full label pool, "
    "odd remainders alternate between A and B"
)


@dataclass(frozen=True)
class HonestSplit:
    label: np.ndarray
    screening: np.ndarray
    effect: np.ndarray
    a: np.ndarray
    b: np.ndarray
    diagnostics: dict[str, Any]


def _center_key(center: Any) -> str:
    value = center.item() if isinstance(center, np.generic) else center
    return str(value)


def _digest(groups: dict[str, np.ndarray]) -> str:
    payload = {
        name: np.asarray(indices, dtype=np.int64).tolist()
        for name, indices in sorted(groups.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_honest_split(data: ObservedDataView, seed: int) -> HonestSplit:
    """Split each center without reading features, treatment, outcome, or truth."""

    observed = _require_observed(data)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[np.ndarray]] = {
        "label": [],
        "screening": [],
        "effect": [],
        "a": [],
        "b": [],
    }
    per_center_counts: dict[str, dict[str, int]] = {}
    odd_label_ordinal = 0

    for center in np.unique(observed.center_codes):
        center_rows = np.flatnonzero(observed.center_codes == center)
        shuffled = rng.permutation(center_rows)
        n = len(shuffled)
        n_label = (n + 1) // 2
        remaining = n - n_label
        n_screening = (remaining + 1) // 2
        n_effect = remaining - n_screening

        label = shuffled[:n_label]
        screening = shuffled[n_label : n_label + n_screening]
        effect = shuffled[n_label + n_screening :]
        n_a = n_label // 2
        if n_label % 2:
            if (seed + odd_label_ordinal) % 2:
                n_a += 1
            odd_label_ordinal += 1
        a = label[:n_a]
        b = label[n_a:]

        allocations = {
            "label": label,
            "screening": screening,
            "effect": effect,
            "a": a,
            "b": b,
        }
        for name, indices in allocations.items():
            groups[name].append(indices)
        per_center_counts[_center_key(center)] = {
            "n": n,
            **{name: len(indices) for name, indices in allocations.items()},
        }

    return _assemble(observed, groups, per_center_counts, _SPLIT_RULE,
                     stages_partition_rows=True)


def build_full_sample_split(data: ObservedDataView, seed: int) -> HonestSplit:
    """Full-sample mode: each of the three stages covers all rows; A/B partition the label pool.

    Packaged addition; `build_honest_split` of the source package is unchanged.
    A/B assignment as in the source: seeded permutation within centers, the first n_a rows to A, the rest to B,
    and odd remainders alternate between A and B by (seed + odd_ordinal) % 2.
    """

    observed = _require_observed(data)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[np.ndarray]] = {
        "label": [], "screening": [], "effect": [], "a": [], "b": [],
    }
    per_center_counts: dict[str, dict[str, int]] = {}
    odd_label_ordinal = 0

    for center in np.unique(observed.center_codes):
        center_rows = np.flatnonzero(observed.center_codes == center)
        shuffled = rng.permutation(center_rows)
        n = len(shuffled)

        # all three stages use every row of the center (label pool = whole sample).
        label = shuffled
        screening = shuffled
        effect = shuffled
        n_label = n
        n_a = n_label // 2
        if n_label % 2:
            if (seed + odd_label_ordinal) % 2:
                n_a += 1
            odd_label_ordinal += 1
        a = label[:n_a]
        b = label[n_a:]

        allocations = {
            "label": label, "screening": screening, "effect": effect, "a": a, "b": b,
        }
        for name, indices in allocations.items():
            groups[name].append(indices)
        per_center_counts[_center_key(center)] = {
            "n": n,
            **{name: len(indices) for name, indices in allocations.items()},
        }

    return _assemble(observed, groups, per_center_counts, _FULL_SAMPLE_SPLIT_RULE,
                     stages_partition_rows=False)


def _assemble(
    observed: ObservedDataView,
    groups: dict[str, list[np.ndarray]],
    per_center_counts: dict[str, dict[str, int]],
    rule: str,
    *,
    stages_partition_rows: bool,
) -> HonestSplit:
    combined = {
        name: np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
        if parts
        else np.empty(0, dtype=np.int64)
        for name, parts in groups.items()
    }
    label_screening_intersection = np.intersect1d(
        combined["label"], combined["screening"], assume_unique=True
    )
    label_effect_intersection = np.intersect1d(
        combined["label"], combined["effect"], assume_unique=True
    )
    screening_effect_intersection = np.intersect1d(
        combined["screening"], combined["effect"], assume_unique=True
    )
    a_b_intersection = np.intersect1d(
        combined["a"], combined["b"], assume_unique=True
    )
    a_b_union = np.union1d(combined["a"], combined["b"])
    primary_union = np.union1d(
        np.union1d(combined["label"], combined["screening"]),
        combined["effect"],
    )
    all_rows = np.arange(observed.n_samples, dtype=np.int64)
    per_center_count_identities = {
        center: {
            "label_screening_effect_equals_n": (
                counts["label"] + counts["screening"] + counts["effect"]
                == counts["n"]
            ),
            "a_b_equals_label": counts["a"] + counts["b"] == counts["label"],
            "each_stage_equals_n": (
                counts["label"] == counts["screening"] == counts["effect"] == counts["n"]
            ),
        }
        for center, counts in per_center_counts.items()
    }
    diagnostics = {
        "rule": rule,
        "split_mode": ("historical_half" if stages_partition_rows else "full_sample"),
        "stages_partition_rows": bool(stages_partition_rows),
        "label_equals_all_rows": np.array_equal(combined["label"], all_rows),
        "screening_equals_all_rows": np.array_equal(combined["screening"], all_rows),
        "effect_equals_all_rows": np.array_equal(combined["effect"], all_rows),
        "stratified_on_x_or_y": False,
        "per_center_counts": per_center_counts,
        "per_center_count_identities": per_center_count_identities,
        "label_screening_disjoint": label_screening_intersection.size == 0,
        "label_effect_disjoint": label_effect_intersection.size == 0,
        "screening_effect_disjoint": screening_effect_intersection.size == 0,
        "a_b_disjoint": a_b_intersection.size == 0,
        "a_b_union_equals_label": np.array_equal(a_b_union, combined["label"]),
        "label_screening_effect_union_equals_all_rows": np.array_equal(
            primary_union, all_rows
        ),
        "expected_total_row_count": int(observed.n_samples),
        "observed_total_row_count": int(primary_union.size),
        "digest_sha256": _digest(combined),
    }
    return HonestSplit(diagnostics=diagnostics, **combined)
