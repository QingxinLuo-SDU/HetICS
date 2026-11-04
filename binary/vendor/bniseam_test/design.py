"""Frozen, B2-only simulation design and deterministic task identities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "b2_multiview_design.json"
SEED_DERIVATION_VERSION = "b2-multiview-seeds-v1"
_UINT32_MODULUS = 2**32 - 1


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


class FrozenMapping(Mapping[str, Any]):
    """Small deterministic, recursively immutable, pickle-safe mapping."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("FrozenMapping requires a mapping")
        items = []
        for key in sorted(values):
            if not isinstance(key, str):
                raise TypeError("parameter mapping keys must be strings")
            items.append((key, _deep_freeze(values[key])))
        self._items = tuple(items)

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self), (dict(self._items),)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"


def _stable_seed(namespace: str, *parts: Any) -> int:
    payload = json.dumps(
        [SEED_DERIVATION_VERSION, namespace, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _UINT32_MODULUS + 1


def derive_dgp_param_seed(combo_id: str, param_rep: int) -> int:
    """Derive the official parameter-RNG seed shared by ten conditional draws."""

    return 1_000_000_000 + _stable_seed("dgp-param", combo_id, int(param_rep)) % 500_000_000


def derive_dgp_data_seed(combo_id: str, param_rep: int, data_rep: int) -> int:
    """Derive the official data-RNG seed for one conditional dataset."""

    return 2_000_000_000 + _stable_seed(
        "dgp-data", combo_id, int(param_rep), int(data_rep)
    ) % 500_000_000


def derive_algorithm_seed_base(combo_id: str, param_rep: int, data_rep: int) -> int:
    """Derive the dataset identity root for all truth-free algorithm streams."""

    return _stable_seed("algorithm-dataset", combo_id, int(param_rep), int(data_rep))


def derive_algorithm_child_seed(
    base_seed: int,
    *,
    stage: str,
    direction: str = "dataset",
    logical_candidate_id: str = "dataset",
    start_index: int = 0,
) -> int:
    """Derive an explicit stage/direction/candidate/start child stream."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise ValueError("base_seed must be a positive integer")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    for name, value in (
        ("stage", stage),
        ("direction", direction),
        ("logical_candidate_id", logical_candidate_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty text")
    return _stable_seed(
        "algorithm-child",
        int(base_seed),
        stage,
        direction,
        logical_candidate_id,
        int(start_index),
    )


@dataclass(frozen=True)
class Combo:
    combo_id: str
    scan_param: str
    scan_value: str
    params: Mapping[str, Any]
    param_reps: int
    data_reps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", FrozenMapping(self.params))


@dataclass(frozen=True)
class Task:
    combo: Combo
    param_rep: int
    data_rep: int
    param_seed: int
    data_seed: int

    @property
    def task_id(self) -> str:
        return f"B2__{self.combo.combo_id}__p{self.param_rep:02d}__d{self.data_rep:02d}"

    @property
    def combo_id(self) -> str:
        return self.combo.combo_id

    @property
    def layer(self) -> str:
        return "B2"

    @property
    def algorithm_seed_base(self) -> int:
        return derive_algorithm_seed_base(self.combo_id, self.param_rep, self.data_rep)

    @property
    def split_seed(self) -> int:
        return derive_algorithm_child_seed(self.algorithm_seed_base, stage="split")

    @property
    def candidate_seed(self) -> int:
        return derive_algorithm_child_seed(self.algorithm_seed_base, stage="candidate")

    def algorithm_child_seed(
        self,
        *,
        stage: str,
        direction: str = "dataset",
        logical_candidate_id: str = "dataset",
        start_index: int = 0,
    ) -> int:
        return derive_algorithm_child_seed(
            self.algorithm_seed_base,
            stage=stage,
            direction=direction,
            logical_candidate_id=logical_candidate_id,
            start_index=start_index,
        )


def _load_design() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


_DESIGN = _load_design()
ANCHOR_COMBO_ID = _DESIGN["anchor"]["combo_id"]
ANCHOR_PARAMS: Mapping[str, Any] = FrozenMapping(_DESIGN["anchor"]["params"])
STRESS_COMBO_IDS = tuple(item["combo_id"] for item in _DESIGN["stress_combos"])


def _combo_from_record(record: dict[str, Any], anchor_params: Mapping[str, Any]) -> Combo:
    params = dict(anchor_params)
    params.update(record["param_overrides"])
    return Combo(
        combo_id=record["combo_id"],
        scan_param=record["scan_param"],
        scan_value=record["scan_value"],
        params=params,
        param_reps=record["param_reps"],
        data_reps=record["data_reps"],
    )


def _build_combos() -> tuple[Combo, ...]:
    anchor_record = _DESIGN["anchor"]
    anchor = Combo(
        combo_id=anchor_record["combo_id"],
        scan_param="anchor",
        scan_value="anchor",
        params=dict(anchor_record["params"]),
        param_reps=anchor_record["param_reps"],
        data_reps=anchor_record["data_reps"],
    )
    stresses = tuple(
        _combo_from_record(record, anchor.params)
        for record in _DESIGN["stress_combos"]
    )
    return (anchor, *stresses)


def build_tasks() -> list[Task]:
    """Build the complete frozen B2 grid without invoking or editing the DGP."""
    tasks: list[Task] = []
    for combo in _build_combos():
        for param_rep in range(1, combo.param_reps + 1):
            for data_rep in range(1, combo.data_reps + 1):
                tasks.append(
                    Task(
                        combo=combo,
                        param_rep=param_rep,
                        data_rep=data_rep,
                        param_seed=derive_dgp_param_seed(combo.combo_id, param_rep),
                        data_seed=derive_dgp_data_seed(
                            combo.combo_id, param_rep, data_rep
                        ),
                    )
                )
    return tasks
