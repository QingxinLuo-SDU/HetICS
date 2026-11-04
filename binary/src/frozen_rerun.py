"""frozen_rerun.py — design rows, frozen labels and base groups shared by the ablation and pooled runners.

Each formal unit stored its learned center labels and its three seeds; tools/extract_frozen_labels.py
collects them into config/labels/labels_<stage>.csv.gz. The ablation and pooled runners regenerate the data
of a unit from the design table and the recorded seeds and reuse the frozen partition instead of learning
the labels again.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import zlib

import design_table as dt

ROUND1_TABLE = os.path.join(dt.CONFIG_DIR, "formal_run_table_v5_main500.csv")
ROUND1_TABLE_SHA256 = "43ef1eee7064ae6b38d10f11d6816a25f3589131191dc0b05ae0e65481aead7f"
LABEL_DIR = os.path.join(dt.CONFIG_DIR, "labels")
STAGES = {
    "r1": {"points": [f"P{i:03d}" for i in range(1, 41)],
           "design_revision": dt.ROUND1_DESIGN_REVISION,
           "source_output_version": "binary-formal-v5-impl-20260927-perf-sigy1-icp1e3"},
    "r2": {"points": [f"P{i:03d}" for i in range(41, 55)] + [f"U{i:03d}" for i in range(1, 21)],
           "design_revision": dt.DESIGN_REVISION,
           "source_output_version": dt.IMPLEMENTATION_VERSION},
}
ATE_TOL = 1e-12
GATE_ABS, GATE_REL = 1e-12, 1e-9


class RegenerationMismatch(RuntimeError):
    """Regenerated seeds or labels disagree with the stored unit."""


def stage_rows() -> dict[str, dict]:
    """The 74 rows with frozen labels: 40 main rows of round 1 and 34 sensitivity rows of round 2."""
    if dt.sha256_of(ROUND1_TABLE) != ROUND1_TABLE_SHA256:
        raise RuntimeError("Checksum of the round-1 run table does not match.")
    with open(ROUND1_TABLE, encoding="utf-8-sig", newline="") as handle:
        r1 = list(csv.DictReader(handle))
    r2 = [r for r in dt.load_rows(frozen_path=os.path.join(LABEL_DIR, "_no_gamma_needed.json"))
          if r["point_id"] in STAGES["r2"]["points"]]
    rows = {r["point_id"]: r for r in r1 + r2}
    if len(rows) != 74 or len(r1) != 40 or len(r2) != 34:
        raise RuntimeError(f"Unexpected number of design rows: r1={len(r1)} r2={len(r2)} total={len(rows)}")
    if {r["design_revision"] for r in r1} != {STAGES["r1"]["design_revision"]}:
        raise RuntimeError("design_revision of the round-1 table does not match.")
    return rows


def load_labels(paths: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["unit_key"] in out:
                    raise RuntimeError(f"Duplicate unit in the label table: {row['unit_key']}")
                out[row["unit_key"]] = row
    return out


def _truthy(text) -> bool:
    return str(text).strip() in ("True", "true", "1")


def frozen_center_map(lab: dict) -> dict[int, int] | None:
    """Frozen labels as a center map, cross-checked with the stored center_map; None for abstaining units."""
    if not _truthy(lab["center_map_available"]):
        return None
    stored = {int(k): int(v) for k, v in json.loads(lab["center_map"]).items()}
    order = json.loads(lab["center_order"])
    labels = json.loads(lab["frozen_labels"])
    rebuilt = {int(c): int(g) for c, g in zip(order, labels)}
    if rebuilt != stored or len(order) != len(labels):
        raise RegenerationMismatch(f"{lab['unit_key']}: frozen_labels and center_map disagree")
    return stored


def build_groups(rows: dict[str, dict], labels: dict[str, dict], points: list[str],
                 param_reps: list[int], data_reps: list[int]) -> tuple[list[dict], dict]:
    """Groups of units by (structure, param_rep, data_rep); seeds and structures are checked before running."""
    structures = dt.build_structures(list(rows.values()))
    p2s = dt.structure_of(list(rows.values()))
    groups: dict[tuple, dict] = {}
    problems = []
    for pid in points:
        sid = p2s[pid]
        for rep in param_reps:
            for d in data_reps:
                key = dt.unit_key(rows[pid], rep, d)
                lab = labels.get(key)
                if lab is None:
                    problems.append(f"{key}: missing from the label table")
                    continue
                gk = (sid, rep, d)
                if gk not in groups:
                    seeds = dt.base_seeds(sid, rep, d)
                    groups[gk] = {"base_group_key": dt.base_group_key(sid, rep, d),
                                  "structure_id": sid, "param_rep": rep, "data_rep": d,
                                  "param_seed": seeds["param_seed"],
                                  "data_seed": seeds["data_seed"],
                                  "u_epsilon_seed": dt.u_epsilon_seed(sid, rep, d),
                                  "sig_x_spec": structures[sid]["sig_x_spec"],
                                  "point_ids": [], "labels": {}}
                g = groups[gk]
                got = (lab["structure_id"], int(lab["param_seed"]), int(lab["data_seed"]),
                       int(lab["u_epsilon_seed"]))
                want = (sid, g["param_seed"], g["data_seed"], g["u_epsilon_seed"])
                if got != want:
                    problems.append(f"{key}: stored structure/seeds {got} != design {want}")
                if not _truthy(lab["unit_complete"]):
                    problems.append(f"{key}: original unit incomplete")
                g["point_ids"].append(pid)
                g["labels"][pid] = lab
    if problems:
        raise RuntimeError("Pre-run checks failed (first 10):\n  " + "\n  ".join(problems[:10])
                           + f"\n  ... {len(problems)} in total")
    ordered = sorted(groups.values(),
                     key=lambda g: zlib.crc32(g["base_group_key"].encode("utf-8")))
    return ordered, structures


def _peak_rss_gb():
    try:
        import resource
    except ImportError:                                   # Windows
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
