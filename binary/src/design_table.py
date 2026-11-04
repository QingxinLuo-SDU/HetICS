"""design_table.py — formal run tables: structure keys, task identities and design checks.

Design identities have three levels:
  1. structure identity structure_id: only the fields that determine the continuous base data. The
     continuous gamma_XY is fixed at 1; the scanned gamma_logor and the three U fields are not part of it.
     param_seed and data_seed derive from it, so paired scenarios share the same F and parameter stream.
  2. analysis identity point_id + scenario + gamma: the cache keys of labels, screening and estimation.
     algorithm_seed_base derives from it, so different gamma or U scenarios are clustered, screened and
     estimated anew.
  3. U data identity: a separate epsilon stream derived from structure, param and data (not point_id), so
     the U points of one structure share the same U and their paired point without U shares the same F.

All seeds derive from the SHA256 scheme of vendor/bniseam_test/design.py.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
VENDOR_DIR = os.path.join(_PKG_ROOT, "vendor")
STEP6_DIR = os.path.join(VENDOR_DIR, "step6_pkg")
CONFIG_DIR = os.path.join(_PKG_ROOT, "config")
RUN_TABLE_PATH = os.path.join(CONFIG_DIR, "formal_run_table_r2.csv")

# Source and byte checksum of the run table (derived from the v4 master table by tools/build_round2_table.py)
RUN_TABLE_SOURCE = ("round 2, 59 points: C001-C012, P041-P054 and U001-U020 of v4, plus S01-S13 "
                    "(calibration at a small gamma frozen after calibration); "
                    "SIG_Y 2 -> 1 and 500 param_rep x 1 data_rep throughout")
RUN_TABLE_SHA256 = "0ee4a23f402a8d453591fd6b95d09e66d75d6c6563d5b38073923ca98fe39c9c"

DESIGN_REVISION = "r2_cal3_sgamma_sens_sigy1_500_20261016"
ROUND1_DESIGN_REVISION = "v5_anchor_sigy1_main500_20260927"

# ── gamma of the S points: the run table holds a sentinel replaced by the frozen value in load_rows() ──
GAMMA_SMALL_SENTINEL = "@gamma_small"
GAMMA_SMALL_TIER = "calibration_small_gamma"
GAMMA_SMALL_FROZEN_PATH = os.path.join(_PKG_ROOT, "results", "gamma_small_frozen.json")

# ── master design (v4, 86 rows) ────────────────────────────────────────────
# Round 1 (v5) is its tier=main subset (40 rows) with SIG_Y 2 -> 1 and n_reps 10 -> 500. The master table
# names the structures (and hence the seeds) and keeps the design invariant and provenance tests.
LEGACY_RUN_TABLE_PATH = os.path.join(CONFIG_DIR, "formal_run_table_v4.csv")
LEGACY_RUN_TABLE_SHA256 = "2d90117b7da373a616125283a14be4969c07145a1ae1d69f21b01844a7e3f073"
LEGACY_DESIGN_REVISION = "v4_frozen_u07_03_cal3_20260918"
IMPLEMENTATION_VERSION = "binary-formal-r2-impl-20261016-perf-sigy1-icp1e3-sgamma"

# continuous base layer: gamma_XY_continuous is always 1 in the CSV, separate from gamma_logor
BASE_GAMMA_XY = 1.0

# ---------------------------------------------------------------------------
# Structure fields and their role in data generation
# ---------------------------------------------------------------------------
# A. keyword arguments of generate_latent_cluster_dgp (22)
DGP_KWARG_FIELDS: tuple[tuple[str, str], ...] = (
    ("n_centers", "int"), ("n_per_center", "int"), ("n_per_center_var", "float"),
    ("p_total", "int"), ("K", "int"), ("n_Z", "int"),
    ("n_causal_shared", "int"), ("n_causal_specific", "int"), ("n_spurious", "int"),
    ("rho_max", "float"), ("rho_min", "float"), ("delta_spec", "float"),
    ("nu_sd", "float"), ("alpha_ZX", "float"), ("alpha_ZY", "float"),
    ("sigma_Z", "float"), ("kappa", "float"), ("f_X", "str"), ("f_Y", "str"),
    ("rho_block", "float"), ("rho_cross", "float"),
    ("beta_mag_lo", "float"), ("beta_mag_hi", "float"), ("signed_beta", "bool"),
)
# B. fields that determine the data but are not DGP keywords
#    gamma_XY_continuous -> passed as gamma_XY, always 1;
#    centers_per_cluster -> "equal" means None (equal split in the source function);
#    SIG_X_SPEC          -> strength of spec -> X in the binary layer; draw_spec_x_coefs draws its
#                           coefficients from beta_mag_lo/hi, so it is part of the parameter stream.
BASE_EXTRA_FIELDS: tuple[str, ...] = (
    "gamma_XY_continuous", "centers_per_cluster", "SIG_X_SPEC",
)
# C. fields of the binary layer: analysis identity only
BINARY_FIELDS: tuple[str, ...] = ("gamma_logor", "SIG_X", "SIG_Y", "C0")
# D. U fields: analysis identity only
U_FIELDS: tuple[str, ...] = (
    "has_unmeasured_U", "u_shared_loading", "u_noise_loading",
    "lambda_UX", "lambda_UY",
)

STRUCTURE_FIELDS: tuple[str, ...] = tuple(
    name for name, _kind in DGP_KWARG_FIELDS) + BASE_EXTRA_FIELDS

_TRUE = {"true", "True", "TRUE", "1"}


def _cast(value: str, kind: str):
    value = value.strip()
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    if kind == "bool":
        return value in _TRUE
    return value


def load_rows(path: str = RUN_TABLE_PATH, *, verify_sha256: bool = True,
              frozen_path: str | None = None) -> list[dict]:
    """Read the round-2 run table; the byte checksum is checked by default.

    gamma_logor of the S rows is a sentinel: it is replaced by the frozen value when the frozen file exists and
    kept otherwise (the S points cannot run then; run_formal refuses them). The frozen file is only read.
    """
    if verify_sha256:
        digest = sha256_of(path)
        if digest != RUN_TABLE_SHA256:
            raise RuntimeError(
                f"Checksum of the run table does not match: {digest} != {RUN_TABLE_SHA256}; "
                f"the table must be byte-identical to {RUN_TABLE_SOURCE}")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty run table: {path}")
    for row in rows:
        if row["design_revision"] != DESIGN_REVISION:
            raise RuntimeError(
                f"{row['point_id']} has design_revision={row['design_revision']}, "
                f"expected {DESIGN_REVISION}")
    frozen = read_gamma_small(frozen_path)
    if frozen is not None:
        for row in rows:
            if row["gamma_logor"] == GAMMA_SMALL_SENTINEL:
                row["gamma_logor"] = row["scan_value"] = frozen["gamma_logor_str"]
    return rows


def read_gamma_small(frozen_path: str | None = None) -> dict | None:
    """The frozen small gamma, or None when the file does not exist; invalid content raises."""
    path = GAMMA_SMALL_FROZEN_PATH if frozen_path is None else frozen_path
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen.get("status") != "frozen" or frozen.get("design_revision") != DESIGN_REVISION:
        raise RuntimeError(f"{path} is not a valid frozen record of {DESIGN_REVISION}")
    value = float(frozen["gamma_logor_str"])
    if not (math.isfinite(value) and 0.0 < value < 1.0):
        raise RuntimeError(f"Frozen small gamma={value} is not in (0, 1)")
    return frozen


def unresolved_points(rows: list[dict]) -> list[str]:
    """Points whose gamma is still the sentinel (not calibrated yet)."""
    return [r["point_id"] for r in rows if r["gamma_logor"] == GAMMA_SMALL_SENTINEL]


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_signature(row: dict) -> tuple:
    """Structure signature: the raw strings of STRUCTURE_FIELDS in order."""
    return tuple(row[name].strip() for name in STRUCTURE_FIELDS)


def dgp_params(row: dict) -> dict[str, Any]:
    """CSV row -> keyword arguments of generate_latent_cluster_dgp (without the two seeds)."""
    params: dict[str, Any] = {name: _cast(row[name], kind)
                              for name, kind in DGP_KWARG_FIELDS}
    gamma_cont = float(row["gamma_XY_continuous"])
    if gamma_cont != BASE_GAMMA_XY:
        raise ValueError(f"{row['point_id']} has gamma_XY_continuous="
                         f"{gamma_cont}; the design requires {BASE_GAMMA_XY}")
    params["gamma_XY"] = BASE_GAMMA_XY
    spread = row["centers_per_cluster"].strip()
    if spread != "equal":
        raise ValueError(f"{row['point_id']} has centers_per_cluster={spread!r}; "
                         "only equal is supported (centers_per_cluster=None in the source function)")
    params["centers_per_cluster"] = None
    return params


@functools.lru_cache(maxsize=1)
def canonical_structure_ids() -> dict[tuple, str]:
    """Structure signature -> structure id, named after the first point_id of the structure in the v4 master table.

    Seeds derive from structure_id. The round-2 table has no main points, so naming by its own first point
    would turn STRUCT_P034 into STRUCT_U001 and change the seeds; the U points would then no longer share F
    with their paired round-1 points without U, nor the C/S points with the round-1 gamma=+1 points. The
    master table lists the P rows first, so these names equal the round-1 names (40 main rows).
    """
    rows = load_legacy_rows()
    out: dict[tuple, str] = {}
    for row in rows:
        out.setdefault(structure_signature(row), f"STRUCT_{row['point_id']}")
    return out


def build_structures(rows: list[dict]) -> dict[str, dict]:
    """Base structures named by the canonical ids of the v4 master table (see canonical_structure_ids).

    Signatures outside the master table (test fixtures only) are named after the first point_id of this
    table; static_checks asserts that every formal row is in the master table.
    Returns {structure_id: {"signature", "point_ids", "params", "sig_x_spec"}}.
    """
    canonical = canonical_structure_ids()
    order: list[tuple] = []
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        sig = structure_signature(row)
        if sig not in grouped:
            grouped[sig] = []
            order.append(sig)
        grouped[sig].append(row)
    out: dict[str, dict] = {}
    for sig in order:
        members = grouped[sig]
        structure_id = canonical.get(sig, f"STRUCT_{members[0]['point_id']}")
        out[structure_id] = {
            "structure_id": structure_id,
            "signature": dict(zip(STRUCTURE_FIELDS, sig)),
            "point_ids": [m["point_id"] for m in members],
            "params": dgp_params(members[0]),
            "sig_x_spec": float(members[0]["SIG_X_SPEC"]),
        }
    return out


def load_legacy_rows(*, verify_sha256: bool = True) -> list[dict]:
    """The 86 rows of the master design (v4), used for structure names and design checks."""
    if verify_sha256:
        digest = sha256_of(LEGACY_RUN_TABLE_PATH)
        if digest != LEGACY_RUN_TABLE_SHA256:
            raise RuntimeError(
                f"Checksum of the master run table does not match: {digest} != {LEGACY_RUN_TABLE_SHA256}")
    with open(LEGACY_RUN_TABLE_PATH, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["design_revision"] != LEGACY_DESIGN_REVISION:
            raise RuntimeError(
                f"{row['point_id']} has a design_revision different from the master design "
                f"{LEGACY_DESIGN_REVISION}")
    return rows


def structure_of(rows: list[dict]) -> dict[str, str]:
    """point_id -> structure_id."""
    mapping: dict[str, str] = {}
    for structure_id, block in build_structures(rows).items():
        for point_id in block["point_ids"]:
            mapping[point_id] = structure_id
    return mapping


# ---------------------------------------------------------------------------
# Seeds: SHA256 derivation of vendor/bniseam_test/design.py
# ---------------------------------------------------------------------------
def _design_module():
    for path in (VENDOR_DIR, STEP6_DIR):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    from bniseam_test import design
    actual = os.path.abspath(getattr(design, "__file__", "") or "")
    if not actual.startswith(os.path.abspath(VENDOR_DIR)):
        raise RuntimeError(f"bniseam_test.design comes from {actual}, not from the packaged copy")
    return design


def base_seeds(structure_id: str, param_rep: int, data_rep: int) -> dict[str, int]:
    """Two base-layer seeds, determined by the structure identity only (paired scenarios share F and parameters)."""
    design = _design_module()
    return {
        "param_seed": int(design.derive_dgp_param_seed(structure_id, param_rep)),
        "data_seed": int(design.derive_dgp_data_seed(structure_id, param_rep, data_rep)),
    }


def algorithm_seed_base(point_id: str, param_rep: int, data_rep: int) -> int:
    """Root seed of the algorithms, determined by the analysis identity (separate per gamma and U scenario)."""
    design = _design_module()
    return int(design.derive_algorithm_seed_base(point_id, param_rep, data_rep))


def u_epsilon_seed(structure_id: str, param_rep: int, data_rep: int) -> int:
    """Separate epsilon stream of U, derived from structure, param and data, without point_id.

    The U points of one structure share the same U; the stream is independent of F, the parameters and the
    binary uniforms.
    """
    design = _design_module()
    return int(design._stable_seed(
        "dgp-unmeasured-u-epsilon", structure_id, int(param_rep), int(data_rep)))


# ---------------------------------------------------------------------------
# Task expansion
# ---------------------------------------------------------------------------
def methods_of(row: dict) -> list[str]:
    return [m for m in row["execution_methods"].split("|") if m]


def estimators_of(row: dict) -> list[str]:
    return [e for e in row["estimators"].split("|") if e]


def panels_of(row: dict) -> list[str]:
    return [p for p in row["report_panels"].split("|") if p]


def unit_key(row: dict, param_rep: int, data_rep: int) -> str:
    return f"{row['point_id']}__p{int(param_rep):03d}__d{int(data_rep):02d}"


def base_group_key(structure_id: str, param_rep: int, data_rep: int) -> str:
    return f"{structure_id}__p{int(param_rep):03d}__d{int(data_rep):02d}"


def build_base_groups(rows: list[dict], param_reps: list[int],
                      data_reps: list[int]) -> list[dict]:
    """Unit of parallel scheduling: one base realization (structure x param x data) and all its analysis units.

    A process generates the continuous base data once, runs every point of the group in turn and releases
    the data.
    """
    structures = build_structures(rows)
    by_structure: dict[str, list[dict]] = {sid: [] for sid in structures}
    point_to_structure = structure_of(rows)
    for row in rows:
        by_structure[point_to_structure[row["point_id"]]].append(row)

    groups: list[dict] = []
    for structure_id, block in structures.items():
        for param_rep in param_reps:
            for data_rep in data_reps:
                seeds = base_seeds(structure_id, param_rep, data_rep)
                groups.append({
                    "base_group_key": base_group_key(structure_id, param_rep, data_rep),
                    "structure_id": structure_id,
                    "param_rep": int(param_rep),
                    "data_rep": int(data_rep),
                    "param_seed": seeds["param_seed"],
                    "data_seed": seeds["data_seed"],
                    "u_epsilon_seed": u_epsilon_seed(structure_id, param_rep, data_rep),
                    "sig_x_spec": block["sig_x_spec"],
                    "point_ids": [r["point_id"] for r in by_structure[structure_id]],
                })
    return groups


# ---------------------------------------------------------------------------
# Design checks (without the DGP)
# ---------------------------------------------------------------------------
# ── design decisions of each design revision ──────────────────────────────────
# Only design decisions are registered (points, tiers, replications, attempts); derived quantities
# (number of structures) are recorded but not asserted. A design change must be registered here.
DESIGN_EXPECTATIONS: dict[str, dict[str, Any]] = {
    LEGACY_DESIGN_REVISION: {
        "n_rows": 86,
        "tiers": {"main": 40, "delta_sensitivity": 7, "correlation_sensitivity": 7,
                  "calibration_added_null": 12, "unmeasured_sensitivity": 20},
        "calibration_member": 26, "has_unmeasured_U": 20,
        "method_slots_per_rep": 334,        # 60 points x 4 arms + 14 points x 5 arms + 12 points x 2 arms
        "planned_estimator_attempts_total": 1_002_000,
        "reps_per_point": 1000,
        "calibration_pairing": "complete",
    },
    ROUND1_DESIGN_REVISION: {
        "n_rows": 40,
        "tiers": {"main": 40},
        # 14 = points with report_panels "main|calibration" (the gamma=+1 half of the calibration pairs);
        # their gamma=0 partners C001-C012 are not in this design.
        "calibration_member": 14, "has_unmeasured_U": 0,
        "method_slots_per_rep": 174,        # 26 points x 4 arms + 14 points x 5 arms
        "planned_estimator_attempts_total": 261_000,
        "reps_per_point": 500,
        "calibration_pairing": "intentionally_incomplete",
    },
    DESIGN_REVISION: {
        "n_rows": 59,
        "tiers": {"calibration_added_null": 12, GAMMA_SMALL_TIER: 13,
                  "delta_sensitivity": 7, "correlation_sensitivity": 7,
                  "unmeasured_sensitivity": 20},
        # 25 = 12 gamma=0 points (C) + 13 small-gamma points (S); the gamma=+1 half is in round 1 (P006/P029-P040)
        "calibration_member": 25, "has_unmeasured_U": 20,
        "method_slots_per_rep": 186,        # 25 points x 2 arms + 34 points x 4 arms
        "planned_estimator_attempts_total": 279_000,
        "reps_per_point": 500,
        # pairing partners may be in round 1 (P004 / P029-P040 / P001-P014): checked by master-table structure
        "calibration_pairing": "cross_round",
    },
}


def static_checks(rows: list[dict] | None = None) -> dict[str, Any]:
    rows = load_rows() if rows is None else rows
    problems: list[str] = []
    checks: dict[str, Any] = {}

    revision = rows[0]["design_revision"]
    exp = DESIGN_EXPECTATIONS.get(revision)
    if exp is None:
        raise RuntimeError(
            f"design_revision={revision} has no registered expectations; registered: "
            f"{sorted(DESIGN_EXPECTATIONS)}")
    checks["design_revision"] = revision
    checks["n_rows"] = len(rows)
    if len(rows) != exp["n_rows"]:
        problems.append(f"run table should have {exp['n_rows']} rows, has {len(rows)}")
    if len({r["design_revision"] for r in rows}) != 1:
        problems.append("several design_revision values in one table")

    point_ids = [r["point_id"] for r in rows]
    if len(set(point_ids)) != len(point_ids):
        problems.append("duplicate point_id")

    signatures = {}
    for row in rows:
        key = (structure_signature(row), row["gamma_logor"].strip(),
               row["has_unmeasured_U"].strip())
        signatures.setdefault(key, []).append(row["point_id"])
    dupes = {k: v for k, v in signatures.items() if len(v) > 1}
    checks["duplicate_scientific_configs"] = {str(k): v for k, v in dupes.items()}
    if dupes:
        problems.append(f"duplicate scientific configurations: {list(dupes.values())}")

    structures = build_structures(rows)
    checks["n_structures"] = len(structures)   # derived: recorded, not asserted
    sizes: dict[int, int] = {}
    for block in structures.values():
        sizes[len(block["point_ids"])] = sizes.get(len(block["point_ids"]), 0) + 1
    checks["structure_group_sizes"] = dict(sorted(sizes.items()))

    tiers: dict[str, int] = {}
    panels: dict[str, int] = {}
    for row in rows:
        tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
        panels[row["report_panels"]] = panels.get(row["report_panels"], 0) + 1
    checks["tier_counts"] = tiers
    checks["panel_counts"] = panels
    expected_tiers = exp["tiers"]
    if tiers != expected_tiers:
        problems.append(f"tier counts differ: {tiers} != {expected_tiers}")

    n_cal = sum(1 for r in rows if r["calibration_member"] == "true")
    n_u = sum(1 for r in rows if r["has_unmeasured_U"] == "true")
    checks["calibration_member"] = n_cal
    checks["has_unmeasured_U"] = n_u
    if n_cal != exp["calibration_member"]:
        problems.append(f"calibration_member should be {exp['calibration_member']}, is {n_cal}")
    if n_u != exp["has_unmeasured_U"]:
        problems.append(f"has_unmeasured_U should be {exp['has_unmeasured_U']}, is {n_u}")

    total = 0
    method_total = 0
    for row in rows:
        methods = methods_of(row)
        estimators = estimators_of(row)
        n_methods = int(row["n_methods"])
        n_est = int(row["n_estimators"])
        planned = int(row["planned_estimator_attempts"])
        reps = int(row["n_reps"])
        if len(methods) != n_methods:
            problems.append(f"{row['point_id']} lists {len(methods)} execution_methods"
                            f" != n_methods {n_methods}")
        if len(estimators) != n_est:
            problems.append(f"{row['point_id']} has a wrong number of estimators")
        if n_methods * n_est * reps != planned:
            problems.append(f"{row['point_id']} has wrong planned_estimator_attempts")
        if int(row["param_reps"]) * int(row["data_reps"]) != reps:
            problems.append(f"{row['point_id']} param_reps×data_reps != n_reps")
        if reps != exp["reps_per_point"]:
            problems.append(f"{row['point_id']} n_reps={reps} != {exp['reps_per_point']} of this design")
        total += planned
        method_total += n_methods
    checks["method_slots_per_rep"] = method_total
    checks["planned_estimator_attempts_total"] = total
    if method_total != exp["method_slots_per_rep"]:
        problems.append(f"method slots per replication should be {exp['method_slots_per_rep']}, are {method_total}")
    if total != exp["planned_estimator_attempts_total"]:
        problems.append(f"total attempts should be {exp['planned_estimator_attempts_total']:,}, are {total:,}")

    if revision == LEGACY_DESIGN_REVISION:
        first_batch_units = len(rows) * 10
        first_batch_attempts = method_total * 3 * 10
        checks["first_batch_units"] = first_batch_units
        checks["first_batch_attempts"] = first_batch_attempts
        if (first_batch_units, first_batch_attempts) != (860, 10_020):
            problems.append("first batch 860 / 10,020 does not add up")
        checks["remainder_if_first_batch_counts"] = {
            "units": 86_000 - first_batch_units,
            "attempts": 1_002_000 - first_batch_attempts,
            "note": "remaining size if the first batch counted towards the formal denominators; "
                    "the computation starts or resumes nothing",
        }
    else:
        # this design runs in full, without a first batch
        checks["planned_units"] = len(rows) * exp["reps_per_point"]
        checks["planned_attempts"] = method_total * 3 * exp["reps_per_point"]
        if checks["planned_attempts"] != total:
            problems.append(f"units x arms x estimators {checks['planned_attempts']:,}"
                            f" differs from the row total {total:,}")

    point_to_structure = structure_of(rows)
    universe = {r["point_id"]: r for r in rows}
    if exp["calibration_pairing"] == "cross_round":
        # partners may be in round 1: "same structure" is checked by the canonical master-table names
        canonical = canonical_structure_ids()
        off_canon = [r["point_id"] for r in rows
                     if structure_signature(r) not in canonical]
        if off_canon:
            problems.append(f"structures of these points are not in the v4 master table; their seeds would differ from round 1: {off_canon}")
        for legacy in load_legacy_rows():
            universe.setdefault(legacy["point_id"], legacy)
            point_to_structure.setdefault(
                legacy["point_id"], canonical[structure_signature(legacy)])
    pair_problems = []
    for row in rows:
        pair = row["pair_reference_point_id"].strip()
        if not pair:
            continue
        if pair not in point_to_structure:
            pair_problems.append(f"{row['point_id']}: paired point {pair} does not exist")
        elif point_to_structure[pair] != point_to_structure[row["point_id"]]:
            pair_problems.append(
                f"{row['point_id']} and its paired point {pair} are in different structures: "
                f"{point_to_structure[row['point_id']]} vs {point_to_structure[pair]}")
    checks["pair_reference_checked"] = sum(
        1 for r in rows if r["pair_reference_point_id"].strip())
    if pair_problems:
        problems.extend(pair_problems)

    if exp["calibration_pairing"] == "cross_round":
        # the partner of each small-gamma point is the gamma=0 point of its structure; one per calibration structure
        small = [r for r in rows if r["tier"] == GAMMA_SMALL_TIER]
        for row in small:
            partner = universe.get(row["pair_reference_point_id"].strip())
            if partner is None or partner["gamma_logor"].strip() != "0":
                problems.append(f"{row['point_id']}: partner is not a gamma=0 point")
        small_structs = [point_to_structure[r["point_id"]] for r in small]
        checks["small_gamma_structures"] = sorted(set(small_structs))
        if len(set(small_structs)) != len(small_structs) or len(small_structs) != 13:
            problems.append(f"small-gamma points should cover 13 calibration structures, cover {small_structs}")
        values = {r["gamma_logor"] for r in small}
        checks["small_gamma_value"] = sorted(values)
        if len(values) != 1:
            problems.append(f"the 13 small-gamma points must share one gamma, have {sorted(values)}")

    # every calibration null point coexists with its gamma=+1 partner of the same structure
    cal_struct = {}
    for row in rows:
        if row["calibration_member"] != "true":
            continue
        cal_struct.setdefault(point_to_structure[row["point_id"]], set()).add(
            row["gamma_logor"].strip())
    missing_pairs = [s for s, g in cal_struct.items() if g != {"0", "1"}]
    checks["calibration_pairing"] = exp["calibration_pairing"]
    if exp["calibration_pairing"] == "cross_round":
        missing_pairs, cal_struct = [], {}     # checked above across rounds
    elif exp["calibration_pairing"] == "intentionally_incomplete":
        # this design runs the main analysis only: the gamma=0 half (C001-C012) of the calibration pairs is
        # absent, recorded as an intended gap
        checks["calibration_structures_without_gamma0"] = sorted(missing_pairs)
        checks["calibration_note"] = (
            "this design does not run the calibration panel; the gamma=0 partners of these structures are "
            "not in the table. The calibration panel needs a separate round with C001-C012")
        missing_pairs = []
        cal_struct = {}
    checks["calibration_structures"] = len(cal_struct)
    if exp["calibration_pairing"] == "complete" and len(cal_struct) != 13:
        problems.append(f"there should be 13 calibration structures, there are {len(cal_struct)}")
    if missing_pairs:
        problems.append(f"calibration structures lack gamma 0/+1 pairs: {missing_pairs}")

    checks["problems"] = problems
    checks["status"] = "ok" if not problems else "failed"
    return checks


__all__ = [
    "DESIGN_REVISION", "ROUND1_DESIGN_REVISION", "IMPLEMENTATION_VERSION", "BASE_GAMMA_XY",
    "GAMMA_SMALL_SENTINEL", "GAMMA_SMALL_TIER", "GAMMA_SMALL_FROZEN_PATH",
    "read_gamma_small", "unresolved_points", "canonical_structure_ids",
    "RUN_TABLE_PATH", "RUN_TABLE_SOURCE", "RUN_TABLE_SHA256",
    "LEGACY_RUN_TABLE_PATH", "LEGACY_DESIGN_REVISION", "load_legacy_rows",
    "STRUCTURE_FIELDS", "DGP_KWARG_FIELDS", "BINARY_FIELDS", "U_FIELDS",
    "load_rows", "sha256_of", "dgp_params", "structure_signature",
    "build_structures", "structure_of", "base_seeds", "algorithm_seed_base",
    "u_epsilon_seed", "methods_of", "estimators_of", "panels_of",
    "unit_key", "base_group_key", "build_base_groups", "static_checks",
]
