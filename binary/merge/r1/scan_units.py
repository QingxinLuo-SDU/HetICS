"""scan_units.py — streaming read of the shards' units tar.gz archives (not extracted to disk), producing:

  units_index.csv            one row per unit: point, version, sig_y_applied, ARI, label status (input of gates a b c)
  attribution_by_cluster.csv one row per unit x arm x analysis cluster: core covariates hit before and after the ICP
  screening_by_unit.csv      one row per unit x arm: spurious, noise and safe-set composition before and after the
                             ICP on the union of the analysis clusters
  expected_table_counts.json rows of every table recomputed per unit with summarize_formal.build_tables, used by
                             merge_tables.py to check that the shard tables come from exactly these units

Usage: python -I scan_units.py --tgz-dir <dir of the tar.gz files> --src <package src dir> --out <dir> --machines <ids>
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys
import tarfile
from multiprocessing import Pool

MACHINES = ()  # Set explicitly by --machines; identities are not part of the design.
REFERENCE_ARM = "oracle_reference"     # arms.REFERENCE_ARM: true partition, cluster label = true cluster

INDEX_FIELDS = [
    "machine", "member", "bytes", "json_error", "dir_point", "file_stem",
    "point_id", "param_rep", "data_rep", "unit_key", "structure_id",
    "design_revision", "output_version", "sig_y_applied", "gamma_logor_meta",
    "unit_complete", "status", "n_errors", "n_methods", "methods",
    "ari", "n_analysis_clusters", "n_single_center_clusters", "mixed_fraction",
    "pipeline_status", "primary_status", "labels_frozen", "fixed_refit_status",
    "fixed_refit_converged", "support_status", "ate_do"]

ATTRIB_FIELDS = [
    "machine", "unit_key", "point_id", "param_rep", "method", "analysis_cluster",
    "own_true_cluster", "ari", "n_core", "has_pre",
    "n_pre", "pre_core_hit", "pre_miss_common", "pre_miss_spec",
    "n_post", "post_core_hit", "post_miss_common", "post_miss_spec",
    "icp_removed_core", "post_not_in_pre"]


SCREEN_FIELDS = [
    "unit_key", "point_id", "param_rep", "method", "p", "n_spur", "n_noise", "n_clusters",
    "min_core_recall", "all_core", "post_size", "post_safe", "post_spur", "post_noise",
    "post_spec_foreign_any", "has_pre", "pre_size", "pre_spur", "pre_noise"]


def _screening(rec: dict) -> list[dict]:
    """One row per unit x arm: composition before and after the ICP on the union of the analysis clusters.

    Columns follow the DGP layout: A_safe (Z, shared, spec) | spurious | noise. The block sizes scale with p,
    so they are read from truth.a_safe and the denominators of feature_selection.
    """
    p = (rec.get("observed_shape") or {}).get("p")
    fs = rec.get("feature_selection") or {}
    a_safe = sorted(int(j) for j in (rec.get("truth") or {}).get("a_safe") or [])
    any_cluster = next((m for blk in fs.values() for m in blk.values()), None)
    if not a_safe or any_cluster is None:
        return []
    n_safe, n_spur, n_noise = len(a_safe), int(any_cluster["n_spur"]), int(any_cluster["n_noise"])
    assert a_safe == list(range(n_safe)) and n_safe + n_spur + n_noise == p,         f"{rec.get('unit_key')}: unexpected covariate layout"
    spur_hi = n_safe + n_spur
    rows = []
    for method, by_cluster in (rec.get("selected_by_cluster") or {}).items():
        if by_cluster is None:
            continue
        post = set().union(*[_sets(v) for v in by_cluster.values()]) if by_cluster else set()
        rec_c = [m.get("observed_core_recall") for m in (fs.get(method) or {}).values()]
        rec_c = [r for r in rec_c if r is not None]
        pre_by = ((rec.get("selections") or {}).get(method) or {}).get("pre_icp_by_cluster") or {}
        row = {"unit_key": rec["unit_key"], "point_id": rec["point_id"],
               "param_rep": rec["param_rep"], "method": method, "p": p,
               "n_spur": n_spur, "n_noise": n_noise, "n_clusters": len(by_cluster),
               "min_core_recall": min(rec_c) if rec_c else None,
               "all_core": (min(rec_c) == 1.0) if rec_c else None,
               "post_size": len(post), "post_safe": sum(j < n_safe for j in post),
               "post_spur": sum(n_safe <= j < spur_hi for j in post),
               "post_noise": sum(j >= spur_hi for j in post),
               "post_spec_foreign_any": sum(m.get("sel_spec_foreign", 0) or 0
                                            for m in (fs.get(method) or {}).values()),
               "has_pre": bool(pre_by)}
        if pre_by:
            pre = set().union(*[_sets(v) for v in pre_by.values()])
            row.update({"pre_size": len(pre),
                        "pre_spur": sum(n_safe <= j < spur_hi for j in pre),
                        "pre_noise": sum(j >= spur_hi for j in pre)})
        rows.append(row)
    return rows


def _sets(xs):
    return set(int(j) for j in xs)


def _attribution(rec: dict, machine: str) -> list[dict]:
    truth = rec.get("truth") or {}
    core_by = {int(k): _sets(v) for k, v in (truth.get("observed_core_by_cluster") or {}).items()}
    if not core_by:
        return []
    common = set.intersection(*core_by.values())      # Z ∪ shared (core shared by all clusters)
    mapping = {int(k): v for k, v in
               ((rec.get("cluster_alignment") or {}).get("mapping") or {}).items()}
    ari = (rec.get("cluster_diagnostics") or {}).get("ari")
    rows = []
    for method, by_cluster in (rec.get("selected_by_cluster") or {}).items():
        if by_cluster is None:
            continue
        pre_by = ((rec.get("selections") or {}).get(method) or {}).get("pre_icp_by_cluster") or {}
        for g, post in by_cluster.items():
            own = int(g) if method == REFERENCE_ARM else mapping.get(int(g))
            core = core_by.get(own, set()) if own is not None else set()
            post = _sets(post)
            pre_raw = pre_by.get(str(g), pre_by.get(g))
            row = {"machine": machine, "unit_key": rec["unit_key"],
                   "point_id": rec["point_id"], "param_rep": rec["param_rep"],
                   "method": method, "analysis_cluster": g, "own_true_cluster": own,
                   "ari": ari, "n_core": len(core), "has_pre": pre_raw is not None,
                   "n_post": len(post), "post_core_hit": len(post & core),
                   "post_miss_common": len((core & common) - post),
                   "post_miss_spec": len((core - common) - post)}
            if pre_raw is not None:
                pre = _sets(pre_raw)
                row.update({"n_pre": len(pre), "pre_core_hit": len(pre & core),
                            "pre_miss_common": len((core & common) - pre),
                            "pre_miss_spec": len((core - common) - pre),
                            "icp_removed_core": len((pre & core) - post),
                            "post_not_in_pre": len(post - pre)})
            rows.append(row)
    return rows


def _index_row(rec: dict) -> dict:
    meta = rec.get("unit_meta") or {}
    cd = rec.get("cluster_diagnostics") or {}
    lab = rec.get("label_status") or {}
    return {
        "point_id": rec.get("point_id"), "param_rep": rec.get("param_rep"),
        "data_rep": rec.get("data_rep"), "unit_key": rec.get("unit_key"),
        "structure_id": rec.get("structure_id"),
        "design_revision": rec.get("design_revision"),
        "output_version": rec.get("output_version"),
        "sig_y_applied": meta.get("sig_y_applied"),
        "gamma_logor_meta": meta.get("gamma_logor"),
        "unit_complete": rec.get("unit_complete"), "status": rec.get("status"),
        "n_errors": len(rec.get("errors") or []),
        "n_methods": len(rec.get("execution_methods") or []),
        "methods": "|".join(sorted(rec.get("execution_methods") or [])),
        "ari": cd.get("ari"), "n_analysis_clusters": cd.get("n_analysis_clusters"),
        "n_single_center_clusters": len(cd.get("single_center_clusters") or []),
        "mixed_fraction": (rec.get("cluster_alignment") or {}).get("mixed_fraction"),
        "pipeline_status": lab.get("pipeline_status"),
        "primary_status": lab.get("primary_status"),
        "labels_frozen": lab.get("labels_frozen"),
        "fixed_refit_status": lab.get("fixed_refit_status"),
        "fixed_refit_converged": lab.get("fixed_refit_converged"),
        "support_status": rec.get("support_status"),
        "ate_do": (rec.get("truth") or {}).get("ate_do")}


def scan_machine(job):
    machine, tgz, src, out = job
    sys.path.insert(0, src)
    import summarize_formal as sf                      # the table code of the runner

    counts = collections.Counter()
    f_idx = open(os.path.join(out, f"_units_index_{machine}.csv"), "w", newline="", encoding="utf-8")
    f_att = open(os.path.join(out, f"_attribution_{machine}.csv"), "w", newline="", encoding="utf-8")
    f_scr = open(os.path.join(out, f"_screening_{machine}.csv"), "w", newline="", encoding="utf-8")
    w_idx = csv.DictWriter(f_idx, INDEX_FIELDS)
    w_att = csv.DictWriter(f_att, ATTRIB_FIELDS)
    w_scr = csv.DictWriter(f_scr, SCREEN_FIELDS)
    w_idx.writeheader()
    w_att.writeheader()
    w_scr.writeheader()
    n = 0
    with tarfile.open(tgz, "r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            raw = tar.extractfile(member).read()
            parts = member.name.split("/")
            row = {"machine": machine, "member": member.name, "bytes": len(raw),
                   "dir_point": parts[-2] if len(parts) > 1 else "",
                   "file_stem": parts[-1][:-5]}
            try:
                rec = json.loads(raw)
            except Exception as exc:                    # empty or truncated JSON after a power loss
                row["json_error"] = repr(exc)[:200]
                w_idx.writerow(row)
                continue
            row.update(_index_row(rec))
            w_idx.writerow(row)
            for r in _attribution(rec, machine):
                w_att.writerow(r)
            for r in _screening(rec):
                w_scr.writerow(r)
            for name, rows in sf.build_tables([rec]).items():
                counts[name] += len(rows)
            n += 1
    f_idx.close()
    f_att.close()
    f_scr.close()
    print(f"[scan] {machine}: {n} units", flush=True)
    return machine, dict(counts)


def _concat(out, prefix, final):
    parts = [os.path.join(out, f"{prefix}_{m}.csv") for m in MACHINES]
    with open(os.path.join(out, final), "w", newline="", encoding="utf-8") as dst:
        for i, p in enumerate(parts):
            with open(p, encoding="utf-8") as src_f:
                header = src_f.readline()
                if i == 0:
                    dst.write(header)
                for line in src_f:
                    dst.write(line)
            os.remove(p)


def main(argv=None) -> int:
    global MACHINES
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgz-dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--machines", required=True, nargs="+", help="Shard IDs used in the input archives and res_<ID> folders")
    args = ap.parse_args(argv)
    if len(set(args.machines)) != len(args.machines) or any(not m or not all(c.isalnum() or c in "-_" for c in m) for m in args.machines):
        ap.error("Shard IDs must be unique and contain only letters, digits, hyphens or underscores.")
    MACHINES = tuple(args.machines)
    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for m in MACHINES:
        hits = glob.glob(os.path.join(args.tgz_dir, f"*_{m}_units.tar.gz"))
        if len(hits) != 1:
            raise SystemExit(f"shard {m} should have exactly one units archive, found {hits}")
        jobs.append((m, hits[0], os.path.abspath(args.src), os.path.abspath(args.out)))
    with Pool(len(jobs)) as pool:
        results = dict(pool.map(scan_machine, jobs))
    _concat(args.out, "_units_index", "units_index.csv")
    _concat(args.out, "_attribution", "attribution_by_cluster.csv")
    _concat(args.out, "_screening", "screening_by_unit.csv")
    with open(os.path.join(args.out, "expected_table_counts.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[scan] done", {m: c.get("unit_status") for m, c in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
