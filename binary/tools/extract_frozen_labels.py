"""extract_frozen_labels.py — compact table of frozen labels, seeds and check values streamed from unit JSON files.

The ablation and pooled runners reuse the learned center labels instead of learning them again (the label stage
takes about 45 minutes per unit at the paper design); the frozen labels are stored in every formal unit. The
tool reads the units from *_units.tar.gz archives (or zip files containing them) without extracting them and
writes:

  labels_<stage>.csv.gz        one row per unit: unit_key, the three seeds, frozen labels / center_map, abstention,
                               the global pooled lasso selection and ate_do (for regeneration checks)
  gate_<stage>.json.gz         reference unit per point for --gate (smallest crc32 among non-abstaining units)
  extract_<stage>_report.json  units, abstentions, duplicates and missing units per shard and point

Usage:
  python -I tools/extract_frozen_labels.py --stage r1 --points P001-P040 --out config/labels <archives>
  python -I tools/extract_frozen_labels.py --stage r2 --points P041-P054,U001-U020 --out config/labels <archives>
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import os
import re
import sys
import tarfile
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor

for _stream in (sys.stdout, sys.stderr):        # python -I ignores PYTHONIOENCODING
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MEMBER_RE = re.compile(r"runs/(?P<machine>[A-Za-z0-9_-]+)/units/(?P<point>[A-Z]\d{2,3})/(?P<key>[^/]+)\.json$")
FIELDS = ["unit_key", "point_id", "param_rep", "data_rep", "structure_id",
          "param_seed", "data_seed", "u_epsilon_seed", "design_revision", "output_version",
          "unit_complete", "source_machine", "n_rows", "p",
          "center_map_available", "labels_frozen", "pipeline_status", "primary_status",
          "frozen_labels", "center_order", "center_map",
          "pooled_lasso_status", "pooled_lasso_selected", "ate_do"]


def parse_points(text: str) -> set[str]:
    out: set[str] = set()
    for part in text.split(","):
        part = part.strip()
        m = re.fullmatch(r"([A-Z])(\d+)-[A-Z]?(\d+)", part)
        if m:
            width = len(m.group(2))
            out |= {f"{m.group(1)}{i:0{width}d}" for i in range(int(m.group(2)), int(m.group(3)) + 1)}
        elif part:
            out.add(part)
    return out


def _tar_streams(path: str):
    """Yields (inner name, readable tar.gz stream); tar.gz members of a zip are opened one by one."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("_units.tar.gz"):
                    with zf.open(info) as handle:
                        yield f"{os.path.basename(path)}!{info.filename}", handle
    else:
        with open(path, "rb") as handle:
            yield os.path.basename(path), handle


def _row(rec: dict, machine: str) -> dict:
    ls = rec.get("label_status") or {}
    bm = rec.get("base_meta") or {}
    pooled = (rec.get("selections") or {}).get("pooled_lasso") or {}
    return {
        "unit_key": rec["unit_key"], "point_id": rec["point_id"],
        "param_rep": rec["param_rep"], "data_rep": rec["data_rep"],
        "structure_id": bm.get("structure_id"),
        "param_seed": bm.get("param_seed"), "data_seed": bm.get("data_seed"),
        "u_epsilon_seed": bm.get("u_epsilon_seed"),
        "design_revision": rec.get("design_revision"), "output_version": rec.get("output_version"),
        "unit_complete": rec.get("unit_complete"), "source_machine": machine,
        "n_rows": (rec.get("observed_shape") or {}).get("n_rows"),
        "p": (rec.get("observed_shape") or {}).get("p"),
        "center_map_available": ls.get("center_map_available"),
        "labels_frozen": ls.get("labels_frozen"),
        "pipeline_status": ls.get("pipeline_status"), "primary_status": ls.get("primary_status"),
        "frozen_labels": json.dumps(ls.get("frozen_labels")),
        "center_order": json.dumps(ls.get("center_order")),
        "center_map": json.dumps((rec.get("cluster_diagnostics") or {}).get("center_map")),
        "pooled_lasso_status": pooled.get("status"),
        "pooled_lasso_selected": json.dumps(pooled.get("global_selected")),
        "ate_do": repr(float((rec.get("truth") or {})["ate_do"])),
    }


def _gate_block(rec: dict) -> dict:
    keep = ("unit_key", "point_id", "param_rep", "data_rep", "structure_id", "observed_shape")
    out = {k: rec.get(k) for k in keep}
    return out


def scan_file(args) -> dict:
    path, points = args
    rows, gate, seen, bad = [], {}, {}, []
    for label, stream in _tar_streams(path):
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                m = MEMBER_RE.search(member.name)
                if not member.isfile() or not m or m.group("point") not in points:
                    continue
                raw = tar.extractfile(member).read()
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as error:
                    bad.append(f"{label}:{member.name}: {error}")
                    continue
                if rec.get("unit_key") != m.group("key") or rec.get("point_id") != m.group("point"):
                    bad.append(f"{label}:{member.name}: unit_key/point_id of the file name and content differ")
                    continue
                rows.append(_row(rec, m.group("machine")))
                seen.setdefault(rec["unit_key"], []).append(label)
                ok = (rec.get("unit_complete") is True
                      and (rec.get("label_status") or {}).get("center_map_available"))
                if ok:
                    h = zlib.crc32(rec["unit_key"].encode())
                    cur = gate.get(rec["point_id"])
                    if cur is None or h < cur[0]:
                        gate[rec["point_id"]] = (h, _gate_block(rec))
    return {"path": path, "rows": rows, "gate": gate, "seen": seen, "bad": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="zip / tar.gz files or wildcards")
    ap.add_argument("--stage", required=True, help="r1 or r2 (output file names)")
    ap.add_argument("--points", required=True, help="e.g. P001-P040 or P041-P054,U001-U020")
    ap.add_argument("--reps", type=int, default=500, help="expected units per point")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    files = sorted({f for pat in args.inputs for f in (glob.glob(pat) or [pat])})
    points = parse_points(args.points)
    with ProcessPoolExecutor(max_workers=min(len(files), 5)) as pool:
        parts = list(pool.map(scan_file, [(f, points) for f in files]))

    rows = sorted((r for p in parts for r in p["rows"]),
                  key=lambda r: (r["point_id"], int(r["param_rep"]), int(r["data_rep"])))
    seen: dict[str, list] = {}
    for p in parts:
        for k, v in p["seen"].items():
            seen.setdefault(k, []).extend(v)
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    per_point: dict[str, int] = {}
    for r in rows:
        per_point[r["point_id"]] = per_point.get(r["point_id"], 0) + 1
    missing_points = sorted(points - set(per_point))
    short = {p: n for p, n in per_point.items() if n != args.reps}
    abstain = [r["unit_key"] for r in rows if not r["center_map_available"]]
    incomplete = [r["unit_key"] for r in rows if r["unit_complete"] is not True]
    gate = {}
    for p in parts:
        for pid, (h, block) in p["gate"].items():
            if pid not in gate or h < gate[pid][0]:
                gate[pid] = (h, block)

    os.makedirs(args.out, exist_ok=True)
    labels_path = os.path.join(args.out, f"labels_{args.stage}.csv.gz")
    with gzip.open(labels_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    gate_path = os.path.join(args.out, f"gate_{args.stage}.json.gz")
    with gzip.open(gate_path, "wt", encoding="utf-8") as handle:
        json.dump({pid: blk for pid, (_h, blk) in sorted(gate.items())}, handle)
    report = {
        "stage": args.stage, "inputs": files, "n_units": len(rows),
        "expected_units": len(points) * args.reps, "per_point": dict(sorted(per_point.items())),
        "per_machine": {m: sum(1 for r in rows if r["source_machine"] == m)
                        for m in sorted({r["source_machine"] for r in rows})},
        "duplicates": dup, "missing_points": missing_points, "points_not_at_reps": short,
        "abstained_units": abstain, "incomplete_units": incomplete,
        "bad_members": [b for p in parts for b in p["bad"]],
        "gate_points": sorted(gate),
        "design_revisions": sorted({r["design_revision"] for r in rows}),
        "output_versions": sorted({r["output_version"] for r in rows}),
    }
    with open(os.path.join(args.out, f"extract_{args.stage}_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    ok = (not dup and not missing_points and not short and not incomplete
          and not report["bad_members"] and len(rows) == report["expected_units"])
    print(f"[extract] {len(rows)}/{report['expected_units']} units; {len(abstain)} abstaining; "
          f"{len(dup)} duplicates; missing points {missing_points}; short {short}; {len(report['bad_members'])} bad members")
    print(f"[extract] gate references for {len(gate)} points -> {gate_path}")
    print(f"[extract] {'complete' if ok else 'INCOMPLETE (see report)'} -> {labels_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
