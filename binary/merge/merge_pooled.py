"""merge_pooled.py — summary tables of the pooled comparators (run_pooled.py units).

  python -I merge/merge_pooled.py --inputs <run directories, .tar.gz or .zip containing .tar.gz> ... --out <dir>

Unit files are read from paths ending in <r1|r2>/units/<point>/<unit>.json (gate/ directories are skipped); the
directory above the stage is recorded as the shard. Only the arms of pooled_arms.POOLED_ARMS are summarized.
  Estimation (overall ATE, one estimate on all centers): bias = mean error; empirical_se = SD of the errors
    (ddof=1); mse = bias² + empirical_se²; coverage = share of valid intervals covering the truth; points with
    gamma = 0 report type1_error and the others power (mean of reject_zero over valid intervals).
  Selection (one set evaluated against every true cluster): core_recall = minimum over true clusters of the
    observed-core recall; precision = selected safe-set covariates / selected (empty selections not counted);
    spurious_recall = selected spurious / spurious; noise_rate = selected noise / noise;
    adjustment_set_size = number selected.
Duplicates (same output version and unit_key) are kept once and the largest AIPW difference is recorded; other
output versions are skipped and counted; units with unit_complete != True count as failures.
Output: raw_effects.csv.gz, raw_selection.csv.gz, est_pooled.csv, sel_pooled.csv, counts.csv, qa.json.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pooled_arms import POOLED_ARMS  # noqa: E402

UNIT_RE = re.compile(r"(?:^|/)([^/]+)/(r1|r2)/units/([^/]+)/([^/]+)\.json$")
# runner output versions: the current runner, the runs reported in the paper (pooled lasso, PDS, OAL and the
# A_safe reference), and the run that added hdps_fp to the same units
VERSIONS = {"binary-pooled-v3-20261104": "pooled", "binary-fully-pooled-v2-20261028": "pooled",
            "binary-fully-pooled-fpx-v1-20261031": "hdps"}
SAFE = ("Z", "shared", "spec_own", "spec_foreign")
N_REPS = 500


def _from_tar(fileobj=None, name=None):
    with tarfile.open(name=name, fileobj=fileobj, mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and UNIT_RE.search(m.name) and "/gate/" not in m.name:
                yield m.name, tf.extractfile(m).read()


def iter_units(src: Path):
    """(member path, raw bytes) from a directory, a tar.gz, or a zip of tar.gz files."""
    if src.is_dir():
        for f in sorted(src.rglob("*.json")):
            p = f.as_posix()
            if UNIT_RE.search(p) and "/gate/" not in p:
                yield p, f.read_bytes()
    elif src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for mem in z.namelist():
                if mem.endswith((".tar.gz", ".tgz")):
                    with z.open(mem) as h:
                        yield from _from_tar(fileobj=io.BytesIO(h.read()))
    else:
        yield from _from_tar(name=str(src))


def parse_unit(rec: dict, armset: str, machine: str, stage: str):
    base = {"armset": armset, "machine": machine, "stage": stage, "unit_key": rec["unit_key"],
            "point_id": rec["point_id"], "param_rep": int(rec["param_rep"]), "data_rep": int(rec["data_rep"]),
            "gamma_logor": rec.get("gamma_logor")}
    eff, sel = [], []
    for arm, by_est in (rec.get("effects") or {}).items():
        if arm not in POOLED_ARMS:
            continue
        for est, rows in by_est.items():
            ov = [r for r in rows if r.get("stage") == "overall"]
            assert len(ov) == 1, (rec["unit_key"], arm, est)
            r = ov[0]
            eff.append({**base, "method": arm, "estimator": est, "estimate": r.get("estimate"), "error": r.get("error"),
                        "se": r.get("se"), "ci_valid": bool(r.get("ci_valid")), "cover_true": r.get("cover_true"),
                        "reject_zero": r.get("reject_zero"), "point_status": r.get("point_status"),
                        "target_different": bool(r.get("target_different"))})
    for arm, by_k in (rec.get("feature_selection") or {}).items():
        if arm not in POOLED_ARMS:
            continue
        ks = list(by_k.values())
        safe = {sum(int(k[f"sel_{f}"]) for f in SAFE) for k in ks}
        assert len(safe) == 1, f"{rec['unit_key']} {arm}: safe-set counts differ between true clusters {safe}"
        k0 = ks[0]
        core = [k["observed_core_recall"] for k in ks if k.get("observed_core_recall") is not None]
        n = int(k0["n_selected"])
        sel.append({**base, "method": arm, "adjustment_set_size": n, "core_recall": min(core) if core else np.nan,
                    "precision": safe.pop() / n if n > 0 else np.nan,
                    "spurious_recall": int(k0["sel_spur"]) / int(k0["n_spur"]),
                    "noise_rate": int(k0["sel_noise"]) / int(k0["n_noise"])})
    return eff, sel


def summarize_effects(eff: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (pid, arm, est), g in eff.groupby(["point_id", "method", "estimator"]):
        err = g.error.dropna().astype(float)
        valid = g[g.ci_valid]
        cov = valid.cover_true.dropna().astype(float)
        rej = valid.reject_zero.dropna().astype(float)
        bias = err.mean() if len(err) else np.nan
        esd = err.std(ddof=1) if len(err) > 1 else np.nan
        gam = set(g.gamma_logor.dropna().astype(float))
        assert len(gam) <= 1, (pid, gam)
        zero = gam == {0.0}
        out.append({"point_id": pid, "method": arm, "estimator": est, "level": "overall", "n_attempted": len(g),
                    "n_valid": len(err), "n_valid_ci": len(valid), "bias": bias, "empirical_se": esd,
                    "mse": bias ** 2 + esd ** 2, "coverage": cov.mean() if len(cov) else np.nan,
                    "type1_error": rej.mean() if zero and len(rej) else np.nan,
                    "power": rej.mean() if not zero and len(rej) else np.nan,
                    "machines": "+".join(sorted(g.machine.unique()))})
    return pd.DataFrame(out)


def summarize_selection(sel: pd.DataFrame) -> pd.DataFrame:
    g = sel.groupby(["point_id", "method"])
    return g.agg(n_units=("unit_key", "size"), core_recall=("core_recall", "mean"), precision=("precision", "mean"),
                 spurious_recall=("spurious_recall", "mean"), noise_rate=("noise_rate", "mean"),
                 adjustment_set_size=("adjustment_set_size", "mean")).reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qa = Counter()
    seen: dict[tuple, dict] = {}          # (version, unit_key) -> {arm: overall AIPW estimate}, for duplicates
    dup_diff = 0.0
    eff_rows, sel_rows = [], []
    for src in map(Path, a.inputs):
        for name, raw in iter_units(src):
            machine, stage, _point, _ = UNIT_RE.search(name).groups()
            qa["files"] += 1
            rec = json.loads(raw.decode("utf-8"))
            armset = VERSIONS.get(rec.get("output_version"))
            if armset is None:
                qa[f"skip_version::{rec.get('output_version')}"] += 1
                continue
            if rec.get("unit_complete") is not True:
                qa[f"failed::{armset}"] += 1
                continue
            key = (armset, rec["unit_key"])
            aipw = {arm: [r for r in by["aipw"] if r["stage"] == "overall"][0].get("estimate")
                    for arm, by in rec["effects"].items() if "aipw" in by and arm in POOLED_ARMS}
            if key in seen:
                qa[f"duplicate::{armset}"] += 1
                for arm in set(seen[key]) & set(aipw):
                    if seen[key][arm] is not None and aipw[arm] is not None:
                        dup_diff = max(dup_diff, abs(seen[key][arm] - aipw[arm]))
                continue
            seen[key] = aipw
            e, s = parse_unit(rec, armset, machine, stage)
            eff_rows += e
            sel_rows += s
            qa[f"units::{armset}"] += 1
    eff, sel = pd.DataFrame(eff_rows), pd.DataFrame(sel_rows)
    if eff.empty:
        raise SystemExit("[merge] no unit was read; check --inputs")
    dup_arms = eff.groupby(["unit_key", "method", "estimator"]).size()
    if (dup_arms > 1).any():
        raise SystemExit(f"[merge] an arm appears in several output versions of a unit: {dup_arms[dup_arms > 1].index[:3].tolist()}")
    eff.to_csv(out / "raw_effects.csv.gz", index=False)
    sel.to_csv(out / "raw_selection.csv.gz", index=False)
    summarize_effects(eff).to_csv(out / "est_pooled.csv", index=False)
    summarize_selection(sel).to_csv(out / "sel_pooled.csv", index=False)
    cnt = (eff.drop_duplicates(["armset", "unit_key"]).groupby(["armset", "stage", "point_id"])
           .agg(n_units=("unit_key", "size"), machines=("machine", lambda m: "+".join(sorted(set(m))))).reset_index())
    cnt["complete"] = cnt.n_units == N_REPS
    cnt.to_csv(out / "counts.csv", index=False)
    qa_out = {"inputs": a.inputs, **dict(qa), "max_duplicate_aipw_diff": dup_diff,
              "points_incomplete": int((~cnt.complete).sum()), "points_complete": int(cnt.complete.sum())}
    (out / "qa.json").write_text(json.dumps(qa_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[merge] {qa['files']} unit files: {qa['units::pooled']} pooled and {qa['units::hdps']} hdps units; "
          f"{qa['duplicate::pooled'] + qa['duplicate::hdps']} duplicates (largest estimate difference {dup_diff:.1e}), "
          f"{qa['failed::pooled'] + qa['failed::hdps']} failed; {int(cnt.complete.sum())} points complete, "
          f"{int((~cnt.complete).sum())} incomplete -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
