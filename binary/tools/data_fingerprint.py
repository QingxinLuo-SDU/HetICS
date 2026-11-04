"""data_fingerprint.py — whether the data generated from the same seeds are bitwise identical on two machines.

For comparing Python environments (the formal data use numpy 1.26.4). Run the script once on each machine and
compare the last two lines:

    DATA_DIGEST      <64 hex>   raw random streams and the continuous data, U and binary data of six U-E
                                structures and the anchor (param_rep 1 and 500)
    ESTIMATE_DIGEST  <64 hex>   one oracle AIPW estimate and SE (scipy/statsmodels agreement)

Equal DATA_DIGEST: the data of a seed are bitwise identical and results can be combined and paired with round 1.
Different: see the per-item table; when only the `rng_*` items agree, the difference comes from the BLAS/LAPACK
kernels (different CPUs), not from the Python version.

Usage: python -B tools/data_fingerprint.py --out results/data_fingerprint_<machine>.json
       python -B tools/data_fingerprint.py --compare A.json B.json
About 2-4 minutes, one process, peak memory about 6 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "vendor"),
           os.path.join(ROOT, "vendor", "step6_pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np                                               # noqa: E402

# six U-E points and the anchor (U006); first and last param_rep
POINTS = ("U015", "U016", "U017", "U018", "U019", "U020", "U006")
PARAM_REPS = (1, 500)


def _h(arr) -> dict:
    a = np.ascontiguousarray(np.asarray(arr))
    digest = hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()
    out = {"sha256": digest}
    if a.dtype.kind == "f":
        out["sum"] = repr(float(a.sum()))           # how far the values differ when the hashes differ
    return out


def fingerprint() -> dict:
    import design_table as dt
    import formal_dgp
    from aipw_sandwich import fit_partition_aipw
    from run_formal import _reference_selection

    items: dict[str, dict] = {}
    rng = np.random.default_rng(20261016)
    items["rng_standard_normal"] = _h(rng.standard_normal(1_000_000))
    items["rng_random"] = _h(rng.random(1_000_000))

    rows = {r["point_id"]: r for r in dt.load_rows()}
    groups = {(g["structure_id"], g["param_rep"]): g
              for g in dt.build_base_groups(list(rows.values()), list(PARAM_REPS), [1])}
    structures = dt.build_structures(list(rows.values()))
    point_to_structure = dt.structure_of(list(rows.values()))
    _gen, b6 = formal_dgp.load_sources()
    estimate = None
    for pid in POINTS:
        row = rows[pid]
        sid = point_to_structure[pid]
        for rep in PARAM_REPS:
            group = groups[(sid, rep)]
            tag = f"{pid}_p{rep:03d}"
            continuous, _meta = formal_dgp.generate_base(group, structures)
            for key in ("F", "X_treat", "Y", "env", "true_cluster_sample"):
                items[f"{tag}_{key}"] = _h(continuous[key])
            u, _s, _um = formal_dgp.unmeasured_u_for_base(
                continuous, group, float(structures[sid]["params"]["rho_block"]))
            items[f"{tag}_U"] = _h(u)
            observed, truth, _m = formal_dgp.binary_unit(continuous, row, group, b6, u=u)
            items[f"{tag}_X_bin"] = _h(observed.treatment)
            items[f"{tag}_Y_bin"] = _h(observed.outcome)
            items[f"{tag}_tau_do_i"] = _h(truth["tau_do_i"])
            if estimate is None:                    # once only: the smallest, U015 p001
                env = np.asarray(observed.center_codes)
                mapping, sel = _reference_selection(truth, env)
                res = fit_partition_aipw(
                    np.asarray(observed.features, float),
                    np.asarray(observed.treatment).astype(int),
                    np.asarray(observed.outcome).astype(int), env, mapping,
                    sel["selected_by_cluster"], backend="statsmodels_glm")
                estimate = {"point": tag, "estimate": repr(float(res["estimate"])),
                            "se": repr(float(res["se"]))}
            del continuous, u, observed, truth
            print(f"[fingerprint] {tag} done", flush=True)

    data_digest = hashlib.sha256("".join(
        f"{k}={v['sha256']};" for k, v in sorted(items.items())).encode()).hexdigest()
    est_digest = hashlib.sha256(json.dumps(estimate, sort_keys=True).encode()).hexdigest()
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "platform": platform.platform()}
    for mod in ("scipy", "sklearn", "pandas", "statsmodels"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    try:
        from threadpoolctl import threadpool_info
        env["blas"] = [{k: v for k, v in d.items() if k in
                        ("internal_api", "version", "architecture", "num_threads")}
                       for d in threadpool_info()]
    except Exception:                                           # noqa: BLE001
        env["blas"] = None
    return {"data_digest": data_digest, "estimate_digest": est_digest,
            "estimate": estimate, "environment": env,
            "points": list(POINTS), "param_reps": list(PARAM_REPS), "items": items}


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    diff = [k for k in sorted(set(a["items"]) | set(b["items"]))
            if a["items"].get(k, {}).get("sha256") != b["items"].get(k, {}).get("sha256")]
    print(f"A: {a['environment']}\nB: {b['environment']}")
    for k in diff:
        print(f"  differs {k}: sum {a['items'].get(k, {}).get('sum')} vs "
              f"{b['items'].get(k, {}).get('sum')}")
    print(f"{len(a['items'])} data items, {len(diff)} differ; "
          f"estimate {a['estimate']} vs {b['estimate']}")
    return 0 if not diff and a["estimate_digest"] == b["estimate_digest"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    record = fingerprint()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1)
    env = record["environment"]
    print(f"[fingerprint] Python {env['python']}  numpy {env['numpy']}  scipy {env['scipy']}"
          f"  sklearn {env['sklearn']}  pandas {env['pandas']}  statsmodels {env['statsmodels']}")
    print(f"DATA_DIGEST      {record['data_digest']}")
    print(f"ESTIMATE_DIGEST  {record['estimate_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
