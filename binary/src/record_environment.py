"""record_environment.py — interpreter, dependency versions and source module paths.

A failure ends with a nonzero exit code.

Usage: python -B src/record_environment.py results/environment.json
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                          # noqa: E402
import platform                                             # noqa: E402

import design_table as dt                                    # noqa: E402
import label_bridge as lb                                    # noqa: E402
from screening_bridge import load_screening, source_paths    # noqa: E402


def environment_record() -> dict:
    import importlib

    def version(name):
        try:
            return getattr(importlib.import_module(name), "__version__", "unknown")
        except ImportError:
            return None

    from aipw_sandwich import _statsmodels_module, statsmodels_import_error
    sm = _statsmodels_module()
    return {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__,
        "scipy": version("scipy"), "scikit_learn": version("sklearn"),
        "pandas": version("pandas"), "statsmodels": version("statsmodels"),
        "pytest": version("pytest"),
        "logistic_fitter": "statsmodels_glm" if sm is not None else "UNAVAILABLE",
        "statsmodels_import_error": statsmodels_import_error(),
        "sys_path_head": sys.path[:6],
        "thread_env": {v: os.environ.get(v) for v in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }

REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "pandas", "statsmodels")

for _stream in (sys.stdout, sys.stderr):                # narrow console encodings
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    argv = [a for a in argv if a != "--out"]
    out = os.path.abspath(argv[0] if argv else "results/environment.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    record = {"environment": environment_record()}
    problems: list[str] = []
    try:
        record["run_table"] = {
            "path": dt.RUN_TABLE_PATH, "source": dt.RUN_TABLE_SOURCE,
            "sha256": dt.sha256_of(dt.RUN_TABLE_PATH),
            "expected_sha256": dt.RUN_TABLE_SHA256,
            "design_revision": dt.DESIGN_REVISION}
        if record["run_table"]["sha256"] != dt.RUN_TABLE_SHA256:
            problems.append("run table checksum differs from the frozen value")
    except Exception as error:                              # noqa: BLE001
        problems.append(f"run table check failed: {type(error).__name__}: {error}")

    try:
        m6 = load_screening(lb.STEP6_DIR)
        record["source_paths"] = source_paths(m6)
        record["source_status"] = "complete"
    except Exception as error:                              # noqa: BLE001
        record["source_status"] = f"failed: {type(error).__name__}: {error}"
        problems.append(record["source_status"])
    try:
        _design, pipeline, _model, _cem = lb.load_label_package()
        record.setdefault("source_paths", {})["multiview_pipeline"] = \
            os.path.abspath(pipeline.__file__)
        record["label_package_status"] = "complete"
    except Exception as error:                              # noqa: BLE001
        record["label_package_status"] = f"failed: {type(error).__name__}: {error}"
        problems.append(record["label_package_status"])

    for name in REQUIRED_MODULES:
        if record["environment"].get(name if name != "sklearn" else "scikit_learn") is None:
            problems.append(f"required module {name} is unavailable")
    if record["environment"].get("logistic_fitter") != "statsmodels_glm":
        problems.append("formal backend statsmodels_glm is unavailable: "
                        f"{record['environment'].get('statsmodels_import_error')}")

    record["problems"] = problems
    record["status"] = "complete" if not problems else "failed"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(f"[record_environment] written {out} (status={record['status']})")
    for problem in problems:
        print(f"[record_environment] problem: {problem}", file=sys.stderr)
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
