"""check_deps.py — dependency check, installation plan and verification.

Three groups:
  * REQUIRED: numpy / scipy / scikit-learn / pandas.
  * BACKEND: statsmodels, the formal logistic backend (statsmodels_glm); when it is missing the run does not
    switch to IRLS, and install_deps.sh fails.
  * TOOLING: pytest (the tests must actually run).

`--plan` lists the distributions missing from the dependency closure, for offline installation.

Usage:
    python -B src/check_deps.py            # versions and missing packages
    python -B src/check_deps.py --full     # also interpreter, platform and module paths
    python -B src/check_deps.py --plan     # distributions to install (one per line)
"""

from __future__ import annotations

import importlib
import json
import platform
import sys

REQUIRED = ("numpy", "scipy", "sklearn", "pandas")

# Every reproduction environment must use these versions; otherwise the data of a seed, or the estimates
# on the same data, may differ and results cannot be combined.
PINNED = {"numpy": "1.26.4", "scipy": "1.14.1", "sklearn": "1.6.1",
          "pandas": "2.2.3", "statsmodels": "0.14.4"}
BACKEND = ("statsmodels",)
TOOLING = ("pytest",)
OPTIONAL = BACKEND + TOOLING

# import name -> distribution name
DIST_OF = {"sklearn": "scikit-learn", "numpy": "numpy", "scipy": "scipy",
           "pandas": "pandas", "statsmodels": "statsmodels", "pytest": "pytest",
           "patsy": "patsy", "packaging": "packaging",
           "iniconfig": "iniconfig", "pluggy": "pluggy"}

# Dependency closure: top-level import name -> every import name it needs in this environment (itself
# included). Only the part covered by the packaged wheels; numpy/scipy/pandas belong to the base Python
# environment. colorama is needed by pytest on Windows only.
CLOSURE = {
    "statsmodels": ("statsmodels", "patsy", "packaging"),
    "pytest": ("pytest", "iniconfig", "pluggy", "packaging"),
}


def _version(name: str):
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except ImportError:
        return None


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def missing_distributions() -> list[str]:
    """Distributions of the dependency closure that are missing (deduplicated, in a fixed order)."""
    plan: list[str] = []
    for top in BACKEND + TOOLING:
        if _importable(top):
            continue
        for member in CLOSURE.get(top, (top,)):
            if _importable(member):
                continue
            dist = DIST_OF.get(member, member)
            if dist not in plan:
                plan.append(dist)
    return plan


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--plan" in argv:
        for dist in missing_distributions():
            print(dist)
        return 0

    record: dict = {"packages": {}, "optional_packages": {}}
    for name in REQUIRED:
        record["packages"][name] = _version(name)
    for name in OPTIONAL:
        record["optional_packages"][name] = _version(name)
    record["missing"] = [k for k, v in record["packages"].items() if v is None]
    record["missing_optional"] = [k for k, v in record["optional_packages"].items()
                                  if v is None]
    record["pytest_available"] = record["optional_packages"]["pytest"] is not None
    record["statsmodels_available"] = record["optional_packages"]["statsmodels"] is not None
    # the formal backend is statsmodels_glm: a missing statsmodels is a failure, with no switch to IRLS
    record["formal_logistic_backend"] = "statsmodels_glm"
    record["formal_backend_available"] = record["statsmodels_available"]
    record["missing_distributions"] = missing_distributions()
    every = {**record["packages"], **record["optional_packages"]}
    record["pinned"] = PINNED
    record["version_mismatch"] = {k: every.get(k) for k, v in PINNED.items()
                                  if every.get(k) != v}
    record["module_paths"] = {}
    for name in REQUIRED + OPTIONAL:
        try:
            record["module_paths"][name] = getattr(
                importlib.import_module(name), "__file__", "unknown")
        except ImportError:
            record["module_paths"][name] = None
    if "--full" in argv:
        record["python"] = sys.version
        record["python_executable"] = sys.executable
        record["platform"] = platform.platform()
        record["sys_path_head"] = sys.path[:6]
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 1 if (record["missing"] or record["version_mismatch"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
