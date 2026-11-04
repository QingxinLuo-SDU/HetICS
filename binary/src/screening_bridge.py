"""screening_bridge.py — loading of the vendored screening module and the center-to-cluster grouping.

load_screening imports vendor/step6_pkg/step6_methods.py and refuses a module of the same name from another
directory. cluster_members groups centers into analysis clusters with explicit integer cluster labels.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Mapping

import numpy as np


def load_screening(step6_dir: str):
    """Import step6_methods from step6_dir and return the module (m6)."""
    root = os.path.abspath(step6_dir)
    if not os.path.isdir(root):
        raise ValueError(f"step6_dir does not exist: {step6_dir}")
    expected = {
        "step6_methods": os.path.join(root, "step6_methods.py"),
        "chow_test": os.path.join(root, "chow_test.py"),
    }
    for name, path in expected.items():
        if not os.path.isfile(path):
            raise ValueError(f"step6_dir lacks {name}.py: {path}")
        loaded = sys.modules.get(name)
        if loaded is not None:
            actual = os.path.abspath(getattr(loaded, "__file__", "") or "")
            if actual != os.path.abspath(path):
                raise RuntimeError(f"{name} is already loaded from {actual}, not from {path}")
    if root not in sys.path:
        sys.path.insert(0, root)
    m6 = importlib.import_module("step6_methods")
    m6_file = os.path.abspath(getattr(m6, "__file__", "") or "")
    if m6_file != os.path.abspath(expected["step6_methods"]):
        raise RuntimeError(f"step6_methods was loaded from {m6_file}")
    return m6


def source_paths(m6) -> dict[str, str]:
    """File paths of the loaded vendored modules (for the environment record)."""
    paths = {"step6_methods": os.path.abspath(getattr(m6, "__file__", "") or "")}
    for name in ("chow_test", "latent_cluster_dgp_formal", "step6b_binary"):
        mod = sys.modules.get(name)
        if mod is not None:
            paths[name] = os.path.abspath(getattr(mod, "__file__", "") or "")
    return paths


def _cluster_members(env: np.ndarray, center_to_cluster: Mapping[Any, Any]) -> dict[int, list]:
    if not isinstance(center_to_cluster, Mapping):
        raise ValueError("center_to_cluster must map center codes to cluster labels")
    if len(center_to_cluster) == 0:
        raise ValueError("center_to_cluster is empty")
    observed_centers = [c.item() if isinstance(c, np.generic) else c
                        for c in np.unique(env)]
    mapping = {}
    for center, label in center_to_cluster.items():
        key = center.item() if isinstance(center, np.generic) else center
        mapping[int(key)] = int(label)
    missing = [c for c in observed_centers if int(c) not in mapping]
    if missing:
        raise ValueError(f"center_to_cluster lacks centers: {missing}")
    members: dict[int, list] = {}
    for center in observed_centers:
        members.setdefault(mapping[int(center)], []).append(center)
    return {k: members[k] for k in sorted(members)}


def cluster_members(env, center_to_cluster) -> dict[int, list]:
    """Centers of each analysis cluster, in ascending cluster order."""
    return _cluster_members(np.asarray(env), center_to_cluster)


__all__ = ["load_screening", "source_paths", "cluster_members"]
