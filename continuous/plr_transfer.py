"""plr_transfer.py — reference PLR with true clusters and S_k^*, and sEAM assignment margins."""
from __future__ import annotations

import numpy as np

from whole_center_plr import fit_whole_center_plr

NAN = float("nan")
SEAM_MARGIN_LOW = 0.5      # assignment margins below this value are counted as indecisive


def fit_reference_plr(data: dict, fold_seed: int) -> dict:
    """Whole-center PLR with the true cluster labels and the target set A_core[k]."""
    out = {}
    env = np.asarray(data["env"]).astype(int)
    for k, selected in data["A_core_by_cluster"].items():
        centers = [c for c, group in data["center_to_cluster"].items()
                   if int(group) == int(k)]
        mask = np.isin(env, centers)
        cols = sorted(int(j) for j in selected)
        out[int(k)] = fit_whole_center_plr(
            np.asarray(data["X_treat"])[mask], np.asarray(data["Y"])[mask],
            np.asarray(data["F"])[mask][:, cols], env[mask], fold_seed,
        )
    return out


def assignment_margins(weights, centers) -> dict:
    """Decisiveness of the sEAM center assignment: margin = largest minus second-largest weight.

    weights has shape (n_centers, K); missing or malformed weights give NaN and -1.
    """
    nan = float("nan")
    if weights is None:
        return dict(seam_margin_min=nan, seam_margin_mean=nan,
                    seam_margin_min_center=-1, seam_n_centers_low_margin=-1)
    w = np.asarray(weights, dtype=float)
    ids = [int(c) for c in np.asarray(centers).ravel()]
    if w.ndim != 2 or w.shape[0] != len(ids) or w.shape[1] < 2 or not np.isfinite(w).all():
        return dict(seam_margin_min=nan, seam_margin_mean=nan,
                    seam_margin_min_center=-1, seam_n_centers_low_margin=-1)
    top2 = np.sort(w, axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]
    worst = int(np.argmin(margins))
    return dict(
        seam_margin_min=float(margins[worst]),
        seam_margin_mean=float(margins.mean()),
        seam_margin_min_center=ids[worst],
        seam_n_centers_low_margin=int((margins < SEAM_MARGIN_LOW).sum()),
    )
