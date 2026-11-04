"""Reference PLR and sEAM assignment margins."""
import numpy as np

from plr_transfer import assignment_margins, fit_reference_plr
from v4_types import ClusterPLRResult


def fake_plr(theta: float, variance: float) -> ClusterPLRResult:
    se = float(np.sqrt(variance))
    return ClusterPLRResult(
        theta=theta, se=se, variance=variance,
        ci_z=(theta - 1.96 * se, theta + 1.96 * se),
        ci_t=(theta - 2.0 * se, theta + 2.0 * se),
        center_scores={}, fold_by_center={}, status="ok", error_code=None,
    )


def test_reference_plr_uses_true_labels_and_core_sets(monkeypatch):
    """fit_reference_plr takes the centers of each true cluster and the columns of A_core[k]."""
    seen = []

    def spy(x, y, w, env, fold_seed):
        seen.append(dict(n=len(x), cols=w.shape[1],
                         centers=sorted(set(int(e) for e in env)),
                         fold_seed=fold_seed))
        return fake_plr(1.0, 0.04)

    import plr_transfer
    monkeypatch.setattr(plr_transfer, "fit_whole_center_plr", spy)

    n = 12
    data = {
        "env": np.repeat([0, 1, 2, 3], 3),
        "X_treat": np.arange(n, dtype=float),
        "Y": np.arange(n, dtype=float),
        "F": np.arange(n * 5, dtype=float).reshape(n, 5),
        "center_to_cluster": {0: 0, 1: 0, 2: 1, 3: 1},
        "A_core_by_cluster": {0: {0, 2}, 1: {1, 3, 4}},
    }
    out = fit_reference_plr(data, fold_seed=13)
    assert set(out) == {0, 1}
    assert seen[0] == dict(n=6, cols=2, centers=[0, 1], fold_seed=13)
    assert seen[1] == dict(n=6, cols=3, centers=[2, 3], fold_seed=13)


def test_assignment_margins_report_the_least_decisive_center():
    out = assignment_margins(np.array([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]]), [10, 11])
    assert np.isclose(out["seam_margin_min"], 0.05)
    assert np.isclose(out["seam_margin_mean"], (0.7 + 0.05) / 2)
    assert out["seam_margin_min_center"] == 11
    assert out["seam_n_centers_low_margin"] == 1


def test_assignment_margins_handle_missing_weights():
    out = assignment_margins(None, [0, 1, 2])
    assert np.isnan(out["seam_margin_min"]) and out["seam_margin_min_center"] == -1
