"""Calibration design axes and seeds (static checks; no data generated). Requires SCENA_BETA_ANCHOR=2.0,3.0."""
import importlib

import pytest

import step6x_config as cfg

pytestmark = pytest.mark.skipif(cfg.BETA_TAG != "b2-3", reason="requires SCENA_BETA_ANCHOR=2.0,3.0")

import hypothesis_calibration_config as hcc       # noqa: E402
from hypothesis_calibration_seeds import calibration_seeds  # noqa: E402

# Reference seeds of the centers axis at (param_index, data_index) = (0, 0), (3, 17), (9, 49).
M_AXIS_SEEDS = {
    "M03_G0": [[1615859437, 605779182, 511067822, 168314782], [2975036001, 3862234177, 3524367207, 3051423837],
               [3774572452, 1884343587, 767911940, 3206627966]],
    "M04_G0035": [[3995490459, 2890841606, 2982170551, 1175220261], [1467413980, 3228611299, 2684550143, 2306635515],
                  [3571177004, 1698372922, 1795396244, 630371227]],
    "M12_G0035": [[3976982006, 1938105365, 1493766171, 47923147], [2117779691, 1578874167, 1493245811, 3742569356],
                  [648300043, 1413258915, 2090633409, 4131635456]],
}


def _bundle(point, p, d):
    s = calibration_seeds(point, p, d)
    return [s.param_seed, s.data_seed, s.fold_seed, s.algorithm_seed]


def test_center_axis_seeds_match_the_reference():
    pts = {p["point_id"]: p for p in hcc.build_calibration_points()}
    assert len(pts) == 14
    for pid, want in M_AXIS_SEEDS.items():
        assert [_bundle(pts[pid], p, d) for p, d in [(0, 0), (3, 17), (9, 49)]] == want, pid


def test_center_axis_points_carry_no_axis_tag():
    assert all("cal_axis" not in p for p in hcc.build_calibration_points())


def test_n_axis_design():
    pts = hcc.build_n_axis_points()
    assert len(pts) == 12
    assert sorted({(p["dgp_params"]["n_per_center"], p["gamma_true"]) for p in pts}) == \
        sorted((n, g) for n in (5000, 6000, 7000, 8000, 9000, 12000) for g in (0.0, 0.035))
    m04 = next(p for p in hcc.build_calibration_points() if p["point_id"] == "M04_G0")
    for p in pts:
        assert p["cal_axis"] == "n" and p["centers_per_cluster"] == 4 and p["n_centers"] == 12
        changed = {k for k, v in p["dgp_params"].items() if v != m04["dgp_params"][k]}
        assert changed <= {"n_per_center", "gamma_XY"}
    m_ids = {p["point_id"] for p in hcc.build_calibration_points()}
    assert not m_ids & {p["point_id"] for p in pts}


def test_n_axis_seeds_independent_across_levels_shared_across_gamma():
    pts = {p["point_id"]: p for p in hcc.build_n_axis_points()}
    m04 = next(p for p in hcc.build_calibration_points() if p["point_id"] == "M04_G0")
    by_level = {}
    for pid, p in pts.items():
        by_level.setdefault(p["dgp_params"]["n_per_center"], []).append(_bundle(p, 2, 5))
    for n, bundles in by_level.items():
        assert bundles[0] == bundles[1], f"n={n}: gamma = 0 and gamma = 0.035 should share seeds"
    firsts = [tuple(b[0]) for b in by_level.values()] + [tuple(_bundle(m04, 2, 5))]
    assert len(set(firsts)) == len(firsts), "seeds must differ across levels and from M04"


@pytest.mark.parametrize("axis,n_points", [("m", 14), ("n", 12), ("p", 7)])
def test_runner_picks_axis_from_env(monkeypatch, axis, n_points):
    monkeypatch.setenv("SCENA_CAL_AXIS", axis)
    import run_cal_ra
    run_cal_ra = importlib.reload(run_cal_ra)
    try:
        assert len(run_cal_ra._points()) == n_points
        assert len(run_cal_ra.all_tasks()) == n_points * 1000
    finally:
        monkeypatch.delenv("SCENA_CAL_AXIS")
        importlib.reload(run_cal_ra)


def test_bucket_ranges_cover_every_task_once(monkeypatch):
    monkeypatch.setenv("SCENA_CAL_AXIS", "n")
    import run_cal_ra
    run_cal_ra = importlib.reload(run_cal_ra)
    try:
        a, b = set(run_cal_ra.all_tasks(0, 600)), set(run_cal_ra.all_tasks(600, 1000))
        assert not a & b and len(a | b) == 12000
    finally:
        monkeypatch.delenv("SCENA_CAL_AXIS")
        importlib.reload(run_cal_ra)


def test_data_ranges_partition_the_data_seeds(monkeypatch):
    import run_cal_ra
    halves = []
    try:
        for rng in ("0-50", "50-100"):
            monkeypatch.setenv("SCENA_CAL_DATA", rng)
            halves.append(set(importlib.reload(run_cal_ra).all_tasks()))
        monkeypatch.delenv("SCENA_CAL_DATA")
        full = set(importlib.reload(run_cal_ra).all_tasks())
        first, second = halves
        assert len(first) == len(second) == 7000 and not first & second and first | second == full
        assert {d for _, _, d in first} == set(range(50)) and {d for _, _, d in second} == set(range(50, 100))
        monkeypatch.setenv("SCENA_CAL_DATA", "50-101")
        with pytest.raises(SystemExit):
            importlib.reload(run_cal_ra).all_tasks()
    finally:
        monkeypatch.delenv("SCENA_CAL_DATA", raising=False)
        importlib.reload(run_cal_ra)


# Reference seeds of the sample-size axis: N05000_G0 / N12000_G0 at (0, 0) and (9, 99).
N_AXIS_SEEDS = {
    "N05000_G0": [[4037353492, 2952052835, 3623727716, 1213701632], [2541639495, 367340385, 251298164, 3543373637]],
    "N12000_G0": [[4055382640, 2167736321, 2650796033, 2740460358], [82473623, 2315121111, 2864741468, 4254027571]],
}


def test_n_axis_seeds_match_the_reference():
    pts = {p["point_id"]: p for p in hcc.build_n_axis_points()}
    for pid, want in N_AXIS_SEEDS.items():
        assert [_bundle(pts[pid], p, d) for p, d in [(0, 0), (9, 99)]] == want, pid


def test_power_axis_design_and_seed_pairing():
    """7 sample sizes x gamma = 0.055; seeds equal those of the gamma = 0 point of the same sample size
    (M04_G0 at 10,000) and differ across sample sizes."""
    pts = {p["point_id"]: p for p in hcc.build_p_axis_points()}
    assert sorted(p["dgp_params"]["n_per_center"] for p in pts.values()) == [5000, 6000, 7000, 8000, 9000, 10000, 12000]
    assert all(p["gamma_true"] == p["dgp_params"]["gamma_XY"] == 0.055 and p["cal_axis"] == "p" for p in pts.values())
    ref = {p["dgp_params"]["n_per_center"]: p for p in hcc.build_n_axis_points() if p["gamma_true"] == 0.0}
    ref[10000] = next(p for p in hcc.build_calibration_points() if p["point_id"] == "M04_G0")
    other_ids = {p["point_id"] for p in hcc.build_calibration_points() + hcc.build_n_axis_points()}
    assert not other_ids & set(pts)
    firsts = []
    for p in pts.values():
        n = p["dgp_params"]["n_per_center"]
        for pd in [(0, 0), (4, 63), (9, 99)]:
            assert _bundle(p, *pd) == _bundle(ref[n], *pd), p["point_id"]
        firsts.append(_bundle(p, 0, 0)[0])
    assert len(set(firsts)) == 7


def test_power_axis_bucket_ranges_cover_every_task_once(monkeypatch):
    monkeypatch.setenv("SCENA_CAL_AXIS", "p")
    import run_cal_ra
    run_cal_ra = importlib.reload(run_cal_ra)
    try:
        parts = [set(run_cal_ra.all_tasks(lo, hi)) for lo, hi in ((0, 240), (240, 450), (450, 760), (760, 1000))]
        assert sum(map(len, parts)) == len(set().union(*parts)) == 7000
    finally:
        monkeypatch.delenv("SCENA_CAL_AXIS")
        importlib.reload(run_cal_ra)
