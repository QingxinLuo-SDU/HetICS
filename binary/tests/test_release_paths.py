"""Release adaptations must preserve seeds and keep reference gates strict."""
import gzip
import csv
import json
from pathlib import Path

import pytest

import design_table as dt
import run_ablation as ra
import run_formal as rf
import run_pooled as rp
import frozen_rerun as fr
import formal_dgp


def test_round1_and_default_round2_keep_original_frozen_seeds():
    assert rf.formal_rows() == dt.load_rows()
    for stage, n_rows in (("r1", 40), ("r2", 59)):
        rows = rf.formal_rows(stage)
        assert len(rows) == n_rows
        assert dt.static_checks(rows)["status"] == "ok"
        by_point = {row["point_id"]: row for row in rows}
        labels = fr.load_labels([str(Path(fr.LABEL_DIR) / f"labels_{stage}.csv.gz")])
        points = fr.STAGES[stage]["points"]
        groups, _ = fr.build_groups(by_point, labels, points, [1, 500], [1])
        assert sum(len(group["point_ids"]) for group in groups) == 2 * len(points)


def test_optional_caches_require_explicit_recomputation(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "ABL_DIR", str(tmp_path))
    monkeypatch.setattr(rp, "POOLED_DIR", str(tmp_path))
    for load in (ra.load_cache, rp.load_sel_cache):
        with pytest.raises(FileNotFoundError, match="--gate"):
            load("r1")
        assert load("r1", allow_missing=True) == {}
    # Existing but corrupt inputs are never silently treated as missing.
    (tmp_path / "cache_r1.csv.gz").write_bytes(b"invalid gzip")
    with pytest.raises(gzip.BadGzipFile):
        ra.load_cache("r1", allow_missing=True)


def test_cli_precheck_accepts_no_cache_only_for_recomputation(tmp_path):
    for runner in (ra, rp):
        args = ["--stage", "r1", "--out-dir", str(tmp_path / runner.__name__),
                "--points", "P001", "--param-reps", "1", "--data-reps", "1", "--precheck"]
        assert runner.main(args + ["--allow-recompute"]) == 0
        with pytest.raises(FileNotFoundError, match="--gate"):
            runner.main(args + ["--allow-recompute", "--gate"])


def test_formal_cli_round_selection_and_configuration_validation(tmp_path):
    for stage in ("r1", "r2"):
        assert rf.main(["--stage", stage, "--static-only", "--out-dir", str(tmp_path / stage)]) == 0
        checks = json.loads((tmp_path / stage / "static_checks.json").read_text(encoding="utf-8"))
        assert checks["n_rows"] == (40 if stage == "r1" else 59)
    with pytest.raises(SystemExit):
        rf.main(["--stage", "r1", "--config", str(Path(dt.CONFIG_DIR) / "formal_r2.json"),
                 "--static-only", "--out-dir", str(tmp_path)])


@pytest.mark.parametrize("runner", [rf, ra, rp])
def test_base_generation_failure_has_nonzero_exit(runner, tmp_path, monkeypatch):
    from screening_bridge import load_screening
    module = load_screening(formal_dgp.STEP6_DIR)
    monkeypatch.setattr(module, "ALPHA_ICP", module.ALPHA_ICP)
    def fail(*args, **kwargs):
        raise RuntimeError("intentional base-generation failure")
    monkeypatch.setattr(formal_dgp, "generate_base", fail)
    out = tmp_path / runner.__name__
    args = ["--stage", "r1", "--out-dir", str(out), "--points", "P001",
            "--param-reps", "1", "--data-reps", "1", "--workers", "1"]
    if runner in (ra, rp):
        args.append("--allow-recompute")
    assert runner.main(args) == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_groups_failed"] == 1
    assert manifest["units_complete"] == 0


def test_gate_precheck_does_not_accept_partial_reference_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "ABL_DIR", str(tmp_path))
    monkeypatch.setattr(rp, "POOLED_DIR", str(tmp_path))
    for filename in ("cache_r1.csv.gz", "sel_cache_r1.csv.gz"):
        with gzip.open(tmp_path / filename, "wt", encoding="utf-8") as handle:
            handle.write("unit_key\nUNRELATED\n")
    for runner in (ra, rp):
        args = ["--stage", "r1", "--out-dir", str(tmp_path / runner.__name__),
                "--points", "P001", "--param-reps", "1", "--data-reps", "1",
                "--precheck", "--gate", "--allow-recompute"]
        assert runner.main(args) == 6


def test_archive_label_extractor_accepts_anonymous_worker_ids():
    import importlib.util
    path = Path(fr.LABEL_DIR).parents[1] / "tools/extract_frozen_labels.py"
    spec = importlib.util.spec_from_file_location("release_label_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    match = module.MEMBER_RE.search("runs/worker-1/units/P001/P001__p001__d01.json")
    assert match and match.group("machine") == "worker-1"


@pytest.mark.parametrize("stage,name", [(stage, name) for stage in ("r1", "r2")
                                       for name in ("scan_units", "merge_tables")])
def test_merge_cli_rejects_duplicate_or_path_like_shard_ids(stage, name, tmp_path):
    import importlib.util
    path = Path(fr.LABEL_DIR).parents[1] / "merge" / stage / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"release_{stage}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = ["--out", str(tmp_path)]
    if name == "scan_units":
        args += ["--tgz-dir", str(tmp_path), "--src", str(tmp_path)]
    else:
        args += ["--res-dir", str(tmp_path), "--scan", str(tmp_path), "--run-table", str(tmp_path)]
    for machines in (["../outside"], ["worker-1", "worker-1"]):
        with pytest.raises(SystemExit) as error:
            module.main(args + ["--machines"] + machines)
        assert error.value.code == 2
