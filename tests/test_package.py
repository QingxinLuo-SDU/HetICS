"""The installed hetics package against the branch code: copied modules, and results on the same synthetic data.

Run from the repository root after `pip install .`:  python -m pytest tests -q
The branch code runs in subprocesses because the module names of the two branches overlap.
"""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

hetics = pytest.importorskip("hetics")

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
SKILL_BLOCKS = re.findall(r"```python\n(.*?)```", (ROOT / ".agents/skills/hetics/SKILL.md").read_text(encoding="utf-8"), flags=re.S)
HETICS_DIR = Path(hetics.__file__).parent


def _top_level(path):
    """Top-level functions and classes without docstrings and import statements."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    body = [b for b in sub.body if not isinstance(b, (ast.Import, ast.ImportFrom))]
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                            and isinstance(body[0].value.value, str):
                        body = body[1:]
                    sub.body = body or [ast.Pass()]
            out[node.name] = ast.dump(node)
    return out


@pytest.mark.parametrize("copied,branch", [
    ("_seam.py", "continuous/seam_clustering.py"),
    ("_chow.py", "continuous/chow_test.py"),
    ("_reml.py", "continuous/random_effects_gls.py"),
    ("_aipw.py", "binary/src/aipw_sandwich.py"),
])
def test_copied_modules_equal_the_branch_code(copied, branch):
    new, old = _top_level(HETICS_DIR / copied), _top_level(ROOT / branch)
    assert old, branch
    for name, code in old.items():
        assert new.get(name) == code, f"{copied}: {name} differs from {branch}"


def _run(script, tmp_path, **arrays):
    data = tmp_path / "data.npz"
    if arrays:
        np.savez(data, **arrays)
    path = tmp_path / "script.py"
    path.write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(path), str(ROOT), str(data), str(tmp_path / "out.json")], check=True,
                   env=ENV, cwd=tmp_path)
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def test_fit_ra_equals_the_branch_recipe(tmp_path):
    rng = np.random.default_rng(11)
    centers, n_c, p, K = 12, 300, 30, 2
    env = np.repeat(np.arange(centers), n_c)
    F = rng.normal(size=(centers * n_c, p))
    conf = F[:, :12].sum(axis=1) / 2
    X = conf + rng.normal(size=env.size)
    spec = np.where(env < centers // 2, F[:, 14:18].sum(axis=1), F[:, 18:22].sum(axis=1))
    Y = X + conf + 2 * spec + rng.normal(size=env.size)
    script = '''
import json, pathlib, sys
import numpy as np
repo_root, data, out = pathlib.Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
F, X, Y, env = data["F"], data["X"], data["Y"], data["env"]
K, sparsity, seed = 2, 12, 1
''' + SKILL_BLOCKS[0] + '''
json.dump({"selected": [[int(j) for j in c["selected"]] for c in clusters],
           "theta": [c["ra"]["theta"] for c in clusters], "se": [c["ra"]["se"] for c in clusters],
           "mu": float(overall.mu), "var_mu": float(overall.var_mu)}, open(out, "w"))
'''
    want = _run(script, tmp_path, F=F, X=X, Y=Y, env=env)
    got = hetics.fit_ra(F, X, Y, [f"hospital-{c:02d}" for c in env], K=2, sparsity=12, seed=1)
    assert [c["selected"] for c in got["clusters"]] == want["selected"]
    assert [c["theta"] for c in got["clusters"]] == want["theta"]
    assert [c["se"] for c in got["clusters"]] == want["se"]
    assert got["overall"]["mu"] == want["mu"] and got["overall"]["se"] ** 2 == pytest.approx(want["var_mu"], rel=1e-14)
    assert all(str(c).startswith("hospital-") for c in got["clusters"][0]["centers"])


def test_fit_aipw_equals_the_branch_recipe(tmp_path):
    script = '''
import json, pathlib, sys
import numpy as np
repo_root, data, out = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
root = repo_root / "binary"
for folder in ("src", "vendor", "vendor/step6_pkg"):
    sys.path.insert(0, str(root / folder))
import formal_dgp
gen, b6 = formal_dgp.load_sources()
params = dict(n_centers=9, n_per_center=2000, n_per_center_var=0.1, p_total=100, K=3, n_Z=8, n_causal_shared=10,
              n_causal_specific=3, n_spurious=10, rho_max=3.0, rho_min=1.0, delta_spec=0.0, nu_sd=0.4, alpha_ZX=1.0,
              alpha_ZY=1.0, sigma_Z=1.0, kappa=0.0, f_X="linear", f_Y="tanh", rho_block=0.6, rho_cross=0.3,
              beta_mag_lo=2.0, beta_mag_hi=3.0, signed_beta=True, gamma_XY=1.0, centers_per_cluster=None)
d = gen(**params, seed=2_000_000_456, param_seed=1_000_000_123)
b = b6.make_binary(d, 1.0, np.random.default_rng([2_000_000_456, 1]), np.random.default_rng([1_000_000_123, 2]))
F, X, Y, env = np.asarray(d["F"], float), b["X_bin"], b["Y_bin"], np.asarray(d["env"]).astype(int)
np.savez(data, F=F, X=X, Y=Y, env=env)
K, analysis_id = 3, "CUSTOM_DATA"
try:
''' + "".join(f"    {line}\n" for line in SKILL_BLOCKS[1].splitlines()) + '''
    json.dump({"center_map": {str(k): v for k, v in labels["center_map"].items()},
               "selected": {str(k): v for k, v in selected["selected_by_cluster"].items()},
               "estimate": effect["estimate"], "se": effect["se"]}, open(out, "w"))
except RuntimeError as error:
    if labels["center_map"] is not None:
        raise
    json.dump({"abstain": True}, open(out, "w"))
'''
    want = _run(script, tmp_path)
    data = np.load(tmp_path / "data.npz")
    if want.get("abstain"):          # the label algorithm may abstain under other package versions
        with pytest.raises(RuntimeError, match="eligible partition"):
            hetics.fit_aipw(data["F"], data["X"], data["Y"], data["env"], K=3, analysis_id="CUSTOM_DATA")
        return
    got = hetics.fit_aipw(data["F"], data["X"], data["Y"], data["env"], K=3, analysis_id="CUSTOM_DATA")
    assert {str(c["cluster"]): c["selected"] for c in got["clusters"]} == want["selected"]
    assert got["estimate"] == want["estimate"] and got["se"] == want["se"]
    assert got["estimand"] == "risk difference" and got["point_status"] == "complete"


def test_inputs_are_checked():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 5))
    X, Y, env = rng.normal(size=200), rng.normal(size=200), np.repeat(np.arange(8), 25)
    with pytest.raises(ValueError, match="finite"):
        hetics.fit_ra(np.where(F > 2, np.nan, F), X, Y, env, K=2)
    with pytest.raises(ValueError, match="0/1"):
        hetics.fit_aipw(F, X, Y, env, K=2)
    with pytest.raises(ValueError, match="constant"):
        hetics.fit_ra(np.column_stack([F, np.ones(200)]), X, Y, env, K=2)
    with pytest.raises(ValueError, match="duplicate"):
        hetics.fit_ra(np.column_stack([F, F[:, 0]]), X, Y, env, K=2)
    with pytest.raises(ValueError, match="2 K centers"):
        hetics.fit_ra(F, X, Y, env, K=5)
