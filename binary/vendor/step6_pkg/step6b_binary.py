"""
step6b_binary.py — binary treatment and outcome generated from the continuous latent-cluster DGP.

  y_baseline_signal = Y_cont − γ_cont·X_cont
      outcome signal without the direct treatment effect (outcome structure and noise; it is not itself a
      confounder; the confounders are idx_Z ∪ shared ∪ spec[k])
  spec_x_score      = linear combination of the own-cluster spec[k], standardized within cluster
  X_bin ~ Bern(σ(SIG_X·zx + SIG_X_SPEC·u_speck))          # zx = std(X_cont)
  Y_bin ~ Bern(σ(C0 + γ_logOR·X_bin + SIG_Y·z_y_baseline_signal))
  Truth: the realized global do-ATE on the risk-difference scale, mean[σ(η1) − σ(η0)], with the
         within-cluster ATE_k and the n_k/N-weighted check value.
  Random streams: binary draws use default_rng([ds, 1]) (passed by the caller); the spec-X
         coefficients use default_rng([ps, 2]) (parameter level).

pooled_logistic_lasso is the pooled L1-logistic selection of the pooled lasso comparator.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SIG_X, SIG_Y, C0 = 1.2, 2.0, 0.0     # exposure slope, outcome slope, outcome intercept
SIG_X_SPEC = 0.6                     # slope of the standardized own-cluster spec score in the exposure model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def draw_spec_x_coefs(data, rng_param):
    """Exposure-side coefficients of spec[k]: ±U(beta_mag_lo, beta_mag_hi), drawn like the signed outcome
    coefficients of the continuous DGP. The score is standardized within cluster, so the band only sets the
    relative weights of the spec covariates; SIG_X_SPEC sets the overall strength."""
    lo = float(data["params"]["beta_mag_lo"])
    hi = float(data["params"]["beta_mag_hi"])
    return {int(k): rng_param.choice([-1.0, 1.0], size=len(idx))
            * (lo + (hi - lo) * rng_param.random(len(idx)))
            for k, idx in data["true_cluster_specific_causal"].items()}


def spec_x_score(F, cluster_sample, idx_specific, coef_by_k):
    """Own-cluster spec score per observation: s_i = Σ_j coef[k(i)][j]·F[i, spec[k(i)][j]].
    The spec covariates of other clusters do not enter."""
    s = np.zeros(len(cluster_sample))
    for k, idx in idx_specific.items():
        msk = cluster_sample == int(k)
        if msk.any() and len(idx):
            s[msk] = F[np.ix_(np.where(msk)[0], list(idx))] @ coef_by_k[int(k)]
    return s


def make_binary(data, gamma_logor, rng_bin, rng_param):
    """Binary treatment and outcome.

    Arguments
      rng_bin   : binary draws, default_rng([ds, 1]) (independent of the continuous DGP's default_rng(ds))
      rng_param : spec-X coefficients, default_rng([ps, 2]) (parameter level)

    Returns a dict with
      X_bin, Y_bin          binary treatment and outcome
      ate_do                realized global do-ATE (risk difference)
      ate_k                 (K,) do-ATE within each true cluster
      n_k                   (K,) cluster sample sizes
      ate_do_nk             Σ n_k/N·ate_k (check value, equals ate_do)
      beta_spec_X           {k: drawn coefficients}
      beta_spec_X_eff       {k: effective logit-scale coefficients SIG_X_SPEC·coef/sd_k}
    """
    F = np.asarray(data["F"], float)
    Xc = np.asarray(data["X_treat"], float)
    Yc = np.asarray(data["Y"], float)
    cs = np.asarray(data["true_cluster_sample"], int)
    idx_spec = data["true_cluster_specific_causal"]
    K = len(idx_spec)

    # Outcome baseline signal: the outcome structure and noise without the direct treatment effect
    y_baseline_signal = Yc - float(data["gamma_XY"]) * Xc
    zx = (Xc - Xc.mean()) / max(Xc.std(), 1e-12)
    z_y_baseline_signal = ((y_baseline_signal - y_baseline_signal.mean())
                           / max(y_baseline_signal.std(), 1e-12))

    # Own-cluster spec score, standardized within cluster
    coef = draw_spec_x_coefs(data, rng_param)
    raw = spec_x_score(F, cs, idx_spec, coef)
    u = np.zeros_like(raw)
    beta_spec_X_eff = {}
    for k in range(K):
        msk = cs == k
        sd_k = max(float(raw[msk].std()), 1e-12) if msk.any() else 1.0
        if msk.any():
            u[msk] = (raw[msk] - float(raw[msk].mean())) / sd_k
        beta_spec_X_eff[k] = SIG_X_SPEC * coef[k] / sd_k

    eta_x = SIG_X * zx + SIG_X_SPEC * u
    Xb = (rng_bin.random(len(zx)) < _sigmoid(eta_x)).astype(int)

    eta1 = C0 + gamma_logor + SIG_Y * z_y_baseline_signal
    eta0 = C0 + SIG_Y * z_y_baseline_signal
    eta = np.where(Xb == 1, eta1, eta0)
    Yb = (rng_bin.random(len(eta)) < _sigmoid(eta)).astype(int)

    # Realized do-ATE, within-cluster ATE_k and the n_k-weighted check value
    diff = _sigmoid(eta1) - _sigmoid(eta0)
    ate_do = float(diff.mean())
    n_k = np.array([int((cs == k).sum()) for k in range(K)])
    ate_k = np.array([float(diff[cs == k].mean()) if n_k[k] else float("nan")
                      for k in range(K)])
    ate_do_nk = float(np.nansum(ate_k * n_k) / max(n_k.sum(), 1))
    return dict(X_bin=Xb, Y_bin=Yb, ate_do=ate_do, ate_k=ate_k, n_k=n_k,
                ate_do_nk=ate_do_nk, beta_spec_X=coef,
                beta_spec_X_eff=beta_spec_X_eff)


def pooled_logistic_lasso(F, Yb):
    """Pooled L1-penalized logistic selection on all observations.

    1-SE rule on the 3-fold neg_log_loss curve: among the C values whose mean score is at least the best
    mean minus one SE of the best, take the strongest penalty (smallest C). Returns the nonzero coefficients."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    Fs = StandardScaler().fit_transform(F)
    y = np.asarray(Yb, int)
    Cs = np.logspace(-3, 1, 8)
    m = LogisticRegressionCV(Cs=Cs, cv=3, penalty="l1", solver="saga",
                             scoring="neg_log_loss", tol=1e-3, max_iter=300,
                             n_jobs=1, random_state=42).fit(Fs, y)
    scores = m.scores_[1]                        # (n_folds, n_Cs), larger is better
    mean_sc = scores.mean(axis=0)
    best = int(np.argmax(mean_sc))
    se_best = float(scores[:, best].std()) / np.sqrt(scores.shape[0])
    cand = np.where(mean_sc >= mean_sc[best] - se_best)[0]
    C_1se = float(Cs[cand].min()) if len(cand) else float(m.C_[0])
    coef = LogisticRegression(C=C_1se, penalty="l1", solver="saga", tol=1e-3,
                              max_iter=300, random_state=42).fit(Fs, y).coef_[0]
    return set(np.where(coef != 0)[0].tolist())
