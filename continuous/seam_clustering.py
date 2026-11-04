"""
sEAM: Sparse EM-type Alternating Maximization for Mixture Regression
=====================================================================
Estimates latent cluster membership of centers by jointly fitting K sparse
regression coefficient vectors and soft cluster assignments.

Reference: Jing, Khalili & Xu (2025, JASA) -- sEAM for mixture regression.

Algorithm
---------
Stage 0  Initialization
  - F is standardized globally (zero mean, unit variance per feature).
  - Per-center Ridge regression  ->  beta_by_center  (n_centers, p)  [std scale]
  - K-means on beta_by_center    ->  initial hard assignments + beta_k^(0)

Stage 1  E-A-M loop  (until convergence or max_iter)
  E-step   soft cluster membership w_{ek}
             log w_{ek} proportional to log pi_k + sum log N(Y_i; F_i^T beta_k, sigma_k^2)
  A-step   weighted gradient S_{n,k} and Lipschitz constant u_k
             S_{n,k}(beta_k) = sum_e w_{ek} F_e^T (Y_e - F_e beta_k) / sigma_k^2
             u_k = lambda_max( sum_e w_{ek} F_e^T F_e / sigma_k^2 )
  M-step   IHT update of beta_k
             beta_k^(t+1) = H( beta_k^(t) + u_k^{-1} S_{n,k}(beta_k^(t)) ; kappa )
  Update   mixture weights pi_k proportional to sum_e w_{ek}
           cluster noise  sigma_k^2 = sum_e w_{ek} ||Y_e - F_e beta_k||^2 / sum_e w_{ek} n_e

Note on standardization
-----------------------
F is standardized at the entry of run_seam_clustering so that Ridge
initialization and the E-A-M main loop operate on the same feature scale.
Returned beta_by_cluster and beta_by_center are on the standardized scale.
Selected variable INDICES are invariant to this scaling.
"""

import warnings
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iht(v: np.ndarray, sparsity: int) -> np.ndarray:
    """Hard thresholding H(v; kappa): keep top kappa entries by magnitude, zero rest."""
    if sparsity >= len(v):
        return v.copy()
    threshold_idx = np.argpartition(np.abs(v), -sparsity)[-sparsity:]
    out = np.zeros_like(v)
    out[threshold_idx] = v[threshold_idx]
    return out


def _lipschitz(F_list: list, w_k: np.ndarray, sigma2: float,
               method: str = "exact", tol: float = 1e-3,
               random_state: int = 0) -> float:
    """
    Lipschitz constant = lambda_max(sum_e w_ek F_e^T F_e / sigma^2).

    method="exact"      : eigvalsh (symmetric PSD, O(p^3); use for p <= 1000)
    method="power_iter" : power iteration (O(p^2 * iters); use for p > 1000)
    method="auto"       : exact when p <= 1000, power_iter otherwise

    For power_iter the Gram matrix G is still formed explicitly; the gain over
    eigvalsh comes from avoiding the O(p^3) full eigendecomposition.
    tol   : relative convergence tolerance for power iteration (default 1e-3)
    """
    p = F_list[0].shape[1]
    G = np.zeros((p, p))
    for Fe, w in zip(F_list, w_k):
        if w > 1e-15:
            G += (w / sigma2) * (Fe.T @ Fe)

    use_exact = (method == "exact") or (method == "auto" and p <= 1000)

    if use_exact:
        eig_max = float(np.linalg.eigvalsh(G)[-1])
    else:
        # Power iteration for largest eigenvalue of PSD matrix G
        rng = np.random.default_rng(random_state)
        v   = rng.standard_normal(p)
        v  /= max(float(np.linalg.norm(v)), 1e-15)
        lam = 1.0
        for _ in range(300):
            Gv      = G @ v
            lam_new = float(v @ Gv)           # Rayleigh quotient
            norm_Gv = float(np.linalg.norm(Gv))
            if norm_Gv < 1e-15:
                break
            v = Gv / norm_Gv
            if abs(lam_new - lam) < tol * max(abs(lam), 1.0):
                lam = lam_new
                break
            lam = lam_new
        eig_max = lam

    return max(eig_max, 1e-8)


def _center_ll(F_list: list, Y_list: list,
               beta_k: np.ndarray, sigma2_k: float) -> np.ndarray:
    """Per-center Gaussian log-likelihood under cluster k parameters."""
    ll = np.empty(len(F_list))
    for i, (Fe, Ye) in enumerate(zip(F_list, Y_list)):
        resid = Ye - Fe @ beta_k
        n_e   = len(Ye)
        ll[i] = (-0.5 * np.dot(resid, resid) / sigma2_k
                 - 0.5 * n_e * np.log(2.0 * np.pi * sigma2_k))
    return ll


def _update_sigma2(F_list: list, Y_list: list,
                   beta_k: np.ndarray, w_k: np.ndarray) -> float:
    """Weighted residual variance for cluster k."""
    rss = 0.0; n_eff = 0.0
    for Fe, Ye, w in zip(F_list, Y_list, w_k):
        resid  = Ye - Fe @ beta_k
        rss   += w * float(np.dot(resid, resid))
        n_eff += w * len(Ye)
    return max(rss / max(n_eff, 1.0), 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Initialization: per-center Ridge + K-means  (expects pre-standardized F)
# ─────────────────────────────────────────────────────────────────────────────

def _init_ridge_kmeans(F: np.ndarray, Y: np.ndarray, env: np.ndarray,
                       K: int, sparsity: int,
                       random_state: int) -> tuple:
    """
    Per-center Ridge regression -> K-means clustering of coefficient vectors.
    F is assumed already standardized (done in run_seam_clustering).

    Returns
    -------
    beta_by_center : (n_centers, p)  per-center Ridge coefficients
    init_labels    : (n_centers,)    initial hard cluster assignments
    beta_k_init    : (K, p)          IHT of cluster-mean coefficients
    """
    centers   = np.unique(env)
    n_centers = len(centers)
    p         = F.shape[1]

    beta_by_center = np.zeros((n_centers, p))
    ridge          = Ridge(alpha=1.0, fit_intercept=False)

    # TODO B4: for n_e << p regimes (e.g., n_e=20, p=2000), replace Ridge
    # with a single gradient step from zero for faster cold-start proxy.
    for i, e in enumerate(centers):
        mask = env == e
        Fe, Ye = F[mask], Y[mask]
        if mask.sum() > 1:            # Ridge works for any n >= 2
            ridge.fit(Fe, Ye)
            beta_by_center[i] = ridge.coef_
        else:
            # Single sample edge case: use gradient proxy
            Yc = Ye - Ye.mean()
            beta_by_center[i] = Fe.T @ Yc / max(float((Fe ** 2).sum()), 1.0)

    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    km.fit(beta_by_center)
    init_labels = km.labels_

    beta_k_init = np.zeros((K, p))
    for k in range(K):
        members = np.where(init_labels == k)[0]
        if len(members) == 0:
            members = np.arange(n_centers)
        beta_k_init[k] = _iht(beta_by_center[members].mean(0), sparsity)

    return beta_by_center, init_labels, beta_k_init


# ─────────────────────────────────────────────────────────────────────────────
# Initialization: per-center |Spearman corr| signatures + K-means  (B1 fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _spearman_signatures(F: np.ndarray, Y: np.ndarray,
                          env: np.ndarray) -> np.ndarray:
    """
    Per-center |Spearman corr(F_j^(e), Y^(e))| signature matrix.
    Returns shape (n_centers, p).  Uses rank-based Pearson (= Spearman).
    Robust to monotone transforms; invariant to feature scaling.
    """
    centers = np.unique(env)
    p       = F.shape[1]
    sig     = np.zeros((len(centers), p))
    for i, e in enumerate(centers):
        mask  = env == e
        Fe    = F[mask]                                    # (n_e, p)
        Ye    = Y[mask]                                    # (n_e,)
        Fe_r  = np.apply_along_axis(rankdata, 0, Fe)      # ranked columns
        Ye_r  = rankdata(Ye)
        Fe_c  = Fe_r - Fe_r.mean(0)                       # centered ranks
        Ye_c  = Ye_r - Ye_r.mean()
        num   = Fe_c.T @ Ye_c                              # (p,)
        den   = np.sqrt((Fe_c ** 2).sum(0)
                        * float((Ye_c ** 2).sum()))
        r     = num / np.where(den > 1e-10, den, 1.0)
        sig[i] = np.abs(np.nan_to_num(r, nan=0.0))
    return sig


def _init_spearman_kmeans(F: np.ndarray, Y: np.ndarray, env: np.ndarray,
                           K: int, sparsity: int,
                           random_state: int) -> tuple:
    """
    K-means on per-center |Spearman corr| signatures for cluster discovery.
    Spearman signatures are used only for K-means; Ridge betas are kept for
    beta_by_center and beta_k_init so the E-A-M loop starts with valid
    regression directions.

    Returns same 3-tuple as _init_ridge_kmeans:
      (beta_by_center, init_labels, beta_k_init)
    """
    centers   = np.unique(env)
    n_centers = len(centers)
    p         = F.shape[1]

    # K-means on Spearman signatures
    sig         = _spearman_signatures(F, Y, env)           # (n_centers, p)
    km          = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    km.fit(sig)
    init_labels = km.labels_

    # Ridge betas for E-A-M initialisation (same as _init_ridge_kmeans)
    # TODO B4: replace Ridge with single gradient step when n_e << p
    ridge          = Ridge(alpha=1.0, fit_intercept=False)
    beta_by_center = np.zeros((n_centers, p))
    for i, e in enumerate(centers):
        mask = env == e
        Fe, Ye = F[mask], Y[mask]
        if mask.sum() > 1:
            ridge.fit(Fe, Ye)
            beta_by_center[i] = ridge.coef_
        else:
            Yc = Ye - Ye.mean()
            beta_by_center[i] = Fe.T @ Yc / max(float((Fe ** 2).sum()), 1.0)

    # Cluster-mean + IHT → initial beta_k
    beta_k_init = np.zeros((K, p))
    for k in range(K):
        members = np.where(init_labels == k)[0]
        if len(members) == 0:
            members = np.arange(n_centers)
        beta_k_init[k] = _iht(beta_by_center[members].mean(0), sparsity)

    return beta_by_center, init_labels, beta_k_init


# ─────────────────────────────────────────────────────────────────────────────
# Main: run_seam_clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_seam_clustering(
    F:                np.ndarray,
    Y:                np.ndarray,
    env:              np.ndarray,
    K:                int,
    sparsity:         int   = None,
    max_iter:         int   = 100,
    tol:              float = 1e-4,
    random_state:     int   = 0,
    init_method:      str   = "ridge",
    lipschitz_method: str   = "exact",
) -> dict:
    """
    sEAM clustering: full E-A-M iteration for mixture sparse regression.

    F is standardized internally so initialization and the E-A-M main loop
    share the same feature scale.

    Parameters
    ----------
    F, Y, env    : covariate (n,p), outcome (n,), center labels (n,)
    K            : number of latent clusters
    sparsity     : kappa (max nonzeros in each beta_k). Default: floor(n / log(n))
    max_iter     : maximum E-A-M iterations
    tol          : convergence: relative Frobenius change in beta_k matrix
    random_state : seed for K-means initialization
    init_method      : "ridge"         -- per-center Ridge + K-means (default)
                       "spearman_corr" -- per-center |Spearman corr| + K-means (B1 fallback)
                       "auto"          -- switching rule:
                         1. if any center n_e <= p  -> use spearman_corr immediately
                         2. else try ridge; if mean(max assignment prob) < 0.7
                            after one E-step -> fallback to spearman_corr
    lipschitz_method : "exact"      -- eigvalsh O(p^3); default, use for p <= 1000
                       "power_iter" -- power iteration O(p^2 * iters); use for p > 1000
                       "auto"       -- exact if p <= 1000, power_iter otherwise (B2)

    Returns
    -------
    dict with keys:
      center_cluster_hat  (n_centers,)   hard cluster assignment per center
      beta_by_cluster     (K, p)         converged sparse coefficient vectors
      beta_by_center      (n_centers, p) per-center Ridge coefficients (from init)
      weights             (n_centers, K) final soft membership matrix
      diagnostics         dict  n_iter, converged, log_likelihood, init_method_used

    Note: beta_by_cluster and beta_by_center are on standardized-F scale.
    Selected variable indices are scale-invariant.

    K is a user-specified hyperparameter.  In simulation experiments, set K to the
    ground-truth value from the DGP.  When K is unknown, use select_k_bic() (C1) or
    domain knowledge.  True cluster labels must never be used to set K in practice.
    """
    centers    = np.unique(env)
    n_centers  = len(centers)
    n_total, p = F.shape

    if sparsity is None:
        sparsity = max(int(n_total / np.log(max(n_total, 2))), 1)
    sparsity = int(min(sparsity, p))

    # ── Standardize F globally for numerical consistency ──────────────────────
    scaler = StandardScaler()
    Fs     = scaler.fit_transform(F)

    # Pre-split standardized data by center
    F_list = [Fs[env == e] for e in centers]
    Y_list = [Y[env == e]  for e in centers]

    # ── Stage 0: initialization (uses standardized F) ─────────────────────────
    if init_method == "spearman_corr":
        beta_by_center, init_labels, beta_k = _init_spearman_kmeans(
            Fs, Y, env, K, sparsity, random_state)
        init_method_used = "spearman_corr"

    elif init_method == "auto":
        # Rule 1: if any center has n_e <= p, use Spearman immediately
        if any((env == e).sum() <= p for e in centers):
            beta_by_center, init_labels, beta_k = _init_spearman_kmeans(
                Fs, Y, env, K, sparsity, random_state)
            init_method_used = "spearman_corr"
        else:
            # Try Ridge first, then assess cluster separation with one E-step
            beta_by_center, init_labels, beta_k = _init_ridge_kmeans(
                Fs, Y, env, K, sparsity, random_state)
            pi_k_tmp  = np.clip(
                np.bincount(init_labels, minlength=K).astype(float) / n_centers,
                1e-10, None)
            sigma2_tmp = max(float(np.var(Y)), 1e-6)
            log_u = np.zeros((n_centers, K))
            for k in range(K):
                log_u[:, k] = (np.log(pi_k_tmp[k])
                               + _center_ll(F_list, Y_list,
                                            beta_k[k], sigma2_tmp))
            log_Z_tmp = logsumexp(log_u, axis=1, keepdims=True)
            w_tmp     = np.exp(log_u - log_Z_tmp)
            w_tmp     = np.clip(w_tmp, 1e-300, None)
            w_tmp    /= w_tmp.sum(1, keepdims=True)
            # Rule 2: if mean max-assignment probability < 0.7, fallback
            if float(w_tmp.max(1).mean()) < 0.7:
                beta_by_center, init_labels, beta_k = _init_spearman_kmeans(
                    Fs, Y, env, K, sparsity, random_state)
                init_method_used = "spearman_corr"
            else:
                init_method_used = "ridge"

    else:  # "ridge" (default — preserves existing behaviour)
        beta_by_center, init_labels, beta_k = _init_ridge_kmeans(
            Fs, Y, env, K, sparsity, random_state)
        init_method_used = "ridge"

    pi_k     = np.bincount(init_labels, minlength=K).astype(float)
    pi_k     = np.clip(pi_k / pi_k.sum(), 1e-10, None)
    sigma2_k = np.full(K, max(float(np.var(Y)), 1e-6))

    # Placeholder so return always has 'w' even if max_iter=0
    w = np.full((n_centers, K), 1.0 / K)

    # ── Stage 1: E-A-M loop ───────────────────────────────────────────────────
    log_ll_trace: list = []
    converged         = False
    t                 = 0

    for t in range(max_iter):
        beta_k_old = beta_k.copy()

        # ── E-step ───────────────────────────────────────────────────────────
        log_unnorm = np.zeros((n_centers, K))
        for k in range(K):
            log_unnorm[:, k] = (np.log(max(pi_k[k], 1e-300))
                                + _center_ll(F_list, Y_list, beta_k[k], sigma2_k[k]))

        log_Z = logsumexp(log_unnorm, axis=1, keepdims=True)
        w     = np.exp(log_unnorm - log_Z)
        w     = np.clip(w, 1e-300, None)
        w    /= w.sum(axis=1, keepdims=True)
        log_ll_trace.append(float(log_Z.sum()))

        # ── A-step + M-step ───────────────────────────────────────────────────
        for k in range(K):
            w_k = w[:, k]

            # Weighted gradient of log-likelihood w.r.t. beta_k
            grad = np.zeros(p)
            for i, (Fe, Ye) in enumerate(zip(F_list, Y_list)):
                if w_k[i] > 1e-15:
                    resid  = Ye - Fe @ beta_k[k]
                    grad  += (w_k[i] / sigma2_k[k]) * (Fe.T @ resid)

            u_k         = _lipschitz(F_list, w_k, sigma2_k[k],
                                     method=lipschitz_method,
                                     random_state=random_state)
            beta_k[k]   = _iht(beta_k[k] + grad / u_k, sparsity)
            sigma2_k[k] = _update_sigma2(F_list, Y_list, beta_k[k], w_k)

        # ── Update mixture weights ────────────────────────────────────────────
        pi_k  = np.clip(w.sum(axis=0) / n_centers, 1e-10, None)
        pi_k /= pi_k.sum()

        # ── Convergence ───────────────────────────────────────────────────────
        delta    = np.linalg.norm(beta_k - beta_k_old, "fro")
        norm_old = np.linalg.norm(beta_k_old, "fro") + 1e-10
        if delta / norm_old < tol:
            converged = True
            break

    center_cluster_hat = w.argmax(axis=1)

    return {
        "center_cluster_hat": center_cluster_hat,
        "beta_by_cluster":    beta_k,
        "beta_by_center":     beta_by_center,
        "weights":            w,
        "diagnostics": {
            "n_iter":            t + 1,
            "converged":         converged,
            "log_likelihood":    log_ll_trace,
            "init_method_used":  init_method_used,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# C1: BIC-based K selection
# ─────────────────────────────────────────────────────────────────────────────

def select_k_bic(
    F:                np.ndarray,
    Y:                np.ndarray,
    env:              np.ndarray,
    k_max:            int,
    sparsity:         int,
    max_iter:         int   = 100,
    tol:              float = 1e-4,
    random_state:     int   = 0,
    init_method:      str   = "ridge",
    lipschitz_method: str   = "exact",
) -> dict:
    """
    BIC-based K selection for sEAM mixture sparse regression.

    Fits sEAM for K ∈ {1, ..., k_max} and returns the K minimising:

        BIC(K)  =  −2 · log-likelihood  +  (K · κ + K − 1) · log(E)

    where:
      log-likelihood  = sum_e log(sum_k π_k · N_e(β_k, σ_k²))  at convergence
                        (last element of diagnostics["log_likelihood"] trace)
      K · κ           = K sparse regression vectors × κ nonzeros each  (regression df)
      K − 1           = free mixture-weight parameters (π sums to 1)
      E               = number of observed centers  (BIC sample unit = center)

    κ is the `sparsity` argument passed to sEAM; it governs both the IHT
    constraint and the BIC penalty.  Sigma² parameters are not counted in
    the penalty because they are nuisance parameters that cancel across K.

    Parameters
    ----------
    F, Y, env    : covariate (n,p), outcome (n,), center labels (n,)
    k_max        : upper bound on K to search; clamped to n_centers if larger
    sparsity     : κ — max nonzeros per β_k (same value used for all K)
    max_iter     : maximum E-A-M iterations per K (passed to run_seam_clustering)
    tol          : convergence tolerance (passed to run_seam_clustering)
    random_state : RNG seed for K-means initialization
    init_method  : passed to run_seam_clustering ("ridge" | "spearman_corr" | "auto")

    Returns
    -------
    dict with keys:
      K_selected   : int          — K that minimises BIC
      bic_values   : {K: float}   — BIC(K) for each K in {1, ..., k_max_eff}
      loglik       : {K: float}   — observed log-likelihood at convergence
      seam_results : {K: dict}    — full run_seam_clustering output per K
    """
    n_centers  = len(np.unique(env))
    k_max_eff  = max(1, min(k_max, n_centers))
    E          = n_centers

    bic_values: dict   = {}
    loglik_map: dict   = {}
    seam_map:   dict   = {}

    for K in range(1, k_max_eff + 1):
        res       = run_seam_clustering(
            F, Y, env, K=K, sparsity=sparsity,
            max_iter=max_iter, tol=tol,
            random_state=random_state, init_method=init_method,
            lipschitz_method=lipschitz_method,
        )
        ll_trace  = res["diagnostics"]["log_likelihood"]
        # Last trace element = log-likelihood under parameters just before
        # convergence check (values are very close to final parameters).
        ll        = ll_trace[-1] if ll_trace else float("-inf")

        # BIC penalty: K*kappa regression params + (K-1) mixing params
        n_params  = K * sparsity + (K - 1)
        bic       = -2.0 * ll + n_params * np.log(max(E, 2))

        bic_values[K] = bic
        loglik_map[K] = ll
        seam_map[K]   = res

    K_selected = min(bic_values, key=bic_values.get)

    return {
        "K_selected":   K_selected,
        "bic_values":   bic_values,
        "loglik":       loglik_map,
        "seam_results": seam_map,
    }
