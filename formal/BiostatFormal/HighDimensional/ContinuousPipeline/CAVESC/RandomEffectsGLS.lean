import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic

/-!
# Covariance-aware random-effects GLS finite algebra for CAVES-C

This module begins after a caller has supplied a nonnegative REML heterogeneity component and a
valid inverse of the resulting working covariance. It proves only finite matrix and weighted-mean
algebra. Symmetry and positive definiteness, covariance consistency, validity of the REML
criterion, and interval calibration remain external source conditions; the inverse equalities in
`CovariancePrecisionCertificate` are algebraic certificates, not estimator-validity theorems.

The random-effects mean is a fixed intercept parameter of the latent-cluster effect distribution.
It is not a fixed-effect meta-analysis estimand and is not automatically a target-population
cluster-prevalence-weighted individual average treatment effect.

Current mutually exclusive latent-cluster outputs naturally supply only diagonal within-cluster
variance. Nonzero off-diagonal dependence induced by common upstream sEAM or selection requires an
additional joint covariance procedure outside this task; a diagonal matrix is not relabelled as a
full covariance estimate here.
-/

open scoped BigOperators

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

/-- Sampling covariance augmented by a common nonnegative heterogeneity component. -/
def workingCovariance {Cluster : Type*} [DecidableEq Cluster]
    (samplingCov : Matrix Cluster Cluster ℝ)
    (tau2 : ℝ) : Matrix Cluster Cluster ℝ :=
  samplingCov + Matrix.diagonal (fun _ => tau2)

/--
A finite algebra certificate for a caller-supplied random-effects working covariance and its
two-sided inverse. The fields do not establish how REML estimated `tau2Hat`, whether the covariance
is positive definite or consistent, or whether resulting intervals are calibrated.
-/
structure CovariancePrecisionCertificate (Cluster : Type*)
    [Fintype Cluster] [DecidableEq Cluster] where
  samplingCov : Matrix Cluster Cluster ℝ
  tau2Hat : ℝ
  precision : Matrix Cluster Cluster ℝ
  tau2_nonneg : 0 ≤ tau2Hat
  left_inverse : workingCovariance samplingCov tau2Hat * precision = 1
  right_inverse : precision * workingCovariance samplingCov tau2Hat = 1

/-- With zero heterogeneity variance, the working covariance is the sampling covariance. -/
theorem workingCovariance_zero {Cluster : Type*} [DecidableEq Cluster]
    (samplingCov : Matrix Cluster Cluster ℝ) :
    workingCovariance samplingCov 0 = samplingCov := by
  simp [workingCovariance]

/-- The GLS weight attached to cluster `j`, obtained by summing its precision-matrix column. -/
def precisionWeight {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ) (j : Cluster) : ℝ :=
  ∑ i, precision i j

/-- The total precision weight `1ᵀ P 1`. -/
def totalPrecisionWeight {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ) : ℝ :=
  ∑ j, precisionWeight precision j

/--
The finite GLS intercept for supplied cluster-effect estimates and precision matrix. Statistical
validity of either input is not proved by this definition.
-/
def glsMean {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ)
    (estimate : Cluster → ℝ) : ℝ :=
  (∑ j, precisionWeight precision j * estimate j) /
    totalPrecisionWeight precision

/-- GLS mean using the precision from a caller-supplied random-effects certificate. -/
def remlGlsMean {Cluster : Type*} [Fintype Cluster] [DecidableEq Cluster]
    (certificate : CovariancePrecisionCertificate Cluster)
    (estimate : Cluster → ℝ) : ℝ :=
  glsMean certificate.precision estimate

/-- Common-effect FGLS boundary or sensitivity analysis using a sampling precision matrix. -/
def fglsMean {Cluster : Type*} [Fintype Cluster]
    (samplingPrecision : Matrix Cluster Cluster ℝ)
    (estimate : Cluster → ℝ) : ℝ :=
  glsMean samplingPrecision estimate

/-- A nondegenerate GLS intercept exactly recovers an effect shared by every cluster. -/
theorem glsMean_eq_of_common
    {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ) (estimate : Cluster → ℝ) (mu : ℝ)
    (hTotal : totalPrecisionWeight precision ≠ 0)
    (hCommon : ∀ k, estimate k = mu) :
    glsMean precision estimate = mu := by
  have hNumerator :
      (∑ j, precisionWeight precision j * estimate j) =
        totalPrecisionWeight precision * mu := by
    calc
      (∑ j, precisionWeight precision j * estimate j) =
          ∑ j, precisionWeight precision j * mu := by
        apply Finset.sum_congr rfl
        intro j hj
        rw [hCommon j]
      _ = totalPrecisionWeight precision * mu := by
        rw [totalPrecisionWeight, Finset.sum_mul]
  rw [glsMean, hNumerator]
  exact mul_div_cancel_left₀ mu hTotal

/-- The common-effect recovery theorem specialized to a random-effects precision certificate. -/
theorem remlGlsMean_eq_of_common
    {Cluster : Type*} [Fintype Cluster] [DecidableEq Cluster]
    (certificate : CovariancePrecisionCertificate Cluster) (estimate : Cluster → ℝ) (mu : ℝ)
    (hTotal : totalPrecisionWeight certificate.precision ≠ 0)
    (hCommon : ∀ k, estimate k = mu) :
    remlGlsMean certificate estimate = mu := by
  exact glsMean_eq_of_common certificate.precision estimate mu hTotal hCommon

/-- Equal precision matrices give identical random-effects GLS and common-effect FGLS means. -/
theorem remlGlsMean_eq_fgls_of_same_precision
    {Cluster : Type*} [Fintype Cluster] [DecidableEq Cluster]
    (certificate : CovariancePrecisionCertificate Cluster)
    (samplingPrecision : Matrix Cluster Cluster ℝ) (estimate : Cluster → ℝ)
    (hPrecision : certificate.precision = samplingPrecision) :
    remlGlsMean certificate estimate = fglsMean samplingPrecision estimate := by
  simp [remlGlsMean, fglsMean, hPrecision]

/--
Named caller-supplied assumptions at the random-effects source-theorem boundary. No field is
automatically discharged here, including when only three latent clusters are available.
-/
structure RandomEffectsSourceConditions where
  covarianceSymmetricPositiveDefinite : Prop
  covarianceConsistent : Prop
  remlCriterionValid : Prop
  intervalCalibrated : Prop
  finiteClusterUseJustified : Prop

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
