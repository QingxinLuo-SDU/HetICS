import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.RandomEffectsGLS
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.WholeCenterPLRInference
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Tactic

/-!
# Mean properties of covariance-aware random-effects GLS

This module proves the two mean layers authorized for CAVES-C. Fixed precision, or random precision
whose conditional cluster-effect means remain `mu`, gives exact unbiasedness. Same-data plug-in
precision is handled by an exact weight-perturbation identity and separate first-order or
expectation-bias transfer assumptions. No finite-sample exact unbiasedness is claimed for plug-in
REML weights, and no random-effects population interval is constructed here.
-/

open Filter MeasureTheory ProbabilityTheory
open scoped BigOperators Topology

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

/-- GLS column weight normalized by `1ᵀ P 1`. -/
def normalizedPrecisionWeight {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ) (k : Cluster) : ℝ :=
  precisionWeight precision k / totalPrecisionWeight precision

/-- Normalized precision weights sum to one whenever the GLS denominator is nonzero. -/
theorem sum_normalizedPrecisionWeight_eq_one
    {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ)
    (hTotal : totalPrecisionWeight precision ≠ 0) :
    ∑ k, normalizedPrecisionWeight precision k = 1 := by
  simp only [normalizedPrecisionWeight, div_eq_mul_inv, ← Finset.sum_mul,
    totalPrecisionWeight]
  exact mul_inv_cancel₀ hTotal

/-- The matrix GLS intercept is exactly the weighted sum using normalized column-sum weights. -/
theorem glsMean_eq_sum_normalizedPrecisionWeight
    {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ) (estimate : Cluster → ℝ)
    (_hTotal : totalPrecisionWeight precision ≠ 0) :
    glsMean precision estimate =
      ∑ k, normalizedPrecisionWeight precision k * estimate k := by
  simp only [glsMean, normalizedPrecisionWeight, div_eq_mul_inv, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- Centering a fixed-weight GLS mean centers each cluster estimate with the same weights. -/
theorem glsMean_sub_mean
    {Cluster : Type*} [Fintype Cluster]
    (precision : Matrix Cluster Cluster ℝ) (estimate : Cluster → ℝ) (mu : ℝ)
    (hTotal : totalPrecisionWeight precision ≠ 0) :
    glsMean precision estimate - mu =
      ∑ k, normalizedPrecisionWeight precision k * (estimate k - mu) := by
  rw [glsMean_eq_sum_normalizedPrecisionWeight precision estimate hTotal]
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul,
    sum_normalizedPrecisionWeight_eq_one precision hTotal, one_mul]

/-- Exact fixed-weight unbiasedness for any linear expectation functional. This applies to ordinary
expectation once integrability supplies its linear-map representation. -/
theorem fixed_precision_gls_unbiased
    {Cluster Ω : Type*} [Fintype Cluster]
    (expectation : (Ω → ℝ) →ₗ[ℝ] ℝ)
    (precision : Matrix Cluster Cluster ℝ)
    (estimate : Cluster → Ω → ℝ) (mu : ℝ)
    (hTotal : totalPrecisionWeight precision ≠ 0)
    (hClusterMean : ∀ k, expectation (estimate k) = mu) :
    expectation (fun ω => glsMean precision (fun k => estimate k ω)) = mu := by
  have hFunction :
      (fun ω => glsMean precision (fun k => estimate k ω)) =
        ∑ k, normalizedPrecisionWeight precision k • estimate k := by
    funext ω
    rw [glsMean_eq_sum_normalizedPrecisionWeight precision (fun k => estimate k ω) hTotal]
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  rw [hFunction, map_sum]
  simp only [LinearMap.map_smul, hClusterMean, smul_eq_mul, ← Finset.sum_mul,
    sum_normalizedPrecisionWeight_eq_one precision hTotal, one_mul]

/-- Conditional version for random but independent/exogenous weights: if the conditional mean of
every cluster estimate given the realized precision is `mu`, the conditional GLS mean is `mu`. -/
theorem conditional_precision_gls_unbiased
    {Cluster State : Type*} [Fintype Cluster]
    (precision : State → Matrix Cluster Cluster ℝ)
    (conditionalMean : State → Cluster → ℝ) (mu : ℝ)
    (hTotal : ∀ state, totalPrecisionWeight (precision state) ≠ 0)
    (hConditionalMean : ∀ state k, conditionalMean state k = mu) :
    ∀ state, glsMean (precision state) (conditionalMean state) = mu := by
  intro state
  exact glsMean_eq_of_common (precision state) (conditionalMean state) mu
    (hTotal state) (hConditionalMean state)

/-- Changing a valid precision matrix changes the GLS point estimate by the normalized-weight
perturbation applied to the same cluster-effect vector. -/
theorem glsMean_precision_perturbation
    {Cluster : Type*} [Fintype Cluster]
    (actualPrecision referencePrecision : Matrix Cluster Cluster ℝ)
    (estimate : Cluster → ℝ)
    (hActualTotal : totalPrecisionWeight actualPrecision ≠ 0)
    (hReferenceTotal : totalPrecisionWeight referencePrecision ≠ 0) :
    glsMean actualPrecision estimate - glsMean referencePrecision estimate =
      ∑ k, (normalizedPrecisionWeight actualPrecision k -
        normalizedPrecisionWeight referencePrecision k) * estimate k := by
  rw [glsMean_eq_sum_normalizedPrecisionWeight actualPrecision estimate hActualTotal,
    glsMean_eq_sum_normalizedPrecisionWeight referencePrecision estimate hReferenceTotal,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- Because both normalized weight vectors sum to one, only centered cluster effects matter in the
precision perturbation. -/
theorem glsMean_precision_perturbation_centered
    {Cluster : Type*} [Fintype Cluster]
    (actualPrecision referencePrecision : Matrix Cluster Cluster ℝ)
    (estimate : Cluster → ℝ) (mu : ℝ)
    (hActualTotal : totalPrecisionWeight actualPrecision ≠ 0)
    (hReferenceTotal : totalPrecisionWeight referencePrecision ≠ 0) :
    glsMean actualPrecision estimate - glsMean referencePrecision estimate =
      ∑ k, (normalizedPrecisionWeight actualPrecision k -
        normalizedPrecisionWeight referencePrecision k) * (estimate k - mu) := by
  rw [glsMean_precision_perturbation actualPrecision referencePrecision estimate
    hActualTotal hReferenceTotal]
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul]
  rw [sum_normalizedPrecisionWeight_eq_one actualPrecision hActualTotal,
    sum_normalizedPrecisionWeight_eq_one referencePrecision hReferenceTotal]
  ring

/-- An inspectable finite upper bound for same-data plug-in precision error. -/
theorem abs_glsMean_precision_perturbation_le
    {Cluster : Type*} [Fintype Cluster]
    (actualPrecision referencePrecision : Matrix Cluster Cluster ℝ)
    (estimate : Cluster → ℝ) (mu : ℝ)
    (hActualTotal : totalPrecisionWeight actualPrecision ≠ 0)
    (hReferenceTotal : totalPrecisionWeight referencePrecision ≠ 0) :
    |glsMean actualPrecision estimate - glsMean referencePrecision estimate| ≤
      ∑ k, |normalizedPrecisionWeight actualPrecision k -
        normalizedPrecisionWeight referencePrecision k| * |estimate k - mu| := by
  rw [glsMean_precision_perturbation_centered actualPrecision referencePrecision estimate mu
    hActualTotal hReferenceTotal]
  calc
    _ ≤ ∑ k, |(normalizedPrecisionWeight actualPrecision k -
        normalizedPrecisionWeight referencePrecision k) * (estimate k - mu)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = _ := by
      apply Finset.sum_congr rfl
      intro k hk
      rw [abs_mul]

/-- Expectation-level plug-in transfer. A vanishing bias envelope must come from moment or uniform
integrability control; finite-sample REML exact unbiasedness is not assumed. -/
theorem plugin_gls_asymptotically_unbiased
    (expectedPlugin expectedReference biasEnvelope : ℕ → ℝ) (mu : ℝ)
    (hReference : ∀ n, expectedReference n = mu)
    (hDifference : ∀ n,
      |expectedPlugin n - expectedReference n| ≤ biasEnvelope n)
    (hEnvelope : Tendsto biasEnvelope atTop (𝓝 0)) :
    Tendsto expectedPlugin atTop (𝓝 mu) := by
  apply thetaK_asymptotically_unbiased expectedPlugin biasEnvelope mu
  · intro n
    simpa [hReference n] using hDifference n
  · exact hEnvelope

section DistributionTransfer

variable {Ω : Type*} [MeasurableSpace Ω]
variable (probability : Measure Ω) [IsProbabilityMeasure probability]

/-- First-order same-data plug-in weight perturbation preserves a centered Gaussian limit. -/
theorem plugin_gls_centered_limit_transfer
    (plugin reference : ℕ → Ω → ℝ) (scale : ℕ → ℝ) (mu : ℝ)
    (hReference : TendstoInDistribution
      (fun n ω => scale n * (reference n ω - mu))
      atTop id (fun _ => probability) (gaussianReal 0 1))
    (hDifference : TendstoInMeasure probability
      (fun n ω => scale n * (plugin n ω - reference n ω)) atTop 0)
    (hmDifference : ∀ n, AEMeasurable
      (fun ω => scale n * (plugin n ω - reference n ω)) probability) :
    TendstoInDistribution (fun n ω => scale n * (plugin n ω - mu))
      atTop id (fun _ => probability) (gaussianReal 0 1) := by
  have hZero : TendstoInMeasure probability (fun _ : ℕ => fun _ : Ω => (0 : ℝ)) atTop 0 := by
    rw [tendstoInMeasure_iff_norm]
    intro ε hε
    simp [not_le.mpr hε]
  have hCombined := plr_linear_plus_remainders_clt probability
    (fun n ω => scale n * (reference n ω - mu))
    (fun n ω => scale n * (plugin n ω - reference n ω))
    (fun _ _ => 0) hReference hDifference hZero hmDifference
    (fun _ => aemeasurable_const)
  convert hCombined using 1
  funext n ω
  ring

end DistributionTransfer

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
