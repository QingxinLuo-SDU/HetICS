import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.WholeCenterPLR
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
# Whole-center PLR inference bridge for CAVES-C

This module separates three layers. First, it proves the exact residual-score decomposition for the
Robinson residual-on-residual estimator. Second, it transports named center-level CLT, nuisance
remainder, and sandwich-consistency inputs to a studentized limit for each fixed latent cluster.
Third, it converts that limit to a reported interval while counting unavailable output as
noncoverage. An index `n` represents `M_k = n + 1` independent centers in the fixed latent cluster
`k`; it never represents the number of individual rows.

The center CLT, nuisance rates, causal identification conditions, and strict expectation-level
uniform-integrability condition are source assumptions. They are not inferred from cross-fitting
alone and are not replaced by a final normality assumption.
-/

open Filter MeasureTheory ProbabilityTheory
open scoped BigOperators Topology

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

/-! ## Population PLR identification coordinate -/

/-- Population PLR moment after the causal assumptions have supplied the oracle score and bread. -/
def populationPLRMoment
    (theta candidate bread oracleScore : ℝ) : ℝ :=
  oracleScore - (candidate - theta) * bread

/-- At the true cluster effect, the population moment equals the oracle orthogonality score. -/
theorem populationPLRMoment_at_true
    (theta bread oracleScore : ℝ) :
    populationPLRMoment theta theta bread oracleScore = oracleScore := by
  simp [populationPLRMoment]

/-- Oracle orthogonality and nonzero treatment-residual bread make `theta_k` the unique PLR root.
Consistency, exchangeability, positivity, and the partially linear causal model are the external
scientific premises that justify the supplied zero oracle score. -/
theorem populationPLRMoment_unique_root
    (theta candidate bread oracleScore : ℝ)
    (hBread : bread ≠ 0) (hOracle : oracleScore = 0)
    (hRoot : populationPLRMoment theta candidate bread oracleScore = 0) :
    candidate = theta := by
  simp only [populationPLRMoment, hOracle, zero_sub, neg_eq_zero] at hRoot
  exact sub_eq_zero.mp (mul_eq_zero.mp hRoot |>.resolve_right hBread)

section ExactPLRDecomposition

variable {Center : Type*} [Fintype Center]
variable {Unit : Center → Type*} [∀ c, Fintype (Unit c)]

/-- Cross-fitted treatment residual written as oracle residual minus nuisance error. -/
def estimatedTreatmentResidual (oracleV treatmentError : Sigma Unit → ℝ) : Sigma Unit → ℝ :=
  fun i => oracleV i - treatmentError i

/-- Cross-fitted outcome residual under the partially linear causal coefficient `θ`. -/
def estimatedOutcomeResidual (θ : ℝ) (oracleV oracleU outcomeError : Sigma Unit → ℝ) :
    Sigma Unit → ℝ :=
  fun i => θ * oracleV i + oracleU i - outcomeError i

/-- The nuisance combination appearing after subtracting `θ` times the fitted treatment residual. -/
def orthogonalNuisanceError (θ : ℝ) (treatmentError outcomeError : Sigma Unit → ℝ) :
    Sigma Unit → ℝ :=
  fun i => outcomeError i - θ * treatmentError i

/-- Exact difference between the fitted-residual score and the oracle center score. -/
def nuisanceScoreRemainder (θ : ℝ)
    (oracleV oracleU treatmentError outcomeError : Sigma Unit → ℝ) : ℝ :=
  pooledScore (estimatedTreatmentResidual oracleV treatmentError)
      (fun i => oracleU i - orthogonalNuisanceError θ treatmentError outcomeError i) -
    pooledScore oracleV oracleU

/-- The nuisance remainder exposes the two first-order cross terms and their product term. -/
theorem nuisanceScoreRemainder_eq
    (θ : ℝ) (oracleV oracleU treatmentError outcomeError : Sigma Unit → ℝ) :
    nuisanceScoreRemainder θ oracleV oracleU treatmentError outcomeError =
      -pooledScore oracleV (orthogonalNuisanceError θ treatmentError outcomeError) -
      pooledScore treatmentError oracleU +
      pooledScore treatmentError (orthogonalNuisanceError θ treatmentError outcomeError) := by
  classical
  simp only [nuisanceScoreRemainder, pooledScore, estimatedTreatmentResidual,
    orthogonalNuisanceError, mul_sub, sub_mul, Finset.sum_sub_distrib]
  ring

omit [Fintype Center] [(c : Center) → Fintype (Unit c)] in
/-- Subtracting `θ` times the fitted treatment residual leaves the oracle error minus the
orthogonal nuisance error. -/
theorem estimatedOutcomeResidual_sub_theta_treatment
    (θ : ℝ) (oracleV oracleU treatmentError outcomeError : Sigma Unit → ℝ) (i : Sigma Unit) :
    estimatedOutcomeResidual θ oracleV oracleU outcomeError i -
        θ * estimatedTreatmentResidual oracleV treatmentError i =
      oracleU i - orthogonalNuisanceError θ treatmentError outcomeError i := by
  simp only [estimatedOutcomeResidual, estimatedTreatmentResidual, orthogonalNuisanceError]
  ring

/-- Exact PLR score identity: oracle center score plus the explicit nuisance remainder. -/
theorem fitted_plr_score_decomposition
    (θ : ℝ) (oracleV oracleU treatmentError outcomeError : Sigma Unit → ℝ) :
    pooledScore (estimatedTreatmentResidual oracleV treatmentError)
        (fun i => estimatedOutcomeResidual θ oracleV oracleU outcomeError i -
          θ * estimatedTreatmentResidual oracleV treatmentError i) =
      pooledScore oracleV oracleU +
        nuisanceScoreRemainder θ oracleV oracleU treatmentError outcomeError := by
  classical
  have hResidual :
      (fun i => estimatedOutcomeResidual θ oracleV oracleU outcomeError i -
        θ * estimatedTreatmentResidual oracleV treatmentError i) =
      (fun i => oracleU i - orthogonalNuisanceError θ treatmentError outcomeError i) := by
    funext i
    exact estimatedOutcomeResidual_sub_theta_treatment
      θ oracleV oracleU treatmentError outcomeError i
  rw [hResidual]
  simp only [nuisanceScoreRemainder]
  ring

/-- The fitted PLR ratio has an exact oracle-linear plus nuisance-remainder representation. -/
theorem clusterEffectEstimate_error_decomposition
    (θ : ℝ) (oracleV oracleU treatmentError outcomeError : Sigma Unit → ℝ)
    (hVariation : pooledScore
      (estimatedTreatmentResidual oracleV treatmentError)
      (estimatedTreatmentResidual oracleV treatmentError) ≠ 0) :
    clusterEffectEstimate
        (estimatedTreatmentResidual oracleV treatmentError)
        (estimatedOutcomeResidual θ oracleV oracleU outcomeError) - θ =
      (pooledScore oracleV oracleU +
        nuisanceScoreRemainder θ oracleV oracleU treatmentError outcomeError) /
      pooledScore (estimatedTreatmentResidual oracleV treatmentError)
        (estimatedTreatmentResidual oracleV treatmentError) := by
  classical
  let vHat := estimatedTreatmentResidual oracleV treatmentError
  let yHat := estimatedOutcomeResidual θ oracleV oracleU outcomeError
  have hScore : pooledScore vHat (fun i => yHat i - θ * vHat i) =
      pooledScore oracleV oracleU +
        nuisanceScoreRemainder θ oracleV oracleU treatmentError outcomeError := by
    exact fitted_plr_score_decomposition θ oracleV oracleU treatmentError outcomeError
  have hNumerator : pooledScore vHat yHat - θ * pooledScore vHat vHat =
      pooledScore oracleV oracleU +
        nuisanceScoreRemainder θ oracleV oracleU treatmentError outcomeError := by
    rw [← hScore]
    simp only [pooledScore, Finset.mul_sum]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  simp only [clusterEffectEstimate]
  field_simp [vHat, hVariation]
  nlinarith

/-- Oracle linear term using a nonrandom limiting/reference bread. -/
def oraclePLRLinearTerm (oracleScore oracleBread : ℝ) : ℝ :=
  oracleScore / oracleBread

/-- Nuisance score remainder divided by the fitted PLR bread. -/
def nuisanceRatioRemainder (nuisanceScore fittedBread : ℝ) : ℝ :=
  nuisanceScore / fittedBread

/-- Exact correction for replacing fitted bread by the oracle/reference bread. -/
def denominatorRatioRemainder
    (oracleScore fittedBread oracleBread : ℝ) : ℝ :=
  oracleScore / fittedBread - oracleScore / oracleBread

/-- Numerical solve error is retained rather than silently treating an approximate solver as exact. -/
def numericalSolveRemainder (computed exactRatio : ℝ) : ℝ :=
  computed - exactRatio

/-- Exact algebraic linearization into oracle, nuisance-score, and denominator pieces. -/
theorem plr_ratio_linearization
    (oracleScore nuisanceScore fittedBread oracleBread : ℝ) :
    (oracleScore + nuisanceScore) / fittedBread =
      oraclePLRLinearTerm oracleScore oracleBread +
      nuisanceRatioRemainder nuisanceScore fittedBread +
      denominatorRatioRemainder oracleScore fittedBread oracleBread := by
  simp only [oraclePLRLinearTerm, nuisanceRatioRemainder, denominatorRatioRemainder]
  ring

/-- Complete finite error accounting for a numerically computed PLR coefficient. -/
theorem computedPLRError_decomposition
    (computed theta oracleScore nuisanceScore fittedBread oracleBread : ℝ)
    (exactRatio : ℝ)
    (hExactRatio : exactRatio - theta =
      (oracleScore + nuisanceScore) / fittedBread) :
    computed - theta =
      oraclePLRLinearTerm oracleScore oracleBread +
      nuisanceRatioRemainder nuisanceScore fittedBread +
      denominatorRatioRemainder oracleScore fittedBread oracleBread +
      numericalSolveRemainder computed exactRatio := by
  calc
    computed - theta = (exactRatio - theta) + (computed - exactRatio) := by ring
    _ = (oracleScore + nuisanceScore) / fittedBread +
        numericalSolveRemainder computed exactRatio := by rw [hExactRatio]; rfl
    _ = oraclePLRLinearTerm oracleScore oracleBread +
        nuisanceRatioRemainder nuisanceScore fittedBread +
        denominatorRatioRemainder oracleScore fittedBread oracleBread +
        numericalSolveRemainder computed exactRatio := by
      rw [plr_ratio_linearization]

end ExactPLRDecomposition

/-! ## Center-count asymptotics for one fixed latent cluster -/

/-- Average of `M_k=n+1` independent center contributions for one fixed latent cluster. -/
def centerLinearMean {Ω : Type*}
    (contribution : (n : ℕ) → Ω → Fin (n + 1) → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ((n + 1 : ℕ) : ℝ)⁻¹ * ∑ c, contribution n ω c

/-- Named source conditions for applying the bridge to every fixed latent cluster `k`. -/
structure ClusterPLRInferenceConditions where
  causalIdentification : Prop
  wholeCenterHonesty : Prop
  nuisanceProductRate : Prop
  denominatorNondegenerate : Prop
  centersPerClusterTendToInfinity : Prop
  centerLevelCLT : Prop
  sandwichConsistent : Prop
  outputMeasurable : Prop
  uniformIntegrabilityForExpectation : Prop

section ProbabilityBridges

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

/-- Equality on an event whose failure probability vanishes gives a negligible scaled difference.
The good event may depend on the full data; no independence is used. -/
theorem plr_scaled_difference_of_high_probability_eq
    (actual reference : ℕ → Ω → ℝ) (scale : ℕ → ℝ) (good : ℕ → Set Ω)
    (heq : ∀ n ω, ω ∈ good n → actual n ω = reference n ω)
    (hgood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => scale n * (actual n ω - reference n ω)) atTop 0 := by
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hgood
  · intro n
    exact bot_le
  · intro n
    apply measure_mono
    intro ω hω
    by_contra hnot
    have hg : ω ∈ good n := by simpa using hnot
    simp [heq n ω hg] at hω
    exact (not_le_of_gt hε) hω

private theorem plr_inMeasure_add_zero (X Y : ℕ → Ω → ℝ)
    (hX : TendstoInMeasure μ X atTop 0) (hY : TendstoInMeasure μ Y atTop 0) :
    TendstoInMeasure μ (fun n ω => X n ω + Y n ω) atTop 0 := by
  rw [tendstoInMeasure_iff_norm] at hX hY ⊢
  intro ε hε
  simp only [Pi.zero_apply, sub_zero] at hX hY ⊢
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by simpa using (hX (ε / 2) (half_pos hε)).add (hY (ε / 2) (half_pos hε)))
  · intro n
    exact bot_le
  · intro n
    refine (measure_mono (fun ω hω => ?_)).trans (measure_union_le _ _)
    change ε / 2 ≤ ‖X n ω‖ ∨ ε / 2 ≤ ‖Y n ω‖
    by_contra h
    push Not at h
    have hadd := norm_add_le (X n ω) (Y n ω)
    change ε ≤ ‖X n ω + Y n ω‖ at hω
    linarith [h.1, h.2]

/-- Point consistency follows from the center-linear term and both explicit remainders. -/
theorem thetaK_plr_consistent
    (estimate : ℕ → Ω → ℝ) (theta : ℕ → ℝ)
    (linear nuisance solve : ℕ → Ω → ℝ)
    (hexpansion : ∀ n ω, estimate n ω - theta n =
      linear n ω + nuisance n ω + solve n ω)
    (hLinear : TendstoInMeasure μ linear atTop 0)
    (hNuisance : TendstoInMeasure μ nuisance atTop 0)
    (hSolve : TendstoInMeasure μ solve atTop 0) :
    TendstoInMeasure μ (fun n ω => estimate n ω - theta n) atTop 0 := by
  have hsum := plr_inMeasure_add_zero μ
    (fun n ω => linear n ω + nuisance n ω) solve
    (plr_inMeasure_add_zero μ linear nuisance hLinear hNuisance) hSolve
  convert hsum using 1
  funext n ω
  rw [hexpansion n ω]

/-- A deterministic envelope on a high-probability event supplies an inspectable remainder-rate
condition, including nuisance-product and numerical-solve remainders. -/
theorem plr_inMeasure_of_good_envelope
    (X : ℕ → Ω → ℝ) (bound : ℕ → ℝ) (good : ℕ → Set Ω)
    (hBound : ∀ n ω, ω ∈ good n → |X n ω| ≤ bound n)
    (hBoundZero : Tendsto bound atTop (𝓝 0))
    (hGood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ X atTop 0 := by
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hGood
  · exact Filter.Eventually.of_forall fun n => bot_le
  · filter_upwards [hBoundZero.eventually (gt_mem_nhds hε)] with n hn
    apply measure_mono
    intro ω hω
    change ω ∉ good n
    intro hw
    have hb := hBound n ω hw
    simp only [Set.mem_setOf_eq, Pi.zero_apply, sub_zero, Real.norm_eq_abs] at hω
    linarith

/-- Consistency of the reference sandwich and a negligible fitted-rule difference imply
consistency of the actual sandwich. -/
theorem plr_variance_consistency_transfer
    (reference actual : ℕ → Ω → ℝ) (varianceLimit : ℝ)
    (hReference : TendstoInMeasure μ reference atTop (fun _ => varianceLimit))
    (hDifference : TendstoInMeasure μ
      (fun n ω => actual n ω - reference n ω) atTop 0) :
    TendstoInMeasure μ actual atTop (fun _ => varianceLimit) := by
  have hr : TendstoInMeasure μ
      (fun n ω => reference n ω - varianceLimit) atTop 0 := by
    simpa [tendstoInMeasure_iff_norm] using hReference
  have hsum := plr_inMeasure_add_zero μ _ _ hr hDifference
  have heq : (fun n ω => reference n ω - varianceLimit +
      (actual n ω - reference n ω)) =
      (fun n ω => actual n ω - varianceLimit) := by
    funext n ω
    ring
  rw [heq] at hsum
  simpa [tendstoInMeasure_iff_norm] using hsum

variable [IsProbabilityMeasure μ]

/-- The external center CLT is applied only to the explicit oracle center-linear term. -/
theorem plr_linear_plus_remainders_clt
    (linear nuisance solve : ℕ → Ω → ℝ)
    (hCLT : TendstoInDistribution linear atTop id (fun _ => μ) (gaussianReal 0 1))
    (hNuisance : TendstoInMeasure μ nuisance atTop 0)
    (hSolve : TendstoInMeasure μ solve atTop 0)
    (hmNuisance : ∀ n, AEMeasurable (nuisance n) μ)
    (hmSolve : ∀ n, AEMeasurable (solve n) μ) :
    TendstoInDistribution (fun n ω => linear n ω + nuisance n ω + solve n ω)
      atTop id (fun _ => μ) (gaussianReal 0 1) := by
  have h1 := hCLT.add_of_tendstoInMeasure_const (c := (0 : ℝ)) hNuisance hmNuisance
  have h2 := h1.add_of_tendstoInMeasure_const (c := (0 : ℝ)) hSolve hmSolve
  convert! h2 using 1
  simp
  rfl

/-- Positive center-level variance and sandwich consistency justify studentization. -/
theorem plr_studentize
    (X variance : ℕ → Ω → ℝ) (sigma : ℝ) (hSigma : 0 < sigma)
    (hX : TendstoInDistribution X atTop id (fun _ => μ) (gaussianReal 0 1))
    (hVariance : TendstoInMeasure μ variance atTop (fun _ => sigma ^ 2))
    (hmVariance : ∀ n, AEMeasurable (variance n) μ) :
    TendstoInDistribution (fun n ω => sigma * X n ω / Real.sqrt (variance n ω))
      atTop id (fun _ => μ) (gaussianReal 0 1) := by
  let g : ℝ × ℝ → ℝ := fun p =>
    sigma * p.1 / Real.sqrt (max p.2 (sigma ^ 2 / 2))
  have hPositive : 0 < sigma ^ 2 / 2 := by positivity
  have hContinuous : Continuous g := by
    apply Continuous.div (by fun_prop) (by fun_prop)
    intro p
    exact ne_of_gt (Real.sqrt_pos.2 (hPositive.trans_le (le_max_right _ _)))
  have hTruncated :=
    hX.continuous_comp_prodMk_of_tendstoInMeasure_const hContinuous hVariance hmVariance
  have hLimit : (fun ω : ℝ => g (id ω, sigma ^ 2)) = id := by
    funext ω
    simp [g, max_eq_left (by nlinarith : sigma ^ 2 / 2 ≤ sigma ^ 2),
      Real.sqrt_sq (le_of_lt hSigma), ne_of_gt hSigma]
  rw [hLimit] at hTruncated
  have hBad : Tendsto (fun n => μ {ω | ¬ sigma ^ 2 / 2 ≤ variance n ω})
      atTop (𝓝 0) := by
    have hv := (tendstoInMeasure_iff_norm.mp hVariance) (sigma ^ 2 / 2) hPositive
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hv
    · intro n
      exact bot_le
    · intro n
      apply measure_mono
      intro ω hω
      simp only [Set.mem_setOf_eq, Real.norm_eq_abs]
      have hlt : variance n ω < sigma ^ 2 / 2 := lt_of_not_ge hω
      rw [abs_sub_comm]
      exact (le_abs_self _).trans' (by linarith)
  have hEqual := plr_scaled_difference_of_high_probability_eq μ
    (fun n ω => sigma * X n ω / Real.sqrt (variance n ω))
    (fun n ω => g (X n ω, variance n ω))
    (fun _ => 1) (fun n => {ω | sigma ^ 2 / 2 ≤ variance n ω})
    (fun n ω hω => by
      change sigma ^ 2 / 2 ≤ variance n ω at hω
      simp [g, max_eq_left hω]) hBad
  apply tendstoInDistribution_of_tendstoInMeasure_sub _ id hTruncated
  · convert! hEqual using 1
    simp
    rfl
  · intro n
    exact (aemeasurable_const.mul (hX.forall_aemeasurable n)).div
      (Real.continuous_sqrt.measurable.comp_aemeasurable (hmVariance n))

/-- For one fixed latent cluster `k`, the exact PLR expansion, center CLT, nuisance/solve rates,
and center-sandwich consistency imply a standard-normal studentized statistic. -/
theorem thetaK_reference_studentized
    (estimate : ℕ → Ω → ℝ) (theta : ℕ → ℝ)
    (contribution : (n : ℕ) → Ω → Fin (n + 1) → ℝ)
    (nuisance solve varianceReference varianceActual : ℕ → Ω → ℝ)
    (sigma : ℝ) (hSigma : 0 < sigma)
    (hExpansion : ∀ n ω, estimate n ω - theta n =
      centerLinearMean contribution n ω + nuisance n ω + solve n ω)
    (hCenterCLT : TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
        centerLinearMean contribution n ω / sigma)
      atTop id (fun _ => μ) (gaussianReal 0 1))
    (hNuisance : TendstoInMeasure μ
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) * nuisance n ω / sigma) atTop 0)
    (hSolve : TendstoInMeasure μ
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) * solve n ω / sigma) atTop 0)
    (hmNuisance : ∀ n, AEMeasurable (nuisance n) μ)
    (hmSolve : ∀ n, AEMeasurable (solve n) μ)
    (hVarianceReference : TendstoInMeasure μ
      (fun n ω => (n + 1) * varianceReference n ω) atTop (fun _ => sigma ^ 2))
    (hVarianceDifference : TendstoInMeasure μ
      (fun n ω => (n + 1) * (varianceActual n ω - varianceReference n ω)) atTop 0)
    (hmVariance : ∀ n, AEMeasurable (varianceActual n) μ) :
    TendstoInDistribution
      (fun n ω => (estimate n ω - theta n) / Real.sqrt (varianceActual n ω))
      atTop id (fun _ => μ) (gaussianReal 0 1) := by
  have hVarianceDiff : TendstoInMeasure μ
      (fun n ω => (n + 1) * varianceActual n ω -
        (n + 1) * varianceReference n ω) atTop 0 := by
    simpa only [mul_sub] using hVarianceDifference
  have hVariance := plr_variance_consistency_transfer μ _ _ (sigma ^ 2)
    hVarianceReference hVarianceDiff
  have hNumerator := plr_linear_plus_remainders_clt μ _ _ _ hCenterCLT
    hNuisance hSolve
    (fun n => (aemeasurable_const.mul (hmNuisance n)).div_const sigma)
    (fun n => (aemeasurable_const.mul (hmSolve n)).div_const sigma)
  have hStudent := plr_studentize μ _ _ sigma hSigma hNumerator hVariance
    (fun n => aemeasurable_const.mul (hmVariance n))
  apply hStudent.congr
    (fun n => Filter.Eventually.of_forall fun ω => ?_) Filter.EventuallyEq.rfl
  simp only [hExpansion]
  have hn : (0 : ℝ) < n + 1 := by positivity
  rw [Real.sqrt_mul (le_of_lt hn)]
  have hSqrt : Real.sqrt (n + 1) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hn)
  calc
    _ = (Real.sqrt (n + 1) *
          (centerLinearMean contribution n ω + nuisance n ω + solve n ω)) /
        (Real.sqrt (n + 1) * Real.sqrt (varianceActual n ω)) := by
      congr 1
      field_simp [ne_of_gt hSigma]
    _ = _ := mul_div_mul_left _ _ hSqrt

/-- Portmanteau for the standard-normal two-sided interval. -/
theorem plr_studentized_interval_probability
    (statistic : ℕ → Ω → ℝ) (z : ℝ) (hz : 0 ≤ z)
    (hStatistic : TendstoInDistribution statistic atTop id (fun _ => μ)
      (gaussianReal 0 1)) :
    Tendsto (fun n => μ {ω | |statistic n ω| ≤ z}) atTop
      (𝓝 ((gaussianReal 0 1) (Set.Icc (-z) z))) := by
  haveI : NullSingletonClass (gaussianReal 0 1) :=
    nullSingletonClass_gaussianReal (by norm_num)
  have hBoundary : ((gaussianReal 0 1).map id) (frontier (Set.Icc (-z) z)) = 0 := by
    rw [Measure.map_id, frontier_Icc (by linarith)]
    exact ((Set.finite_singleton z).insert (-z)).measure_zero _
  have hPortmanteau :=
    ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto'
      hStatistic.tendsto hBoundary
  simp only [ProbabilityMeasure.coe_mk, Measure.map_id] at hPortmanteau
  convert hPortmanteau using 1
  funext n
  rw [Measure.map_apply_of_aemeasurable
    (hStatistic.forall_aemeasurable n) measurableSet_Icc]
  congr 1
  ext ω
  simp [abs_le]

/-- Algebraic equivalence between the Wald interval and the studentized event. -/
theorem plr_interval_iff_studentized
    (estimate theta standardError z : ℝ)
    (hStandardError : 0 < standardError) (_hz : 0 ≤ z) :
    (estimate - z * standardError ≤ theta ∧ theta ≤ estimate + z * standardError) ↔
      |(estimate - theta) / standardError| ≤ z := by
  rw [abs_le]
  constructor
  · rintro ⟨hLeft, hRight⟩
    constructor
    · apply (le_div_iff₀ hStandardError).2
      nlinarith
    · apply (div_le_iff₀ hStandardError).2
      nlinarith
  · rintro ⟨hLeft, hRight⟩
    have hLeft' := (le_div_iff₀ hStandardError).1 hLeft
    have hRight' := (div_le_iff₀ hStandardError).1 hRight
    constructor <;> nlinarith

/-- Reported cluster-specific coverage counts unavailable or nonpositive-standard-error output as
noncoverage instead of conditioning those repetitions away. -/
theorem thetaK_reported_interval_coverage
    (estimate standardError : ℕ → Ω → ℝ) (theta : ℕ → ℝ)
    (z : ℝ) (hz : 0 ≤ z)
    (available : ℕ → Set Ω) (hMeasurable : ∀ n, MeasurableSet (available n))
    (hPositive : ∀ n ω, ω ∈ available n → 0 < standardError n ω)
    (hAvailable : Tendsto (fun n => μ (available n)ᶜ) atTop (𝓝 0))
    (hStatistic : TendstoInDistribution
      (fun n ω => (estimate n ω - theta n) / standardError n ω)
      atTop id (fun _ => μ) (gaussianReal 0 1)) :
    Tendsto (fun n => μ.real {ω | ω ∈ available n ∧
      estimate n ω - z * standardError n ω ≤ theta n ∧
      theta n ≤ estimate n ω + z * standardError n ω})
      atTop (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) := by
  let event : ℕ → Set Ω := fun n =>
    {ω | |(estimate n ω - theta n) / standardError n ω| ≤ z}
  have hInterval := plr_studentized_interval_probability μ _ z hz hStatistic
  have hIntervalReal : Tendsto (fun n => μ.real (event n)) atTop
      (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) :=
    (ENNReal.continuousAt_toReal (measure_ne_top _ _)).tendsto.comp hInterval
  have hUnavailableReal : Tendsto (fun n => μ.real (available n)ᶜ) atTop (𝓝 0) := by
    simpa [measureReal_def, Function.comp_def] using!
      ENNReal.continuousAt_toReal ENNReal.zero_ne_top |>.tendsto.comp hAvailable
  have hEvent : ∀ n, {ω | ω ∈ available n ∧
      estimate n ω - z * standardError n ω ≤ theta n ∧
      theta n ≤ estimate n ω + z * standardError n ω} =
      event n ∩ available n := by
    intro n
    ext ω
    constructor
    · rintro ⟨ha, hci⟩
      exact ⟨(plr_interval_iff_studentized _ _ _ _
        (hPositive n ω ha) hz).1 hci, ha⟩
    · rintro ⟨ht, ha⟩
      exact ⟨ha, (plr_interval_iff_studentized _ _ _ _
        (hPositive n ω ha) hz).2 ht⟩
  simp_rw [hEvent]
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le
    (by simpa using hIntervalReal.sub hUnavailableReal) hIntervalReal
  · intro n
    have hSplit := measureReal_inter_add_sdiff (μ := μ) (s := event n) (hMeasurable n)
    have hLe : μ.real (event n \ available n) ≤ μ.real (available n)ᶜ :=
      measureReal_mono (Set.sdiff_subset_compl _ _)
    linarith
  · intro n
    exact measureReal_mono Set.inter_subset_left

end ProbabilityBridges

/-! ## Strict expectation-level asymptotic unbiasedness -/

/-- Once a uniform-integrability or sufficient-moment argument supplies a vanishing deterministic
bias envelope, expectation bias converges to zero. Distributional convergence alone is not used. -/
theorem thetaK_asymptotically_unbiased
    (expectedEstimate biasEnvelope : ℕ → ℝ) (theta : ℝ)
    (hBias : ∀ n, |expectedEstimate n - theta| ≤ biasEnvelope n)
    (hEnvelope : Tendsto biasEnvelope atTop (𝓝 0)) :
    Tendsto expectedEstimate atTop (𝓝 theta) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  obtain ⟨N, hN⟩ := (Metric.tendsto_atTop.mp hEnvelope) ε hε
  refine ⟨N, fun n hn => ?_⟩
  have hEnvelopeN : |biasEnvelope n| < ε := by
    simpa only [Real.dist_eq, sub_zero] using hN n hn
  rw [Real.dist_eq]
  exact lt_of_le_of_lt (hBias n) (lt_of_le_of_lt (le_abs_self _) hEnvelopeN)

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
