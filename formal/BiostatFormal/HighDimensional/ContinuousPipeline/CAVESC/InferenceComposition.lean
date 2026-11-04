import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.LearnedPipelineTransfer
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.RandomEffectsGLSInference
import Mathlib.Tactic

/-!
# Final cluster-specific inference composition for CAVES-C

This module consumes, rather than assumes away, the two preceding bridges. It transports the
reference PLR limit through a learned-rule point/variance perturbation and then forms the reported
interval for each fixed latent cluster. Its six-event ledger keeps label, Y-side, strict-stability,
adjustment/identification, usability/nondegeneracy, and reference-inference failures in the
unconditional denominator. No independence among stages is used.
-/

open Filter MeasureTheory ProbabilityTheory
open scoped Topology

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

section StudentizedTransfer

variable {Ω : Type*} [MeasurableSpace Ω]
variable (probability : Measure Ω) [IsProbabilityMeasure probability]

/-- A reference center-CLT plus the learned-rule point perturbation gives the actual numerator
limit; reference-sandwich consistency plus the variance perturbation gives the actual denominator
limit. Thus the actual learned PLR is studentized without pretending the random rule was fixed. -/
theorem thetaK_actual_studentized_from_reference
    (actual reference : ℕ → Ω → ℝ) (theta : ℕ → ℝ)
    (varianceActual varianceReference : ℕ → Ω → ℝ)
    (sigma : ℝ) (hSigma : 0 < sigma)
    (hReference : TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
        (reference n ω - theta n) / sigma)
      atTop id (fun _ => probability) (gaussianReal 0 1))
    (hPointDifference : TendstoInMeasure probability
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
        (actual n ω - reference n ω) / sigma) atTop 0)
    (hmPointDifference : ∀ (n : ℕ), AEMeasurable
      (fun ω => Real.sqrt ((n : ℝ) + 1) *
        (actual n ω - reference n ω) / sigma) probability)
    (hVarianceReference : TendstoInMeasure probability
      (fun n ω => (n + 1) * varianceReference n ω) atTop (fun _ => sigma ^ 2))
    (hVarianceDifference : TendstoInMeasure probability
      (fun n ω => (n + 1) *
        (varianceActual n ω - varianceReference n ω)) atTop 0)
    (hmVarianceActual : ∀ n, AEMeasurable (varianceActual n) probability) :
    TendstoInDistribution
      (fun n ω => (actual n ω - theta n) / Real.sqrt (varianceActual n ω))
      atTop id (fun _ => probability) (gaussianReal 0 1) := by
  have hZero : TendstoInMeasure probability (fun _ : ℕ => fun _ : Ω => (0 : ℝ)) atTop 0 := by
    rw [tendstoInMeasure_iff_norm]
    intro ε hε
    simp [not_le.mpr hε]
  have hNumerator := plr_linear_plus_remainders_clt probability
    (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
      (reference n ω - theta n) / sigma)
    (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
      (actual n ω - reference n ω) / sigma)
    (fun _ _ => 0) hReference hPointDifference hZero
    hmPointDifference (fun _ => aemeasurable_const)
  have hNumeratorActual : TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) *
        (actual n ω - theta n) / sigma)
      atTop id (fun _ => probability) (gaussianReal 0 1) := by
    convert hNumerator using 1
    funext n ω
    ring
  have hVarianceDifference' : TendstoInMeasure probability
      (fun n ω => (n + 1) * varianceActual n ω -
        (n + 1) * varianceReference n ω) atTop 0 := by
    simpa only [mul_sub] using hVarianceDifference
  have hVarianceActual := plr_variance_consistency_transfer probability _ _ (sigma ^ 2)
    hVarianceReference hVarianceDifference'
  have hStudentized := plr_studentize probability _ _ sigma hSigma
    hNumeratorActual hVarianceActual
    (fun n => aemeasurable_const.mul (hmVarianceActual n))
  apply hStudentized.congr
    (fun n => Filter.Eventually.of_forall fun ω => ?_) Filter.EventuallyEq.rfl
  have hn : (0 : ℝ) < n + 1 := by positivity
  rw [Real.sqrt_mul (le_of_lt hn)]
  have hSqrt : Real.sqrt (n + 1) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hn)
  calc
    _ = (Real.sqrt (n + 1) * (actual n ω - theta n)) /
        (Real.sqrt (n + 1) * Real.sqrt (varianceActual n ω)) := by
      congr 1
      field_simp [ne_of_gt hSigma]
    _ = _ := mul_div_mul_left _ _ hSqrt

end StudentizedTransfer

/-! ## Six-source failure ledger for each fixed latent cluster -/

/-- All events needed for an actually reported cluster-specific interval. -/
def thetaKPipelineSuccess {Ω : Type*}
    (labelGood ySideGood stabilityGood adjustmentGood usableGood inferenceGood : Set Ω) : Set Ω :=
  (((((labelGood ∩ ySideGood) ∩ stabilityGood) ∩ adjustmentGood) ∩ usableGood) ∩
    inferenceGood)

/-- Six named complement budgets bound failure of the full cluster-specific inference event. -/
theorem measure_thetaKPipelineFailure_le
    {Ω : Type*} [MeasurableSpace Ω] (measure : Measure Ω)
    (labelGood ySideGood stabilityGood adjustmentGood usableGood inferenceGood : Set Ω)
    (deltaLabel deltaYSide deltaStability deltaAdjustment deltaUsable deltaInference : ENNReal)
    (hLabel : measure labelGoodᶜ ≤ deltaLabel)
    (hYSide : measure ySideGoodᶜ ≤ deltaYSide)
    (hStability : measure stabilityGoodᶜ ≤ deltaStability)
    (hAdjustment : measure adjustmentGoodᶜ ≤ deltaAdjustment)
    (hUsable : measure usableGoodᶜ ≤ deltaUsable)
    (hInference : measure inferenceGoodᶜ ≤ deltaInference) :
    measure (thetaKPipelineSuccess labelGood ySideGood stabilityGood adjustmentGood
      usableGood inferenceGood)ᶜ ≤
      deltaLabel + deltaYSide + deltaStability + deltaAdjustment + deltaUsable +
        deltaInference := by
  simp only [thetaKPipelineSuccess, Set.compl_inter]
  calc
    measure (((((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ) ∪ adjustmentGoodᶜ) ∪
        usableGoodᶜ) ∪ inferenceGoodᶜ) ≤
      measure ((((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ) ∪ adjustmentGoodᶜ) ∪
        usableGoodᶜ) + measure inferenceGoodᶜ :=
      measure_union_le
        (((((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ) ∪ adjustmentGoodᶜ) ∪
          usableGoodᶜ)) inferenceGoodᶜ
    _ ≤ (measure (((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ) ∪ adjustmentGoodᶜ) +
        measure usableGoodᶜ) + measure inferenceGoodᶜ := by
      gcongr
      exact measure_union_le
        ((((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ) ∪ adjustmentGoodᶜ))
        usableGoodᶜ
    _ ≤ ((measure ((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ) +
        measure adjustmentGoodᶜ) + measure usableGoodᶜ) + measure inferenceGoodᶜ := by
      gcongr
      exact measure_union_le
        (((labelGoodᶜ ∪ ySideGoodᶜ) ∪ stabilityGoodᶜ)) adjustmentGoodᶜ
    _ ≤ (((measure (labelGoodᶜ ∪ ySideGoodᶜ) + measure stabilityGoodᶜ) +
        measure adjustmentGoodᶜ) + measure usableGoodᶜ) + measure inferenceGoodᶜ := by
      gcongr
      exact measure_union_le (labelGoodᶜ ∪ ySideGoodᶜ) stabilityGoodᶜ
    _ ≤ ((((measure labelGoodᶜ + measure ySideGoodᶜ) + measure stabilityGoodᶜ) +
        measure adjustmentGoodᶜ) + measure usableGoodᶜ) + measure inferenceGoodᶜ := by
      gcongr
      exact measure_union_le labelGoodᶜ ySideGoodᶜ
    _ ≤ ((((deltaLabel + deltaYSide) + deltaStability) + deltaAdjustment) + deltaUsable) +
        deltaInference := by
      exact add_le_add
        (add_le_add
          (add_le_add (add_le_add (add_le_add hLabel hYSide) hStability) hAdjustment)
          hUsable)
        hInference
    _ = deltaLabel + deltaYSide + deltaStability + deltaAdjustment + deltaUsable +
        deltaInference := rfl

/-- If every named failure probability vanishes, the complete success event has probability tending
to one. -/
theorem thetaK_pipeline_failure_tendsto
    {Ω : Type*} [MeasurableSpace Ω] (measure : Measure Ω)
    (labelGood ySideGood stabilityGood adjustmentGood usableGood inferenceGood : ℕ → Set Ω)
    (hLabel : Tendsto (fun n => measure (labelGood n)ᶜ) atTop (𝓝 0))
    (hYSide : Tendsto (fun n => measure (ySideGood n)ᶜ) atTop (𝓝 0))
    (hStability : Tendsto (fun n => measure (stabilityGood n)ᶜ) atTop (𝓝 0))
    (hAdjustment : Tendsto (fun n => measure (adjustmentGood n)ᶜ) atTop (𝓝 0))
    (hUsable : Tendsto (fun n => measure (usableGood n)ᶜ) atTop (𝓝 0))
    (hInference : Tendsto (fun n => measure (inferenceGood n)ᶜ) atTop (𝓝 0)) :
    Tendsto (fun n => measure (thetaKPipelineSuccess (labelGood n) (ySideGood n)
      (stabilityGood n) (adjustmentGood n) (usableGood n) (inferenceGood n))ᶜ)
      atTop (𝓝 0) := by
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by
      simpa using (((((hLabel.add hYSide).add hStability).add hAdjustment).add hUsable).add
        hInference))
  · intro n
    exact bot_le
  · intro n
    exact measure_thetaKPipelineFailure_le measure
      (labelGood n) (ySideGood n) (stabilityGood n) (adjustmentGood n)
      (usableGood n) (inferenceGood n)
      (measure (labelGood n)ᶜ) (measure (ySideGood n)ᶜ) (measure (stabilityGood n)ᶜ)
      (measure (adjustmentGood n)ᶜ) (measure (usableGood n)ᶜ)
      (measure (inferenceGood n)ᶜ) le_rfl le_rfl le_rfl le_rfl le_rfl le_rfl

section CompleteCoverage

variable {Ω : Type*} [MeasurableSpace Ω]
variable (probability : Measure Ω) [IsProbabilityMeasure probability]

/-- Final unconditional reported interval for each fixed latent cluster. The success event itself is
the availability event, so failed labels, screening, stability, identification, PLR usability, or
reference inference remain noncoverage. -/
theorem thetaK_complete_reported_coverage
    (estimate standardError : ℕ → Ω → ℝ) (theta : ℕ → ℝ)
    (z : ℝ) (hz : 0 ≤ z)
    (labelGood ySideGood stabilityGood adjustmentGood usableGood inferenceGood : ℕ → Set Ω)
    (hMeasurable : ∀ n, MeasurableSet
      (thetaKPipelineSuccess (labelGood n) (ySideGood n) (stabilityGood n)
        (adjustmentGood n) (usableGood n) (inferenceGood n)))
    (hPositive : ∀ n ω, ω ∈ thetaKPipelineSuccess (labelGood n) (ySideGood n)
      (stabilityGood n) (adjustmentGood n) (usableGood n) (inferenceGood n) →
      0 < standardError n ω)
    (hLabel : Tendsto (fun n => probability (labelGood n)ᶜ) atTop (𝓝 0))
    (hYSide : Tendsto (fun n => probability (ySideGood n)ᶜ) atTop (𝓝 0))
    (hStability : Tendsto (fun n => probability (stabilityGood n)ᶜ) atTop (𝓝 0))
    (hAdjustment : Tendsto (fun n => probability (adjustmentGood n)ᶜ) atTop (𝓝 0))
    (hUsable : Tendsto (fun n => probability (usableGood n)ᶜ) atTop (𝓝 0))
    (hInference : Tendsto (fun n => probability (inferenceGood n)ᶜ) atTop (𝓝 0))
    (hStatistic : TendstoInDistribution
      (fun n ω => (estimate n ω - theta n) / standardError n ω)
      atTop id (fun _ => probability) (gaussianReal 0 1)) :
    Tendsto (fun n => probability.real {ω |
      ω ∈ thetaKPipelineSuccess (labelGood n) (ySideGood n) (stabilityGood n)
        (adjustmentGood n) (usableGood n) (inferenceGood n) ∧
      estimate n ω - z * standardError n ω ≤ theta n ∧
      theta n ≤ estimate n ω + z * standardError n ω})
      atTop (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) := by
  exact thetaK_reported_interval_coverage probability estimate standardError theta z hz
    (fun n => thetaKPipelineSuccess (labelGood n) (ySideGood n) (stabilityGood n)
      (adjustmentGood n) (usableGood n) (inferenceGood n))
    hMeasurable hPositive
    (thetaK_pipeline_failure_tendsto probability labelGood ySideGood stabilityGood adjustmentGood
      usableGood inferenceGood hLabel hYSide hStability hAdjustment hUsable hInference)
    hStatistic

end CompleteCoverage

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
