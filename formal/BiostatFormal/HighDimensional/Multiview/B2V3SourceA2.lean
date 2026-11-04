import BiostatFormal.HighDimensional.Multiview.B2V3A2
import BiostatFormal.HighDimensional.Multiview.B2V3Connections
import BiostatFormal.HighDimensional.Multiview.B2V3PseudoTrueRisk
import Mathlib.Tactic.Positivity

/-!
# B2-v3 source primary risk and A2 bridge

This file is the mathematical shadow of the preregistered source implementation.  The primary Y
and X components are individual-weighted means of clipped Bernoulli negative log loss.  Both use
the deterministic identity coordinate, so no validation-selected denominator occurs.

The sixty-index theorem is suitable when one independent evaluation law controls the complete
envelope.  The two-direction theorem records the actual AB/BA factorization: condition and
decondition each direction separately, then union the two marginal failure events.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

set_option maxHeartbeats 200000
set_option maxRecDepth 10000

noncomputable section

universe uSample uOmega uContextAB uContextBA

/-- The source primary coordinate is preregistered, deterministic, and common to Y and X. -/
def b2V3SourceIdentityCoordinate : CommonRiskCoordinate where
  location := fun _ ↦ 0
  scale := fun _ ↦ 1
  scale_pos := fun _ ↦ by norm_num

@[simp] theorem b2V3SourceIdentityCoordinate_transform
    (component : RiskComponent) (rawRisk : ℝ) :
    b2V3SourceIdentityCoordinate.transform component rawRisk = rawRisk := by
  simp [b2V3SourceIdentityCoordinate, CommonRiskCoordinate.transform]

/-- The exact source row loss used by both primary views. -/
def b2V3SourcePrimaryLoss (ε : ℝ) (response : Bool) (probability : ℝ) : ℝ :=
  bernoulliNegLogLoss response (clipProbability ε probability)

theorem b2V3SourcePrimaryLoss_mem_Icc
    (ε probability : ℝ) (response : Bool)
    (hε : 0 < ε) (hεHalf : ε ≤ 1 / 2) :
    b2V3SourcePrimaryLoss ε response probability ∈ Set.Icc 0 (-Real.log ε) := by
  exact clippedBernoulliNegLogLoss_mem_Icc ε probability response hε hεHalf

/-- Conditional expected source loss for a Bernoulli response with success probability `q`. -/
def b2V3SourceConditionalBernoulliRisk
    (ε q predictedProbability : ℝ) : ℝ :=
  q * b2V3SourcePrimaryLoss ε true predictedProbability +
    (1 - q) * b2V3SourcePrimaryLoss ε false predictedProbability

/-- Directional class mismatch cost: wrong-class conditional risk minus correct-class risk. -/
def b2V3SourceConditionalPairGap
    (ε q correctProbability wrongProbability : ℝ) : ℝ :=
  b2V3SourceConditionalBernoulliRisk ε q wrongProbability -
    b2V3SourceConditionalBernoulliRisk ε q correctProbability

theorem b2V3SourceConditionalPairGap_eq
    (ε q correctProbability wrongProbability : ℝ) :
    b2V3SourceConditionalPairGap ε q correctProbability wrongProbability =
      q * (b2V3SourcePrimaryLoss ε true wrongProbability -
        b2V3SourcePrimaryLoss ε true correctProbability) +
      (1 - q) * (b2V3SourcePrimaryLoss ε false wrongProbability -
        b2V3SourcePrimaryLoss ε false correctProbability) := by
  unfold b2V3SourceConditionalPairGap b2V3SourceConditionalBernoulliRisk
  ring

/-- Population pair gap after integrating the exact conditional Bernoulli formula. -/
noncomputable def b2V3SourceIntegratedPairGap
    {Covariate : Type*} [MeasurableSpace Covariate]
    (μ : Measure Covariate) (ε : ℝ)
    (trueProbability correctProbability wrongProbability : Covariate → ℝ) : ℝ :=
  ∫ covariate,
    b2V3SourceConditionalPairGap ε (trueProbability covariate)
      (correctProbability covariate) (wrongProbability covariate) ∂μ

/-- Equal individual-row weight in a held-out direction. -/
def b2V3SourceRowWeight (validationRows : ℕ) : Fin validationRows → ℝ :=
  fun _ ↦ 1 / (validationRows : ℝ)

theorem sum_sq_b2V3SourceRowWeight
    (validationRows : ℕ) (hRows : 0 < validationRows) :
    ∑ row, b2V3SourceRowWeight validationRows row ^ 2 =
      1 / (validationRows : ℝ) := by
  simp [b2V3SourceRowWeight]
  field_simp

/-- Source primary empirical risk for one view and one frozen record. -/
def b2V3SourceEmpiricalViewRisk
    {Candidate : Type*} {validationRows : ℕ} {Sample : Type uSample}
    (component : RiskComponent)
    (response : Candidate → Fin validationRows → Sample → Bool)
    (probability : Candidate → Fin validationRows → Sample → ℝ)
    (ε : ℝ) (candidate : Candidate) (sample : Sample) : ℝ :=
  b2V3DirectionalEmpiricalViewRisk b2V3SourceIdentityCoordinate component
    (fun _ ↦ b2V3SourceRowWeight validationRows)
    (fun record row omega ↦
      b2V3SourcePrimaryLoss ε (response record row omega) (probability record row omega))
    candidate sample

/-- Matching conditional population risk after the record predictor has been frozen. -/
noncomputable def b2V3SourcePopulationViewRisk
    {Candidate : Type*} {validationRows : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample) (component : RiskComponent)
    (response : Candidate → Fin validationRows → Sample → Bool)
    (probability : Candidate → Fin validationRows → Sample → ℝ)
    (ε : ℝ) (candidate : Candidate) : ℝ :=
  b2V3DirectionalPopulationViewRisk μ b2V3SourceIdentityCoordinate component
    (fun _ ↦ b2V3SourceRowWeight validationRows)
    (fun record row omega ↦
      b2V3SourcePrimaryLoss ε (response record row omega) (probability record row omega))
    candidate

/-- Actual source empirical primary risk: exactly one half Y plus one half X. -/
def b2V3SourceEmpiricalPrimaryRisk
    {Candidate : Type*} {validationRows : ℕ} {Sample : Type uSample}
    (outcomeResponse treatmentResponse :
      Candidate → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      Candidate → Fin validationRows → Sample → ℝ)
    (ε : ℝ) (candidate : Candidate) (sample : Sample) : ℝ :=
  (b2V3SourceEmpiricalViewRisk .outcome outcomeResponse outcomeProbability
      ε candidate sample +
    b2V3SourceEmpiricalViewRisk .treatment treatmentResponse treatmentProbability
      ε candidate sample) / 2

/-- Matching conditional population primary risk in the identical coordinate. -/
noncomputable def b2V3SourcePopulationPrimaryRisk
    {Candidate : Type*} {validationRows : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (outcomeResponse treatmentResponse :
      Candidate → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      Candidate → Fin validationRows → Sample → ℝ)
    (ε : ℝ) (candidate : Candidate) : ℝ :=
  (b2V3SourcePopulationViewRisk μ .outcome outcomeResponse outcomeProbability ε candidate +
    b2V3SourcePopulationViewRisk μ .treatment treatmentResponse treatmentProbability
      ε candidate) / 2

theorem b2V3SourceEmpiricalPrimaryRisk_eq
    {Candidate : Type*} {validationRows : ℕ} {Sample : Type uSample}
    (outcomeResponse treatmentResponse :
      Candidate → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      Candidate → Fin validationRows → Sample → ℝ)
    (ε : ℝ) (candidate : Candidate) (sample : Sample) :
    b2V3SourceEmpiricalPrimaryRisk outcomeResponse treatmentResponse
        outcomeProbability treatmentProbability ε candidate sample =
      b2V3FusionRisk
        (fun record ↦ b2V3SourceEmpiricalViewRisk .outcome outcomeResponse
          outcomeProbability ε record sample)
        (fun record ↦ b2V3SourceEmpiricalViewRisk .treatment treatmentResponse
          treatmentProbability ε record sample) candidate := by
  rfl

theorem b2V3SourcePopulationPrimaryRisk_eq
    {Candidate : Type*} {validationRows : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (outcomeResponse treatmentResponse :
      Candidate → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      Candidate → Fin validationRows → Sample → ℝ)
    (ε : ℝ) (candidate : Candidate) :
    b2V3SourcePopulationPrimaryRisk μ outcomeResponse treatmentResponse
        outcomeProbability treatmentProbability ε candidate =
      b2V3FusionRisk
        (b2V3SourcePopulationViewRisk μ .outcome outcomeResponse outcomeProbability ε)
        (b2V3SourcePopulationViewRisk μ .treatment treatmentResponse
          treatmentProbability ε) candidate := by
  rfl

/-- Scalar fusion radius used by the source selector. -/
def b2V3SourceUA2 (uY uX : ℝ) : ℝ := (uY + uX) / 2

/-- The exact two-sided pointwise Hoeffding tail in the identity coordinate. -/
noncomputable def b2V3SourcePointwiseTail
    (component : RiskComponent) (ε : ℝ) (validationRows : ℕ) (u : ℝ) : ENNReal :=
  b2V3ViewTailBudget b2V3SourceIdentityCoordinate component
    (fun _ : Unit ↦ b2V3SourceRowWeight validationRows)
    (fun _ ↦ -Real.log ε) (fun _ ↦ u) ()

theorem b2V3SourcePointwiseTail_eq_formula
    (component : RiskComponent) (ε u : ℝ) (validationRows : ℕ)
    (hε : 0 < ε) (hεOne : ε < 1) (hRows : 0 < validationRows) :
    b2V3SourcePointwiseTail component ε validationRows u =
      ENNReal.ofReal
        (2 * Real.exp
          (-2 * (validationRows : ℝ) * u ^ 2 / (-Real.log ε) ^ 2)) := by
  have hEnvelopePos : 0 < -Real.log ε :=
    neg_pos.mpr (Real.log_neg hε hεOne)
  unfold b2V3SourcePointwiseTail b2V3ViewTailBudget
  rw [b2PointwiseHoeffdingProxy_coe _ _ hEnvelopePos.le]
  rw [sum_sq_b2V3SourceRowWeight validationRows hRows]
  simp only [b2V3SourceIdentityCoordinate, one_mul]
  congr 2
  field_simp [hEnvelopePos.ne', Nat.cast_ne_zero.mpr hRows.ne']
  ring_nf

theorem b2V3SourcePointwiseTail_antitone_validationRows
    (component : RiskComponent) (ε u : ℝ) {minimumRows validationRows : ℕ}
    (hε : 0 < ε) (hεOne : ε < 1) (hMinimumRows : 0 < minimumRows)
    (hRows : minimumRows ≤ validationRows) :
    b2V3SourcePointwiseTail component ε validationRows u ≤
      b2V3SourcePointwiseTail component ε minimumRows u := by
  have hValidationRows : 0 < validationRows := hMinimumRows.trans_le hRows
  rw [b2V3SourcePointwiseTail_eq_formula component ε u validationRows
      hε hεOne hValidationRows]
  rw [b2V3SourcePointwiseTail_eq_formula component ε u minimumRows
      hε hεOne hMinimumRows]
  apply ENNReal.ofReal_le_ofReal
  apply mul_le_mul_of_nonneg_left
  · apply Real.exp_le_exp.mpr
    have hRowsReal : (minimumRows : ℝ) ≤ validationRows := by exact_mod_cast hRows
    have hEnvelopeSq : 0 < (-Real.log ε) ^ 2 :=
      sq_pos_of_pos (neg_pos.mpr (Real.log_neg hε hεOne))
    apply (div_le_div_iff_of_pos_right hEnvelopeSq).2
    nlinarith [sq_nonneg u]
  · norm_num

/-- Total Y or X budget after union over all sixty preregistered records. -/
noncomputable def b2V3SourceViewDelta
    (component : RiskComponent) (ε : ℝ) (validationRows : ℕ) (u : ℝ) : ENNReal :=
  60 * b2V3SourcePointwiseTail component ε validationRows u

/-- Final source A2 budget; both terms are already whole-envelope component budgets. -/
noncomputable def b2V3SourceDeltaD
    (ε : ℝ) (validationRows : ℕ) (uY uX : ℝ) : ENNReal :=
  b2V3SourceViewDelta .outcome ε validationRows uY +
    b2V3SourceViewDelta .treatment ε validationRows uX

/-- One direction contains 15 slots and two stages, hence 30 source records. -/
noncomputable def b2V3SourceDirectionDelta
    (ε : ℝ) (validationRows : ℕ) (uY uX : ℝ) : ENNReal :=
  30 * (b2V3SourcePointwiseTail .outcome ε validationRows uY +
    b2V3SourcePointwiseTail .treatment ε validationRows uX)

/-- Registered source constants used by the frozen B2-v3 grid. -/
def b2V3RegisteredEpsilonProb : ℝ := 1 / 100
def b2V3RegisteredMinimumValidationRows : ℕ := 2700
def b2V3RegisteredDeltaY : ℝ := 1 / 40
def b2V3RegisteredDeltaX : ℝ := 1 / 40

/-- Inversion of the two-sided 60-record Hoeffding budget. -/
noncomputable def b2V3RegisteredViewRadius : ℝ :=
  -Real.log b2V3RegisteredEpsilonProb *
    Real.sqrt
      (Real.log (120 / b2V3RegisteredDeltaY) /
        (2 * b2V3RegisteredMinimumValidationRows))

noncomputable def b2V3RegisteredUA2 : ℝ :=
  b2V3SourceUA2 b2V3RegisteredViewRadius b2V3RegisteredViewRadius

theorem b2V3RegisteredViewRadius_nonneg : 0 ≤ b2V3RegisteredViewRadius := by
  unfold b2V3RegisteredViewRadius b2V3RegisteredEpsilonProb b2V3RegisteredDeltaY
  have hLog : Real.log (1 / 100 : ℝ) < 0 :=
    Real.log_neg (by norm_num) (by norm_num)
  exact mul_nonneg (le_of_lt (neg_pos.mpr hLog)) (Real.sqrt_nonneg _)

/-- The strict source-margin region is nonempty after the registered A2 radius is frozen. -/
theorem b2V3RegisteredStrictMarginRegion_nonempty
    (centerCount : ℕ) (hCenterCount : centerCount = 9 ∨ centerCount = 12) :
    ∃ gMin : ℝ, 0 < gMin ∧
      b2V3StrictMargin centerCount gMin 0 1
        b2V3RegisteredViewRadius b2V3RegisteredViewRadius 0 := by
  exact b2V3StrictMarginRegion_nonempty_for_fixed_radii centerCount hCenterCount
    b2V3RegisteredViewRadius b2V3RegisteredViewRadius 0
    b2V3RegisteredViewRadius_nonneg b2V3RegisteredViewRadius_nonneg (by norm_num)

theorem b2V3RegisteredDeltaD_real :
    b2V3RegisteredDeltaY + b2V3RegisteredDeltaX = (1 / 20 : ℝ) := by
  norm_num [b2V3RegisteredDeltaY, b2V3RegisteredDeltaX]

theorem b2V3RegisteredHoeffdingProxy_eq :
    (b2PointwiseHoeffdingProxy
        (b2V3SourceRowWeight b2V3RegisteredMinimumValidationRows)
        (-Real.log b2V3RegisteredEpsilonProb) : ℝ) =
      (-Real.log b2V3RegisteredEpsilonProb) ^ 2 /
        (4 * b2V3RegisteredMinimumValidationRows) := by
  rw [b2PointwiseHoeffdingProxy_coe]
  · rw [sum_sq_b2V3SourceRowWeight]
    · ring
    · norm_num [b2V3RegisteredMinimumValidationRows]
  · unfold b2V3RegisteredEpsilonProb
    have hLog : Real.log (1 / 100 : ℝ) < 0 :=
      Real.log_neg (by norm_num) (by norm_num)
    linarith

theorem b2V3RegisteredViewRadius_sq :
    b2V3RegisteredViewRadius ^ 2 =
      (-Real.log b2V3RegisteredEpsilonProb) ^ 2 *
        Real.log (120 / b2V3RegisteredDeltaY) /
          (2 * b2V3RegisteredMinimumValidationRows) := by
  have hLog : 0 ≤ Real.log (120 / b2V3RegisteredDeltaY) := by
    apply Real.log_nonneg
    norm_num [b2V3RegisteredDeltaY]
  unfold b2V3RegisteredViewRadius
  rw [mul_pow, Real.sq_sqrt]
  · ring
  · positivity

theorem b2V3RegisteredPointwiseTail_eq
    (component : RiskComponent) :
    b2V3SourcePointwiseTail component b2V3RegisteredEpsilonProb
        b2V3RegisteredMinimumValidationRows b2V3RegisteredViewRadius =
      (1 / 2400 : ENNReal) := by
  unfold b2V3SourcePointwiseTail b2V3ViewTailBudget
  rw [b2V3RegisteredHoeffdingProxy_eq]
  simp only [b2V3SourceIdentityCoordinate, one_mul]
  have hLogEpsilon : Real.log b2V3RegisteredEpsilonProb ≠ 0 := by
    unfold b2V3RegisteredEpsilonProb
    have h := Real.log_neg (by norm_num : (0 : ℝ) < 1 / 100)
      (by norm_num : (1 / 100 : ℝ) < 1)
    linarith
  have hEnvelopeSq : (-Real.log b2V3RegisteredEpsilonProb) ^ 2 ≠ 0 := by
    exact pow_ne_zero 2 (neg_ne_zero.mpr hLogEpsilon)
  have hRows : (b2V3RegisteredMinimumValidationRows : ℝ) ≠ 0 := by
    norm_num [b2V3RegisteredMinimumValidationRows]
  have hExponent :
      -(b2V3RegisteredViewRadius ^ 2) /
          (2 * ((-Real.log b2V3RegisteredEpsilonProb) ^ 2 /
            (4 * b2V3RegisteredMinimumValidationRows))) =
        -Real.log (120 / b2V3RegisteredDeltaY) := by
    rw [b2V3RegisteredViewRadius_sq]
    field_simp [hEnvelopeSq, hRows]
    ring
  rw [hExponent, Real.exp_neg, Real.exp_log]
  · apply (ENNReal.toReal_eq_toReal_iff' (by finiteness) (by finiteness)).mp
    simp [ENNReal.toReal_mul, b2V3RegisteredDeltaY]
    norm_num
  · norm_num [b2V3RegisteredDeltaY]

/-- The registered pointwise tail remains valid for every larger held-out direction. -/
theorem b2V3RegisteredPointwiseTail_le
    (component : RiskComponent) {validationRows : ℕ}
    (hRows : b2V3RegisteredMinimumValidationRows ≤ validationRows) :
    b2V3SourcePointwiseTail component b2V3RegisteredEpsilonProb
        validationRows b2V3RegisteredViewRadius ≤
      (1 / 2400 : ENNReal) := by
  calc
    b2V3SourcePointwiseTail component b2V3RegisteredEpsilonProb
        validationRows b2V3RegisteredViewRadius ≤
      b2V3SourcePointwiseTail component b2V3RegisteredEpsilonProb
        b2V3RegisteredMinimumValidationRows b2V3RegisteredViewRadius := by
          apply b2V3SourcePointwiseTail_antitone_validationRows
          · norm_num [b2V3RegisteredEpsilonProb]
          · norm_num [b2V3RegisteredEpsilonProb]
          · norm_num [b2V3RegisteredMinimumValidationRows]
          · exact hRows
    _ = (1 / 2400 : ENNReal) := b2V3RegisteredPointwiseTail_eq component

theorem b2V3RegisteredDeltaD_eq :
    b2V3SourceDeltaD b2V3RegisteredEpsilonProb
        b2V3RegisteredMinimumValidationRows
        b2V3RegisteredViewRadius b2V3RegisteredViewRadius =
      (1 / 20 : ENNReal) := by
  simp [b2V3SourceDeltaD, b2V3SourceViewDelta,
    b2V3RegisteredPointwiseTail_eq, div_eq_mul_inv]
  apply (ENNReal.toReal_eq_toReal_iff' (by finiteness) (by finiteness)).mp
  rw [ENNReal.toReal_add (by finiteness) (by finiteness)]
  simp [ENNReal.toReal_inv]
  norm_num

theorem b2V3RegisteredDirectionDelta_eq :
    b2V3SourceDirectionDelta b2V3RegisteredEpsilonProb
        b2V3RegisteredMinimumValidationRows
        b2V3RegisteredViewRadius b2V3RegisteredViewRadius =
      (1 / 40 : ENNReal) := by
  simp [b2V3SourceDirectionDelta, b2V3RegisteredPointwiseTail_eq,
    div_eq_mul_inv]
  apply (ENNReal.toReal_eq_toReal_iff' (by finiteness) (by finiteness)).mp
  rw [ENNReal.toReal_mul]
  rw [ENNReal.toReal_add (by finiteness) (by finiteness)]
  simp [ENNReal.toReal_inv]
  norm_num

/-- The 30 frozen records belonging to one direction. -/
def b2V3SourceDirectionEnvelope (direction : Direction) :
    Finset B2CandidateRecordIndex :=
  b2V3CandidateRecordEnvelope.filter fun record ↦ record.2.1 = direction

theorem b2V3SourceDirectionEnvelope_card (direction : Direction) :
    (b2V3SourceDirectionEnvelope direction).card = 30 := by
  cases direction <;> decide

/-- Source weighted-Hoeffding union bound for any frozen finite record pool. -/
theorem measure_b2V3SourceFusionUniformDeviation_le_card_mul
    {Candidate : Type*} {validationRows : ℕ}
    {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (pool : Finset Candidate)
    (outcomeResponse treatmentResponse :
      Candidate → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      Candidate → Fin validationRows → Sample → ℝ)
    (ε uY uX : ℝ)
    (hε : 0 < ε) (hεHalf : ε ≤ 1 / 2)
    (hOutcomeIndep : ∀ candidate ∈ pool,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss ε
          (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentIndep : ∀ candidate ∈ pool,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss ε
          (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hOutcomeMeasurable : ∀ candidate ∈ pool, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss ε
        (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentMeasurable : ∀ candidate ∈ pool, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss ε
        (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hUY : 0 ≤ uY) (hUX : 0 ≤ uX) :
    μ (uniformDeviationFailureEvent pool
        (b2V3SourceEmpiricalPrimaryRisk outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability ε)
        (b2V3SourcePopulationPrimaryRisk μ outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability ε)
        (fun _ ↦ b2V3SourceUA2 uY uX)) ≤
      pool.card *
        (b2V3SourcePointwiseTail .outcome ε validationRows uY +
          b2V3SourcePointwiseTail .treatment ε validationRows uX) := by
  classical
  let outcomeLoss := fun candidate row sample ↦ b2V3SourcePrimaryLoss ε
    (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)
  let treatmentLoss := fun candidate row sample ↦ b2V3SourcePrimaryLoss ε
    (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)
  have hBound := measure_b2V3FrozenFusionUniformDeviation_le_sum μ
    pool b2V3SourceIdentityCoordinate
    (fun _ ↦ b2V3SourceRowWeight validationRows)
    (fun _ ↦ b2V3SourceRowWeight validationRows)
    outcomeLoss treatmentLoss
    (fun _ ↦ -Real.log ε) (fun _ ↦ -Real.log ε)
    (fun _ ↦ uY) (fun _ ↦ uX)
    hOutcomeIndep hTreatmentIndep hOutcomeMeasurable hTreatmentMeasurable
    (by
      intro candidate _ row
      filter_upwards [] with sample
      exact b2V3SourcePrimaryLoss_mem_Icc ε
        (outcomeProbability candidate row sample) (outcomeResponse candidate row sample)
        hε hεHalf)
    (by
      intro candidate _ row
      filter_upwards [] with sample
      exact b2V3SourcePrimaryLoss_mem_Icc ε
        (treatmentProbability candidate row sample) (treatmentResponse candidate row sample)
        hε hεHalf)
    (by intro _ _; exact hUY) (by intro _ _; exact hUX)
  have hOutcomeTail : ∀ candidate : Candidate,
      b2V3ViewTailBudget b2V3SourceIdentityCoordinate .outcome
          (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
          (fun _ ↦ uY) candidate =
        b2V3SourcePointwiseTail .outcome ε validationRows uY := by
    intro candidate
    rfl
  have hTreatmentTail : ∀ candidate : Candidate,
      b2V3ViewTailBudget b2V3SourceIdentityCoordinate .treatment
          (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
          (fun _ ↦ uX) candidate =
        b2V3SourcePointwiseTail .treatment ε validationRows uX := by
    intro candidate
    rfl
  change μ (uniformDeviationFailureEvent pool
      (b2V3DirectionalEmpiricalFusionRisk b2V3SourceIdentityCoordinate
        (fun _ ↦ b2V3SourceRowWeight validationRows)
        (fun _ ↦ b2V3SourceRowWeight validationRows) outcomeLoss treatmentLoss)
      (b2V3DirectionalPopulationFusionRisk μ b2V3SourceIdentityCoordinate
        (fun _ ↦ b2V3SourceRowWeight validationRows)
        (fun _ ↦ b2V3SourceRowWeight validationRows) outcomeLoss treatmentLoss)
      (b2V3FusionThreshold (fun _ ↦ uY) (fun _ ↦ uX))) ≤
    pool.card *
      (b2V3SourcePointwiseTail .outcome ε validationRows uY +
        b2V3SourcePointwiseTail .treatment ε validationRows uX)
  calc
    _ ≤ ∑ candidate ∈ pool,
        (b2V3ViewTailBudget b2V3SourceIdentityCoordinate .outcome
            (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
            (fun _ ↦ uY) candidate +
          b2V3ViewTailBudget b2V3SourceIdentityCoordinate .treatment
            (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
            (fun _ ↦ uX) candidate) := hBound
    _ = pool.card *
        (b2V3SourcePointwiseTail .outcome ε validationRows uY +
          b2V3SourcePointwiseTail .treatment ε validationRows uX) := by
      simp_rw [hOutcomeTail, hTreatmentTail]
      rw [mul_add]
      simp

/-- One frozen AB or BA source direction has the registered 30-record tail budget. -/
theorem measure_b2V3SourceDirectionFusionUniformDeviation_le
    {validationRows : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (direction : Direction)
    (outcomeResponse treatmentResponse :
      B2CandidateRecordIndex → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      B2CandidateRecordIndex → Fin validationRows → Sample → ℝ)
    (ε uY uX : ℝ)
    (hε : 0 < ε) (hεHalf : ε ≤ 1 / 2)
    (hOutcomeIndep : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss ε
          (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentIndep : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss ε
          (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hOutcomeMeasurable : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss ε
        (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentMeasurable : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss ε
        (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hUY : 0 ≤ uY) (hUX : 0 ≤ uX) :
    μ (uniformDeviationFailureEvent (b2V3SourceDirectionEnvelope direction)
        (b2V3SourceEmpiricalPrimaryRisk outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability ε)
        (b2V3SourcePopulationPrimaryRisk μ outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability ε)
        (fun _ ↦ b2V3SourceUA2 uY uX)) ≤
      b2V3SourceDirectionDelta ε validationRows uY uX := by
  have h := measure_b2V3SourceFusionUniformDeviation_le_card_mul μ
    (b2V3SourceDirectionEnvelope direction) outcomeResponse treatmentResponse
    outcomeProbability treatmentProbability ε uY uX hε hεHalf
    hOutcomeIndep hTreatmentIndep hOutcomeMeasurable hTreatmentMeasurable hUY hUX
  simpa [b2V3SourceDirectionEnvelope_card, b2V3SourceDirectionDelta] using h

/-- Registered one-direction source result, valid once that direction has at least 2700 rows. -/
theorem measure_b2V3RegisteredSourceDirectionFusionUniformDeviation_le
    {validationRows : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (direction : Direction)
    (outcomeResponse treatmentResponse :
      B2CandidateRecordIndex → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      B2CandidateRecordIndex → Fin validationRows → Sample → ℝ)
    (hRows : b2V3RegisteredMinimumValidationRows ≤ validationRows)
    (hOutcomeIndep : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss b2V3RegisteredEpsilonProb
          (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentIndep : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss b2V3RegisteredEpsilonProb
          (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hOutcomeMeasurable : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss b2V3RegisteredEpsilonProb
        (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentMeasurable : ∀ candidate ∈ b2V3SourceDirectionEnvelope direction, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss b2V3RegisteredEpsilonProb
        (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ) :
    μ (uniformDeviationFailureEvent (b2V3SourceDirectionEnvelope direction)
        (b2V3SourceEmpiricalPrimaryRisk outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability b2V3RegisteredEpsilonProb)
        (b2V3SourcePopulationPrimaryRisk μ outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability b2V3RegisteredEpsilonProb)
        (fun _ ↦ b2V3RegisteredUA2)) ≤ (1 / 40 : ENNReal) := by
  have hDirection := measure_b2V3SourceDirectionFusionUniformDeviation_le μ direction
    outcomeResponse treatmentResponse outcomeProbability treatmentProbability
    b2V3RegisteredEpsilonProb b2V3RegisteredViewRadius b2V3RegisteredViewRadius
    (by norm_num [b2V3RegisteredEpsilonProb])
    (by norm_num [b2V3RegisteredEpsilonProb])
    hOutcomeIndep hTreatmentIndep hOutcomeMeasurable hTreatmentMeasurable
    b2V3RegisteredViewRadius_nonneg b2V3RegisteredViewRadius_nonneg
  have hOutcomeTail := b2V3RegisteredPointwiseTail_le .outcome hRows
  have hTreatmentTail := b2V3RegisteredPointwiseTail_le .treatment hRows
  calc
    _ ≤ b2V3SourceDirectionDelta b2V3RegisteredEpsilonProb validationRows
        b2V3RegisteredViewRadius b2V3RegisteredViewRadius := hDirection
    _ ≤ 30 * ((1 / 2400 : ENNReal) + (1 / 2400 : ENNReal)) := by
          unfold b2V3SourceDirectionDelta
          gcongr
    _ = (1 / 40 : ENNReal) := by
      apply (ENNReal.toReal_eq_toReal_iff' (by finiteness) (by finiteness)).mp
      rw [ENNReal.toReal_mul]
      rw [ENNReal.toReal_add (by finiteness) (by finiteness)]
      simp [ENNReal.toReal_inv]
      norm_num

/--
Full sixty-index conditional uniform deviation for the exact source losses.  This version applies
when all sixty records are evaluated under one independent law.
-/
theorem measure_b2V3SourceFusionUniformDeviation_le_deltaD
    {validationRows : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (outcomeResponse treatmentResponse :
      B2CandidateRecordIndex → Fin validationRows → Sample → Bool)
    (outcomeProbability treatmentProbability :
      B2CandidateRecordIndex → Fin validationRows → Sample → ℝ)
    (ε uY uX : ℝ)
    (hε : 0 < ε) (hεHalf : ε ≤ 1 / 2)
    (hOutcomeIndep : ∀ candidate ∈ b2V3CandidateRecordEnvelope,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss ε
          (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentIndep : ∀ candidate ∈ b2V3CandidateRecordEnvelope,
      ProbabilityTheory.iIndepFun
        (fun row sample ↦ b2V3SourcePrimaryLoss ε
          (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hOutcomeMeasurable : ∀ candidate ∈ b2V3CandidateRecordEnvelope, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss ε
        (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)) μ)
    (hTreatmentMeasurable : ∀ candidate ∈ b2V3CandidateRecordEnvelope, ∀ row,
      AEMeasurable (fun sample ↦ b2V3SourcePrimaryLoss ε
        (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)) μ)
    (hUY : 0 ≤ uY) (hUX : 0 ≤ uX) :
    μ (uniformDeviationFailureEvent b2V3CandidateRecordEnvelope
        (b2V3SourceEmpiricalPrimaryRisk outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability ε)
        (b2V3SourcePopulationPrimaryRisk μ outcomeResponse treatmentResponse
          outcomeProbability treatmentProbability ε)
        (fun _ ↦ b2V3SourceUA2 uY uX)) ≤
      b2V3SourceDeltaD ε validationRows uY uX := by
  let outcomeLoss := fun candidate row sample ↦ b2V3SourcePrimaryLoss ε
    (outcomeResponse candidate row sample) (outcomeProbability candidate row sample)
  let treatmentLoss := fun candidate row sample ↦ b2V3SourcePrimaryLoss ε
    (treatmentResponse candidate row sample) (treatmentProbability candidate row sample)
  have hBound := measure_b2V3FrozenFusionUniformDeviation_le_sum μ
    b2V3CandidateRecordEnvelope b2V3SourceIdentityCoordinate
    (fun _ ↦ b2V3SourceRowWeight validationRows)
    (fun _ ↦ b2V3SourceRowWeight validationRows)
    outcomeLoss treatmentLoss
    (fun _ ↦ -Real.log ε) (fun _ ↦ -Real.log ε)
    (fun _ ↦ uY) (fun _ ↦ uX)
    hOutcomeIndep hTreatmentIndep hOutcomeMeasurable hTreatmentMeasurable
    (by
      intro candidate _ row
      filter_upwards [] with sample
      exact b2V3SourcePrimaryLoss_mem_Icc ε
        (outcomeProbability candidate row sample) (outcomeResponse candidate row sample)
        hε hεHalf)
    (by
      intro candidate _ row
      filter_upwards [] with sample
      exact b2V3SourcePrimaryLoss_mem_Icc ε
        (treatmentProbability candidate row sample) (treatmentResponse candidate row sample)
        hε hεHalf)
    (by intro _ _; exact hUY) (by intro _ _; exact hUX)
  have hOutcomeTail : ∀ candidate : B2CandidateRecordIndex,
      b2V3ViewTailBudget b2V3SourceIdentityCoordinate .outcome
          (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
          (fun _ ↦ uY) candidate =
        b2V3SourcePointwiseTail .outcome ε validationRows uY := by
    intro candidate
    rfl
  have hTreatmentTail : ∀ candidate : B2CandidateRecordIndex,
      b2V3ViewTailBudget b2V3SourceIdentityCoordinate .treatment
          (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
          (fun _ ↦ uX) candidate =
        b2V3SourcePointwiseTail .treatment ε validationRows uX := by
    intro candidate
    rfl
  change μ (uniformDeviationFailureEvent b2V3CandidateRecordEnvelope
      (b2V3DirectionalEmpiricalFusionRisk b2V3SourceIdentityCoordinate
        (fun _ ↦ b2V3SourceRowWeight validationRows)
        (fun _ ↦ b2V3SourceRowWeight validationRows) outcomeLoss treatmentLoss)
      (b2V3DirectionalPopulationFusionRisk μ b2V3SourceIdentityCoordinate
        (fun _ ↦ b2V3SourceRowWeight validationRows)
        (fun _ ↦ b2V3SourceRowWeight validationRows) outcomeLoss treatmentLoss)
      (b2V3FusionThreshold (fun _ ↦ uY) (fun _ ↦ uX))) ≤
    b2V3SourceDeltaD ε validationRows uY uX
  calc
    _ ≤ ∑ candidate ∈ b2V3CandidateRecordEnvelope,
        (b2V3ViewTailBudget b2V3SourceIdentityCoordinate .outcome
            (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
            (fun _ ↦ uY) candidate +
          b2V3ViewTailBudget b2V3SourceIdentityCoordinate .treatment
            (fun _ ↦ b2V3SourceRowWeight validationRows) (fun _ ↦ -Real.log ε)
            (fun _ ↦ uX) candidate) := hBound
    _ = b2V3SourceDeltaD ε validationRows uY uX := by
      simp_rw [hOutcomeTail, hTreatmentTail]
      rw [Finset.sum_const, b2V3CandidateRecordEnvelope_card]
      simp [b2V3SourceDeltaD, b2V3SourceViewDelta]

/--
Actual AB/BA conditioning structure.  Each 30-record direction is deconditioned under its own
factorization before the two marginal failures are unioned.  A validation-selected guard subset is
handled later by the fixed-envelope set inclusion, not by conditioning on the guard.
-/
theorem measure_b2V3SourceTwoDirectionFailure_le_deltaD
    {ContextAB : Type uContextAB} {ContextBA : Type uContextBA}
    {Omega : Type uOmega}
    [MeasurableSpace ContextAB] [MeasurableSpace ContextBA] [MeasurableSpace Omega]
    (jointLaw : Measure Omega)
    (contextLawAB : Measure ContextAB) [IsProbabilityMeasure contextLawAB]
    (contextLawBA : Measure ContextBA) [IsProbabilityMeasure contextLawBA]
    (conditionalLawAB : ContextAB → Measure Omega)
    (conditionalLawBA : ContextBA → Measure Omega)
    (failureAB failureBA : Set Omega)
    (ε : ℝ) (validationRows : ℕ) (uY uX : ℝ)
    (hFailureAB : MeasurableSet failureAB) (hFailureBA : MeasurableSet failureBA)
    (hConditionalLawAB : AEMeasurable conditionalLawAB contextLawAB)
    (hConditionalLawBA : AEMeasurable conditionalLawBA contextLawBA)
    (hJointAB : jointLaw = contextLawAB.bind conditionalLawAB)
    (hJointBA : jointLaw = contextLawBA.bind conditionalLawBA)
    (hConditionalAB : ∀ context, conditionalLawAB context failureAB ≤
      b2V3SourceDirectionDelta ε validationRows uY uX)
    (hConditionalBA : ∀ context, conditionalLawBA context failureBA ≤
      b2V3SourceDirectionDelta ε validationRows uY uX) :
    jointLaw (failureAB ∪ failureBA) ≤
      b2V3SourceDeltaD ε validationRows uY uX := by
  have h := measure_twoDirectionFailure_le jointLaw contextLawAB contextLawBA
    conditionalLawAB conditionalLawBA failureAB failureBA
    (b2V3SourceDirectionDelta ε validationRows uY uX)
    (b2V3SourceDirectionDelta ε validationRows uY uX)
    hFailureAB hFailureBA hConditionalLawAB hConditionalLawBA hJointAB hJointBA
    hConditionalAB hConditionalBA
  calc
    jointLaw (failureAB ∪ failureBA) ≤
        b2V3SourceDirectionDelta ε validationRows uY uX +
          b2V3SourceDirectionDelta ε validationRows uY uX := h
    _ = b2V3SourceDeltaD ε validationRows uY uX := by
      simp only [b2V3SourceDirectionDelta, b2V3SourceDeltaD, b2V3SourceViewDelta]
      ring

/-- Registered AB/BA tower-law result: the final A2 failure probability is at most 0.05. -/
theorem measure_b2V3RegisteredTwoDirectionFailure_le
    {ContextAB : Type uContextAB} {ContextBA : Type uContextBA}
    {Omega : Type uOmega}
    [MeasurableSpace ContextAB] [MeasurableSpace ContextBA] [MeasurableSpace Omega]
    (jointLaw : Measure Omega)
    (contextLawAB : Measure ContextAB) [IsProbabilityMeasure contextLawAB]
    (contextLawBA : Measure ContextBA) [IsProbabilityMeasure contextLawBA]
    (conditionalLawAB : ContextAB → Measure Omega)
    (conditionalLawBA : ContextBA → Measure Omega)
    (failureAB failureBA : Set Omega)
    (hFailureAB : MeasurableSet failureAB) (hFailureBA : MeasurableSet failureBA)
    (hConditionalLawAB : AEMeasurable conditionalLawAB contextLawAB)
    (hConditionalLawBA : AEMeasurable conditionalLawBA contextLawBA)
    (hJointAB : jointLaw = contextLawAB.bind conditionalLawAB)
    (hJointBA : jointLaw = contextLawBA.bind conditionalLawBA)
    (hConditionalAB : ∀ context, conditionalLawAB context failureAB ≤ (1 / 40 : ENNReal))
    (hConditionalBA : ∀ context, conditionalLawBA context failureBA ≤ (1 / 40 : ENNReal)) :
    jointLaw (failureAB ∪ failureBA) ≤ (1 / 20 : ENNReal) := by
  have h := measure_b2V3SourceTwoDirectionFailure_le_deltaD jointLaw
    contextLawAB contextLawBA conditionalLawAB conditionalLawBA failureAB failureBA
    b2V3RegisteredEpsilonProb b2V3RegisteredMinimumValidationRows
    b2V3RegisteredViewRadius b2V3RegisteredViewRadius
    hFailureAB hFailureBA hConditionalLawAB hConditionalLawBA hJointAB hJointBA
    (by simpa [b2V3RegisteredDirectionDelta_eq] using hConditionalAB)
    (by simpa [b2V3RegisteredDirectionDelta_eq] using hConditionalBA)
  simpa [b2V3RegisteredDeltaD_eq] using h

end

end BiostatFormal.HighDimensional.Multiview
