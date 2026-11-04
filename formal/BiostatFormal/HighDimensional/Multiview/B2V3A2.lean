import BiostatFormal.HighDimensional.Multiview.B2V3PseudoTrueRisk
import BiostatFormal.HighDimensional.Multiview.RiskSelection
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Conditional finite-pool A2 and selector stitch for B2-v3

The pool in this file is fixed after conditioning on the training half and before observing the
independent evaluation sample.  X and Y pointwise tails are obtained from the already-verified
weighted-Hoeffding theorem.  Their union controls the equal-weight fusion risk without assuming X
and Y are independent inside a center.

Pure F and stability do not occur in any risk, threshold, event, budget, or selector declaration in
this file.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

noncomputable section

universe uCandidate uSample uCenter uLabel uSlot uContext

/-- Equal-weight empirical fusion of two already normalized view-risk functions. -/
def b2V3EmpiricalFusionFromViews
    {Candidate : Type uCandidate} {Sample : Type uSample}
    (outcomeRisk treatmentRisk : Candidate → Sample → ℝ)
    (candidate : Candidate) (sample : Sample) : ℝ :=
  (outcomeRisk candidate sample + treatmentRisk candidate sample) / 2

/-- Equal-weight population fusion of the matching normalized view risks. -/
def b2V3PopulationFusionFromViews
    {Candidate : Type uCandidate}
    (outcomeRisk treatmentRisk : Candidate → ℝ) (candidate : Candidate) : ℝ :=
  (outcomeRisk candidate + treatmentRisk candidate) / 2

/-- Candidate-specific primary radius from Y and X component radii. -/
def b2V3FusionThreshold
    {Candidate : Type uCandidate}
    (outcomeThreshold treatmentThreshold : Candidate → ℝ) (candidate : Candidate) : ℝ :=
  (outcomeThreshold candidate + treatmentThreshold candidate) / 2

/-- Deterministic composition of the Y/X component deviations. -/
theorem abs_b2V3FusionRisk_sub_le
    (empiricalOutcome empiricalTreatment populationOutcome populationTreatment : ℝ)
    (outcomeThreshold treatmentThreshold : ℝ)
    (hOutcome : |empiricalOutcome - populationOutcome| ≤ outcomeThreshold)
    (hTreatment : |empiricalTreatment - populationTreatment| ≤ treatmentThreshold) :
    |(empiricalOutcome + empiricalTreatment) / 2 -
        (populationOutcome + populationTreatment) / 2| ≤
      (outcomeThreshold + treatmentThreshold) / 2 := by
  calc
    |(empiricalOutcome + empiricalTreatment) / 2 -
        (populationOutcome + populationTreatment) / 2| =
      |(empiricalOutcome - populationOutcome) +
        (empiricalTreatment - populationTreatment)| / 2 := by
          rw [← abs_of_nonneg (show (0 : ℝ) ≤ 2 by norm_num), ← abs_div]
          congr 1
          ring
    _ ≤ (|empiricalOutcome - populationOutcome| +
          |empiricalTreatment - populationTreatment|) / 2 := by
      exact div_le_div_of_nonneg_right (abs_add_le _ _) (by norm_num)
    _ ≤ (outcomeThreshold + treatmentThreshold) / 2 := by
      exact div_le_div_of_nonneg_right (add_le_add hOutcome hTreatment) (by norm_num)

/--
If fusion exceeds its averaged radius, at least one of the Y or X component radii fails.  This event
inclusion is deterministic and requires no independence between the two views.
-/
theorem b2V3FusionUniformDeviationFailure_subset
    {Candidate : Type uCandidate} {Sample : Type uSample}
    (pool : Finset Candidate)
    (empiricalOutcome empiricalTreatment : Candidate → Sample → ℝ)
    (populationOutcome populationTreatment : Candidate → ℝ)
    (outcomeThreshold treatmentThreshold : Candidate → ℝ) :
    uniformDeviationFailureEvent pool
        (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
        (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold) ⊆
      uniformDeviationFailureEvent pool empiricalOutcome populationOutcome outcomeThreshold ∪
        uniformDeviationFailureEvent pool empiricalTreatment populationTreatment
          treatmentThreshold := by
  intro sample hSample
  rcases Set.mem_iUnion.mp hSample with ⟨candidate, hSample⟩
  rcases Set.mem_iUnion.mp hSample with ⟨hCandidate, hFailure⟩
  change
    b2V3FusionThreshold outcomeThreshold treatmentThreshold candidate <
      |b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment candidate sample -
        b2V3PopulationFusionFromViews populationOutcome populationTreatment candidate|
    at hFailure
  by_cases hOutcomeFailure :
      outcomeThreshold candidate <
        |empiricalOutcome candidate sample - populationOutcome candidate|
  · apply Set.mem_union_left
    apply Set.mem_iUnion.2
    refine ⟨candidate, ?_⟩
    apply Set.mem_iUnion.2
    exact ⟨hCandidate, hOutcomeFailure⟩
  · have hOutcomeDeviation :
        |empiricalOutcome candidate sample - populationOutcome candidate| ≤
          outcomeThreshold candidate := le_of_not_gt hOutcomeFailure
    have hTreatmentFailure :
        treatmentThreshold candidate <
          |empiricalTreatment candidate sample - populationTreatment candidate| := by
      by_contra hNoTreatmentFailure
      have hTreatmentDeviation :
          |empiricalTreatment candidate sample - populationTreatment candidate| ≤
            treatmentThreshold candidate := le_of_not_gt hNoTreatmentFailure
      have hFusionDeviation := abs_b2V3FusionRisk_sub_le
        (empiricalOutcome candidate sample) (empiricalTreatment candidate sample)
        (populationOutcome candidate) (populationTreatment candidate)
        (outcomeThreshold candidate) (treatmentThreshold candidate)
        hOutcomeDeviation hTreatmentDeviation
      exact (not_lt_of_ge hFusionDeviation) hFailure
    apply Set.mem_union_right
    apply Set.mem_iUnion.2
    refine ⟨candidate, ?_⟩
    apply Set.mem_iUnion.2
    exact ⟨hCandidate, hTreatmentFailure⟩

/-- Candidate-specific tail budgets for the fixed conditional pool. -/
theorem measure_b2V3FusionUniformDeviation_le_sum
    {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample) (pool : Finset Candidate)
    (empiricalOutcome empiricalTreatment : Candidate → Sample → ℝ)
    (populationOutcome populationTreatment : Candidate → ℝ)
    (outcomeThreshold treatmentThreshold : Candidate → ℝ)
    (outcomeTail treatmentTail : Candidate → ENNReal)
    (hOutcomeTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalOutcome populationOutcome outcomeThreshold candidate) ≤
        outcomeTail candidate)
    (hTreatmentTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalTreatment populationTreatment treatmentThreshold candidate) ≤
        treatmentTail candidate) :
    μ (uniformDeviationFailureEvent pool
        (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
        (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) ≤
      ∑ candidate ∈ pool, (outcomeTail candidate + treatmentTail candidate) := by
  have hOutcome := measure_uniformDeviationFailureEvent_le_sum
    μ pool empiricalOutcome populationOutcome outcomeThreshold outcomeTail hOutcomeTail
  have hTreatment := measure_uniformDeviationFailureEvent_le_sum
    μ pool empiricalTreatment populationTreatment treatmentThreshold treatmentTail hTreatmentTail
  calc
    μ (uniformDeviationFailureEvent pool
        (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
        (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) ≤
      μ (uniformDeviationFailureEvent pool empiricalOutcome populationOutcome outcomeThreshold ∪
        uniformDeviationFailureEvent pool empiricalTreatment populationTreatment
          treatmentThreshold) :=
        measure_mono (b2V3FusionUniformDeviationFailure_subset pool
          empiricalOutcome empiricalTreatment populationOutcome populationTreatment
          outcomeThreshold treatmentThreshold)
    _ ≤ μ (uniformDeviationFailureEvent pool empiricalOutcome populationOutcome outcomeThreshold) +
        μ (uniformDeviationFailureEvent pool empiricalTreatment populationTreatment
          treatmentThreshold) := measure_union_le _ _
    _ ≤ (∑ candidate ∈ pool, outcomeTail candidate) +
        ∑ candidate ∈ pool, treatmentTail candidate := add_le_add hOutcome hTreatment
    _ = ∑ candidate ∈ pool, (outcomeTail candidate + treatmentTail candidate) := by
      rw [Finset.sum_add_distrib]

/-- Exact pointwise weighted-Hoeffding budget for one common-coordinate view. -/
noncomputable def b2V3ViewTailBudget
    {Candidate : Type uCandidate} {numberOfCenters : ℕ}
    (coordinate : CommonRiskCoordinate) (component : RiskComponent)
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (lossEnvelope threshold : Candidate → ℝ) (candidate : Candidate) : ENNReal :=
  ENNReal.ofReal
    (2 * Real.exp
      (-(coordinate.scale component * threshold candidate) ^ 2 /
        (2 * (b2PointwiseHoeffdingProxy
          (weight candidate) (lossEnvelope candidate) : ℝ))))

/--
Weighted-Hoeffding instantiation for the frozen common coordinate.  X/Y may be dependent within a
center; each view only requires conditional independence of its center-block loss functions.
-/
theorem measure_b2V3FrozenFusionUniformDeviation_le_sum
    {Candidate : Type uCandidate} {numberOfCenters : ℕ}
    {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (pool : Finset Candidate) (coordinate : CommonRiskCoordinate)
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (outcomeEnvelope treatmentEnvelope outcomeThreshold treatmentThreshold : Candidate → ℝ)
    (hOutcomeIndep : ∀ candidate ∈ pool,
      ProbabilityTheory.iIndepFun (outcomeLoss candidate) μ)
    (hTreatmentIndep : ∀ candidate ∈ pool,
      ProbabilityTheory.iIndepFun (treatmentLoss candidate) μ)
    (hOutcomeMeasurable : ∀ candidate ∈ pool, ∀ center,
      AEMeasurable (outcomeLoss candidate center) μ)
    (hTreatmentMeasurable : ∀ candidate ∈ pool, ∀ center,
      AEMeasurable (treatmentLoss candidate center) μ)
    (hOutcomeEnvelope : ∀ candidate ∈ pool, ∀ center, ∀ᵐ sample ∂μ,
      outcomeLoss candidate center sample ∈ Set.Icc 0 (outcomeEnvelope candidate))
    (hTreatmentEnvelope : ∀ candidate ∈ pool, ∀ center, ∀ᵐ sample ∂μ,
      treatmentLoss candidate center sample ∈ Set.Icc 0 (treatmentEnvelope candidate))
    (hOutcomeThreshold : ∀ candidate ∈ pool, 0 ≤ outcomeThreshold candidate)
    (hTreatmentThreshold : ∀ candidate ∈ pool, 0 ≤ treatmentThreshold candidate) :
    μ (uniformDeviationFailureEvent pool
        (b2V3DirectionalEmpiricalFusionRisk coordinate outcomeWeight treatmentWeight
          outcomeLoss treatmentLoss)
        (b2V3DirectionalPopulationFusionRisk μ coordinate outcomeWeight treatmentWeight
          outcomeLoss treatmentLoss)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) ≤
      ∑ candidate ∈ pool,
        (b2V3ViewTailBudget coordinate .outcome outcomeWeight outcomeEnvelope
            outcomeThreshold candidate +
          b2V3ViewTailBudget coordinate .treatment treatmentWeight treatmentEnvelope
            treatmentThreshold candidate) := by
  apply measure_b2V3FusionUniformDeviation_le_sum
  · intro candidate hCandidate
    simpa [b2V3DirectionalEmpiricalFusionRisk,
      b2V3DirectionalPopulationFusionRisk,
      b2V3EmpiricalFusionFromViews, b2V3PopulationFusionFromViews,
      b2V3DirectionalEmpiricalViewRisk, b2V3DirectionalEmpiricalRawRisk,
      b2V3DirectionalPopulationViewRisk, b2V3DirectionalPopulationRawRisk,
      b2FrozenAffineEmpiricalRisk, b2FrozenAffinePopulationRisk,
      b2V3ViewTailBudget, deviationFailureEvent, CommonRiskCoordinate.transform,
      BiostatFormal.Probability.frozenAffineRisk] using
      (measure_b2FrozenAffinePointwiseDeviation_le μ
        (outcomeWeight candidate) (outcomeLoss candidate)
        (outcomeEnvelope candidate) (coordinate.location .outcome)
        (coordinate.scale .outcome) (outcomeThreshold candidate)
        (hOutcomeIndep candidate hCandidate)
        (hOutcomeMeasurable candidate hCandidate)
        (hOutcomeEnvelope candidate hCandidate)
        (coordinate.scale_pos .outcome)
        (hOutcomeThreshold candidate hCandidate))
  · intro candidate hCandidate
    simpa [b2V3DirectionalEmpiricalFusionRisk,
      b2V3DirectionalPopulationFusionRisk,
      b2V3EmpiricalFusionFromViews, b2V3PopulationFusionFromViews,
      b2V3DirectionalEmpiricalViewRisk, b2V3DirectionalEmpiricalRawRisk,
      b2V3DirectionalPopulationViewRisk, b2V3DirectionalPopulationRawRisk,
      b2FrozenAffineEmpiricalRisk, b2FrozenAffinePopulationRisk,
      b2V3ViewTailBudget, deviationFailureEvent, CommonRiskCoordinate.transform,
      BiostatFormal.Probability.frozenAffineRisk] using
      (measure_b2FrozenAffinePointwiseDeviation_le μ
        (treatmentWeight candidate) (treatmentLoss candidate)
        (treatmentEnvelope candidate) (coordinate.location .treatment)
        (coordinate.scale .treatment) (treatmentThreshold candidate)
        (hTreatmentIndep candidate hCandidate)
        (hTreatmentMeasurable candidate hCandidate)
        (hTreatmentEnvelope candidate hCandidate)
        (coordinate.scale_pos .treatment)
        (hTreatmentThreshold candidate hCandidate))

/-- Common-tail specialization for any frozen valid active pool of cardinality at most 60. -/
theorem measure_b2V3FusionUniformDeviation_le_60_mul
    {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample) (pool : Finset Candidate)
    (empiricalOutcome empiricalTreatment : Candidate → Sample → ℝ)
    (populationOutcome populationTreatment : Candidate → ℝ)
    (outcomeThreshold treatmentThreshold : Candidate → ℝ)
    (outcomeTail treatmentTail : ENNReal)
    (hCard : pool.card ≤ 60)
    (hOutcomeTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalOutcome populationOutcome outcomeThreshold candidate) ≤
        outcomeTail)
    (hTreatmentTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalTreatment populationTreatment treatmentThreshold candidate) ≤
        treatmentTail) :
    μ (uniformDeviationFailureEvent pool
        (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
        (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) ≤
      60 * (outcomeTail + treatmentTail) := by
  have hBound := measure_b2V3FusionUniformDeviation_le_sum
    μ pool empiricalOutcome empiricalTreatment populationOutcome populationTreatment
      outcomeThreshold treatmentThreshold (fun _ ↦ outcomeTail) (fun _ ↦ treatmentTail)
      hOutcomeTail hTreatmentTail
  calc
    μ (uniformDeviationFailureEvent pool
        (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
        (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) ≤
      ∑ candidate ∈ pool, (outcomeTail + treatmentTail) := hBound
    _ = pool.card * (outcomeTail + treatmentTail) := by simp [mul_add]
    _ ≤ 60 * (outcomeTail + treatmentTail) := by
      gcongr
      exact_mod_cast hCard

/-- The common-tail bound on the actual preregistered 60-index B2-v3 envelope. -/
theorem measure_b2V3CandidateRecordEnvelopeFusionDeviation_le_60_mul
    {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample)
    (empiricalOutcome empiricalTreatment : B2CandidateRecordIndex → Sample → ℝ)
    (populationOutcome populationTreatment : B2CandidateRecordIndex → ℝ)
    (outcomeThreshold treatmentThreshold : B2CandidateRecordIndex → ℝ)
    (outcomeTail treatmentTail : ENNReal)
    (hOutcomeTail : ∀ candidate ∈ b2V3CandidateRecordEnvelope,
      μ (deviationFailureEvent empiricalOutcome populationOutcome outcomeThreshold candidate) ≤
        outcomeTail)
    (hTreatmentTail : ∀ candidate ∈ b2V3CandidateRecordEnvelope,
      μ (deviationFailureEvent empiricalTreatment populationTreatment treatmentThreshold candidate) ≤
        treatmentTail) :
    μ (uniformDeviationFailureEvent b2V3CandidateRecordEnvelope
        (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
        (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
        (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) ≤
      60 * (outcomeTail + treatmentTail) := by
  apply measure_b2V3FusionUniformDeviation_le_60_mul
  · exact b2V3CandidateRecordEnvelope_card.le
  · exact hOutcomeTail
  · exact hTreatmentTail

/--
Outside the uniform failure event, every member of the frozen pool satisfies its candidate-specific
fusion radius.  This is the deterministic bridge from the A2 event to the selector hypothesis.
-/
theorem b2V3FusionDeviation_le_of_not_mem_uniformFailure
    {Candidate : Type uCandidate} {Sample : Type uSample} [DecidableEq Candidate]
    (pool : Finset Candidate)
    (empiricalOutcome empiricalTreatment : Candidate → Sample → ℝ)
    (populationOutcome populationTreatment : Candidate → ℝ)
    (outcomeThreshold treatmentThreshold : Candidate → ℝ)
    (sample : Sample)
    (hGood : sample ∉ uniformDeviationFailureEvent pool
      (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
      (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
      (b2V3FusionThreshold outcomeThreshold treatmentThreshold))
    (candidate : GuardedPoolMember pool) :
    |b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment candidate.1 sample -
        b2V3PopulationFusionFromViews populationOutcome populationTreatment candidate.1| ≤
      b2V3FusionThreshold outcomeThreshold treatmentThreshold candidate.1 := by
  by_contra hDeviation
  apply hGood
  apply Set.mem_iUnion.2
  refine ⟨candidate.1, ?_⟩
  apply Set.mem_iUnion.2
  exact ⟨candidate.2, lt_of_not_ge hDeviation⟩

/-- A registered common radius dominating all candidate-specific radii controls the frozen pool. -/
theorem b2V3FusionDeviation_le_common_of_not_mem_uniformFailure
    {Candidate : Type uCandidate} {Sample : Type uSample} [DecidableEq Candidate]
    (pool : Finset Candidate)
    (empiricalOutcome empiricalTreatment : Candidate → Sample → ℝ)
    (populationOutcome populationTreatment : Candidate → ℝ)
    (outcomeThreshold treatmentThreshold : Candidate → ℝ)
    (sample : Sample) (u : ℝ)
    (hThreshold : ∀ candidate ∈ pool,
      b2V3FusionThreshold outcomeThreshold treatmentThreshold candidate ≤ u)
    (hGood : sample ∉ uniformDeviationFailureEvent pool
      (b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment)
      (b2V3PopulationFusionFromViews populationOutcome populationTreatment)
      (b2V3FusionThreshold outcomeThreshold treatmentThreshold)) :
    ∀ candidate : GuardedPoolMember pool,
      |b2V3EmpiricalFusionFromViews empiricalOutcome empiricalTreatment candidate.1 sample -
          b2V3PopulationFusionFromViews populationOutcome populationTreatment candidate.1| ≤ u := by
  intro candidate
  exact (b2V3FusionDeviation_le_of_not_mem_uniformFailure pool
    empiricalOutcome empiricalTreatment populationOutcome populationTreatment
    outcomeThreshold treatmentThreshold sample hGood candidate).trans
      (hThreshold candidate.1 candidate.2)

/--
Selector stitch for the B2-v3 primary risk.  Compatibility, uniform deviation, and empirical
near-minimization all mention the very same equal-weight Y/X fusion-risk functions.  Hence the
statement cannot be instantiated with one population estimand in compatibility and another in A2.

The finite pool is already frozen in the theorem's context.  The conclusion concerns selection
among active guarded records only; pure F diagnostics and stability do not occur in the signature.
-/
theorem b2V3_exact_of_restrictedCompatibility_A2
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (truth : Partition Center Label)
    (zGood zSelected : GuardedPoolMember pool)
    (outcomePopulation treatmentPopulation :
      CandidateRecord Center Label Slot Context → ℝ)
    (outcomeEmpirical treatmentEmpirical :
      CandidateRecord Center Label Slot Context → ℝ)
    (κ u ηTie ηGuard ηNum : ℝ)
    (hKappa : 0 ≤ κ)
    (hExactWitness : zGood.1 ∈ exactCandidateSet truth)
    (hCompatibility : ∀ z : GuardedPoolMember pool,
      z.1 ∈ badCandidateSet truth →
        κ * labelDistance z.1.label truth ≤
          b2V3FusionRisk outcomePopulation treatmentPopulation z.1 -
            b2V3FusionRisk outcomePopulation treatmentPopulation zGood.1)
    (hDeviation : ∀ z : GuardedPoolMember pool,
      |b2V3FusionRisk outcomeEmpirical treatmentEmpirical z.1 -
        b2V3FusionRisk outcomePopulation treatmentPopulation z.1| ≤ u)
    (hSeparation :
      2 * u + selectionTolerance ηTie ηGuard ηNum <
        κ / (Fintype.card Center : ℝ))
    (hSelected :
      b2V3FusionRisk outcomeEmpirical treatmentEmpirical zSelected.1 ≤
        b2V3FusionRisk outcomeEmpirical treatmentEmpirical zGood.1 +
          selectionTolerance ηTie ηGuard ηNum) :
    zSelected.1 ∈ exactCandidateSet truth := by
  apply BiostatFormal.HighDimensional.exact_of_uniform_deviation_and_margin
    (fun z : GuardedPoolMember pool => z.1 ∈ exactCandidateSet truth)
    (fun z : GuardedPoolMember pool =>
      b2V3FusionRisk outcomePopulation treatmentPopulation z.1)
    (fun z : GuardedPoolMember pool =>
      b2V3FusionRisk outcomeEmpirical treatmentEmpirical z.1)
    zGood zSelected u (selectionTolerance ηTie ηGuard ηNum)
    hExactWitness hDeviation
  · intro z _ hNotExact
    have hBadPartition : ¬ Exact z.1.label truth := by
      exact hNotExact
    have hDistance := one_div_card_le_labelDistance_of_bad hBadPartition
    have hScaled :
        κ / (Fintype.card Center : ℝ) ≤ κ * labelDistance z.1.label truth := by
      calc
        κ / (Fintype.card Center : ℝ) =
            κ * (1 / (Fintype.card Center : ℝ)) := by ring
        _ ≤ κ * labelDistance z.1.label truth :=
          mul_le_mul_of_nonneg_left hDistance hKappa
    have hRiskGap := hCompatibility z hNotExact
    linarith
  · exact hSelected

/--
Source-facing active-slot selector.  The candidate type is fixed to `B2V3ActiveSlot`, and both
population and empirical scores are constructed directly from the frozen Y/X directional risks.
The A2 good event supplies the common-`u` deviation premise; pure F and stability cannot be passed
as selector components in this interface.
-/
theorem b2V3_exact_of_frozenDirectionalRisk_A2Event
    {Center : Type uCenter} {Label : Type uLabel} {Context : Type uContext}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [Fintype Label] [DecidableEq Label]
    [DecidableEq (CandidateRecord Center Label B2V3ActiveSlot Context)]
    {numberOfCenters : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (coordinate : CommonRiskCoordinate)
    (pool : GuardedCandidatePool
      (CandidateRecord Center Label B2V3ActiveSlot Context))
    (truth : Partition Center Label)
    (zGood zSelected : GuardedPoolMember pool)
    (outcomeWeight treatmentWeight :
      CandidateRecord Center Label B2V3ActiveSlot Context → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss :
      CandidateRecord Center Label B2V3ActiveSlot Context →
        Fin numberOfCenters → Sample → ℝ)
    (outcomeThreshold treatmentThreshold :
      CandidateRecord Center Label B2V3ActiveSlot Context → ℝ)
    (sample : Sample) (κ u ηTie ηGuard ηNum : ℝ)
    (hKappa : 0 ≤ κ)
    (hExactWitness : zGood.1 ∈ exactCandidateSet truth)
    (hCompatibility : ∀ z : GuardedPoolMember pool,
      z.1 ∈ badCandidateSet truth →
        κ * labelDistance z.1.label truth ≤
          b2V3DirectionalPopulationFusionRisk μ coordinate outcomeWeight treatmentWeight
              outcomeLoss treatmentLoss z.1 -
            b2V3DirectionalPopulationFusionRisk μ coordinate outcomeWeight treatmentWeight
              outcomeLoss treatmentLoss zGood.1)
    (hThreshold : ∀ record ∈ pool,
      b2V3FusionThreshold outcomeThreshold treatmentThreshold record ≤ u)
    (hA2Good : sample ∉ uniformDeviationFailureEvent pool
      (b2V3DirectionalEmpiricalFusionRisk coordinate outcomeWeight treatmentWeight
        outcomeLoss treatmentLoss)
      (b2V3DirectionalPopulationFusionRisk μ coordinate outcomeWeight treatmentWeight
        outcomeLoss treatmentLoss)
      (b2V3FusionThreshold outcomeThreshold treatmentThreshold))
    (hSeparation :
      2 * u + selectionTolerance ηTie ηGuard ηNum <
        κ / (Fintype.card Center : ℝ))
    (hSelected :
      b2V3DirectionalEmpiricalFusionRisk coordinate outcomeWeight treatmentWeight
          outcomeLoss treatmentLoss zSelected.1 sample ≤
        b2V3DirectionalEmpiricalFusionRisk coordinate outcomeWeight treatmentWeight
            outcomeLoss treatmentLoss zGood.1 sample +
          selectionTolerance ηTie ηGuard ηNum) :
    zSelected.1 ∈ exactCandidateSet truth := by
  apply b2V3_exact_of_restrictedCompatibility_A2 pool truth zGood zSelected
    (b2V3DirectionalPopulationViewRisk μ coordinate .outcome outcomeWeight outcomeLoss)
    (b2V3DirectionalPopulationViewRisk μ coordinate .treatment treatmentWeight treatmentLoss)
    (fun record => b2V3DirectionalEmpiricalViewRisk coordinate .outcome
      outcomeWeight outcomeLoss record sample)
    (fun record => b2V3DirectionalEmpiricalViewRisk coordinate .treatment
      treatmentWeight treatmentLoss record sample)
    κ u ηTie ηGuard ηNum hKappa hExactWitness
  · intro z hBad
    simpa [b2V3DirectionalPopulationFusionRisk, b2V3FusionRisk] using
      hCompatibility z hBad
  · have hDeviation := b2V3FusionDeviation_le_common_of_not_mem_uniformFailure
      pool
      (b2V3DirectionalEmpiricalViewRisk coordinate .outcome outcomeWeight outcomeLoss)
      (b2V3DirectionalEmpiricalViewRisk coordinate .treatment treatmentWeight treatmentLoss)
      (b2V3DirectionalPopulationViewRisk μ coordinate .outcome outcomeWeight outcomeLoss)
      (b2V3DirectionalPopulationViewRisk μ coordinate .treatment treatmentWeight treatmentLoss)
      outcomeThreshold treatmentThreshold sample u hThreshold
      hA2Good
    intro z
    simpa [b2V3FusionRisk, b2V3EmpiricalFusionFromViews,
      b2V3PopulationFusionFromViews] using hDeviation z
  · exact hSeparation
  · simpa [b2V3DirectionalEmpiricalFusionRisk, b2V3FusionRisk] using hSelected

end

end BiostatFormal.HighDimensional.Multiview
