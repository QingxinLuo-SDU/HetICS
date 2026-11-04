import BiostatFormal.HighDimensional.Multiview.B2RiskDeviation
import BiostatFormal.HighDimensional.Multiview.B2V3Slots
import BiostatFormal.HighDimensional.Multiview.PreregisteredP0Risk

/-!
# Conditional directional risk for B2-v3

All definitions in this file are evaluated after fixing one training-half realization, the fitted
directional predictors, the active candidate pool, and the external common risk coordinate.  The
remaining measure is the independent evaluation law.  Raw weighted expectations are transformed
exactly once.

The active B2-v3 universe has 15 slots and two stages in each direction, hence 30 record indices per
direction and 60 after the AB/BA union already encoded by `b2V3CandidateRecordEnvelope`.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

noncomputable section

universe uCandidate uSample

/-- One fixed direction contains all 15 active slots at the pre and post stages. -/
abbrev B2V3DirectionRecordIndex := B2V3ActiveSlot × CandidateStage

/-- The preregistered evaluation-independent index envelope in one direction. -/
def b2V3DirectionEnvelope : Finset B2V3DirectionRecordIndex := Finset.univ

theorem b2V3DirectionEnvelope_card : b2V3DirectionEnvelope.card = 30 := by
  decide

/-- Raw empirical weighted center risk after the training context has been frozen. -/
def b2V3DirectionalEmpiricalRawRisk
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (loss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) (sample : Sample) : ℝ :=
  BiostatFormal.Probability.weightedCenterRisk Finset.univ
    (weight candidate) (loss candidate) sample

/-- Raw conditional population risk: center expectations under the fixed evaluation law. -/
noncomputable def b2V3DirectionalPopulationRawRisk
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (loss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) : ℝ :=
  BiostatFormal.Probability.weightedCenterPopulationRisk μ Finset.univ
    (weight candidate) (loss candidate)

/-- Apply the one common coordinate to one raw empirical X or Y risk. -/
def b2V3DirectionalEmpiricalViewRisk
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    (coordinate : CommonRiskCoordinate) (component : RiskComponent)
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (loss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) (sample : Sample) : ℝ :=
  coordinate.transform component
    (b2V3DirectionalEmpiricalRawRisk weight loss candidate sample)

/-- Apply the identical common coordinate to the matching conditional population expectation. -/
noncomputable def b2V3DirectionalPopulationViewRisk
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample) (coordinate : CommonRiskCoordinate) (component : RiskComponent)
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (loss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) : ℝ :=
  coordinate.transform component
    (b2V3DirectionalPopulationRawRisk μ weight loss candidate)

/--
The conditional view risk is definitionally the existing frozen-affine risk.  This theorem records
that the raw expectation is normalized once, rather than passed through a second transform.
-/
theorem b2V3DirectionalPopulationViewRisk_eq_frozenAffine
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample) (coordinate : CommonRiskCoordinate) (component : RiskComponent)
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (loss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) :
    b2V3DirectionalPopulationViewRisk μ coordinate component weight loss candidate =
      b2FrozenAffinePopulationRisk μ (weight candidate) (loss candidate)
        (coordinate.location component) (coordinate.scale component) := by
  rfl

/-- Equal-weight Y/X fusion empirical risk.  F and stability are absent from its signature. -/
def b2V3DirectionalEmpiricalFusionRisk
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    (coordinate : CommonRiskCoordinate)
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) (sample : Sample) : ℝ :=
  (b2V3DirectionalEmpiricalViewRisk coordinate .outcome outcomeWeight outcomeLoss
      candidate sample +
    b2V3DirectionalEmpiricalViewRisk coordinate .treatment treatmentWeight treatmentLoss
      candidate sample) / 2

/-- Equal-weight Y/X fusion conditional population risk. -/
noncomputable def b2V3DirectionalPopulationFusionRisk
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample) (coordinate : CommonRiskCoordinate)
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (candidate : Candidate) : ℝ :=
  (b2V3DirectionalPopulationViewRisk μ coordinate .outcome outcomeWeight outcomeLoss candidate +
    b2V3DirectionalPopulationViewRisk μ coordinate .treatment treatmentWeight treatmentLoss
      candidate) / 2

/-- Package the raw conditional Y/X expectations with arbitrary auxiliary diagnostic values. -/
noncomputable def b2V3DirectionalPopulationRawComponents
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (auxiliaryRaw : RiskComponent → Candidate → ℝ) :
    RiskComponent → Candidate → ℝ
  | .outcome, candidate =>
      b2V3DirectionalPopulationRawRisk μ outcomeWeight outcomeLoss candidate
  | .treatment, candidate =>
      b2V3DirectionalPopulationRawRisk μ treatmentWeight treatmentLoss candidate
  | .covariate, candidate => auxiliaryRaw .covariate candidate
  | .stability, candidate => auxiliaryRaw .stability candidate

/-- Package the matching raw empirical Y/X risks with arbitrary auxiliary diagnostic values. -/
def b2V3DirectionalEmpiricalRawComponents
    {Candidate : Type uCandidate} {numberOfCenters : ℕ} {Sample : Type uSample}
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (auxiliaryRaw : RiskComponent → Candidate → Sample → ℝ) :
    RiskComponent → Candidate → Sample → ℝ
  | .outcome, candidate, sample =>
      b2V3DirectionalEmpiricalRawRisk outcomeWeight outcomeLoss candidate sample
  | .treatment, candidate, sample =>
      b2V3DirectionalEmpiricalRawRisk treatmentWeight treatmentLoss candidate sample
  | .covariate, candidate, sample => auxiliaryRaw .covariate candidate sample
  | .stability, candidate, sample => auxiliaryRaw .stability candidate sample

/--
The source-facing preregistered population risk is exactly the directional conditional fusion risk
when its raw Y/X inputs are the weighted conditional expectations above.  This equality prevents
both a population-estimand mismatch and accidental double normalization.  Auxiliary F/stability
raw entries are arbitrary because their primary weights are zero.
-/
theorem populationCompositeRisk_preregisteredP0RiskPair_eq_b2V3DirectionalFusion
    {Candidate : Type uCandidate} [RiskInput Candidate]
    {numberOfCenters : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (coordinate : CommonRiskCoordinate)
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (populationAuxiliaryRaw : RiskComponent → Candidate → ℝ)
    (empiricalRaw : RiskComponent → Candidate → ℝ)
    (candidate : Candidate) :
    populationCompositeRisk
        (preregisteredP0RiskPair coordinate
          (b2V3DirectionalPopulationRawComponents μ outcomeWeight treatmentWeight
            outcomeLoss treatmentLoss populationAuxiliaryRaw)
          empiricalRaw) candidate =
      b2V3DirectionalPopulationFusionRisk μ coordinate outcomeWeight treatmentWeight
        outcomeLoss treatmentLoss candidate := by
  rw [populationCompositeRisk_preregisteredP0RiskPair_eq]
  rfl

/-- The matching preregistered empirical risk is the same directional Y/X fusion score. -/
theorem empiricalCompositeRisk_preregisteredP0RiskPair_eq_b2V3DirectionalFusion
    {Candidate : Type uCandidate} [RiskInput Candidate]
    {numberOfCenters : ℕ} {Sample : Type uSample}
    (coordinate : CommonRiskCoordinate)
    (outcomeWeight treatmentWeight : Candidate → Fin numberOfCenters → ℝ)
    (outcomeLoss treatmentLoss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (populationRaw : RiskComponent → Candidate → ℝ)
    (empiricalAuxiliaryRaw : RiskComponent → Candidate → Sample → ℝ)
    (candidate : Candidate) (sample : Sample) :
    empiricalCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw
          (fun component candidate =>
            b2V3DirectionalEmpiricalRawComponents outcomeWeight treatmentWeight
              outcomeLoss treatmentLoss empiricalAuxiliaryRaw component candidate sample)) candidate =
      b2V3DirectionalEmpiricalFusionRisk coordinate outcomeWeight treatmentWeight
        outcomeLoss treatmentLoss candidate sample := by
  rw [empiricalCompositeRisk_preregisteredP0RiskPair_eq]
  rfl

end

end BiostatFormal.HighDimensional.Multiview
