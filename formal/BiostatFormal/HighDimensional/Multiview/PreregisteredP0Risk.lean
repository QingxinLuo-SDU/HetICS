import BiostatFormal.HighDimensional.Multiview.Guards
import BiostatFormal.HighDimensional.Multiview.Risks
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Preregistered P0 common risk coordinate

This file freezes the primary risk semantics for the revised preregistered P0 estimator:

* one affine component-risk coordinate is shared by all candidates and both directions;
* Y and X receive primary identification weights `1/2` and `1/2`;
* capped F and evaluation-only frozen-predictor agreement remain available as auxiliary component
  reports but have zero primary-selection weight;
* there is no post-aggregation clipping in the common coordinate.

The file is deterministic.  Independence and concentration are supplied by
`BiostatFormal.Probability.WeightedHoeffding`; analytic B2 pair separation and the oracle-to-record
bridge remain later obligations.
-/

namespace BiostatFormal.HighDimensional.Multiview

noncomputable section

universe uCandidate uCenter uLabel uSlot uContext uCoordinate

/-- One externally frozen affine coordinate shared by every record and direction. -/
structure CommonRiskCoordinate where
  location : RiskComponent → ℝ
  scale : RiskComponent → ℝ
  scale_pos : ∀ component, 0 < scale component

/-- Apply the shared affine coordinate to one raw component risk. -/
def CommonRiskCoordinate.transform
    (coordinate : CommonRiskCoordinate) (component : RiskComponent) (rawRisk : ℝ) : ℝ :=
  (rawRisk - coordinate.location component) / coordinate.scale component

/-- Primary preregistered P0 weights: equal Y/X identification and no F/stability selection role. -/
def preregisteredP0IdentificationWeight : RiskComponent → ℝ
  | .outcome => 1 / 2
  | .treatment => 1 / 2
  | .covariate => 0
  | .stability => 0

@[simp] theorem preregisteredP0IdentificationWeight_outcome :
    preregisteredP0IdentificationWeight .outcome = 1 / 2 := rfl

@[simp] theorem preregisteredP0IdentificationWeight_treatment :
    preregisteredP0IdentificationWeight .treatment = 1 / 2 := rfl

@[simp] theorem preregisteredP0IdentificationWeight_covariate :
    preregisteredP0IdentificationWeight .covariate = 0 := rfl

@[simp] theorem preregisteredP0IdentificationWeight_stability :
    preregisteredP0IdentificationWeight .stability = 0 := rfl

theorem preregisteredP0IdentificationWeight_nonneg
    (component : RiskComponent) :
    0 ≤ preregisteredP0IdentificationWeight component := by
  cases component <;> norm_num [preregisteredP0IdentificationWeight]

private theorem riskComponent_univ :
    (Finset.univ : Finset RiskComponent) =
      {.outcome, .treatment, .covariate, .stability} := by
  ext component
  cases component <;> simp

theorem sum_preregisteredP0IdentificationWeight :
    ∑ component : RiskComponent, preregisteredP0IdentificationWeight component = 1 := by
  rw [riskComponent_univ]
  simp [preregisteredP0IdentificationWeight, Finset.sum_insert]
  norm_num

/--
Build the complete empirical/population component pair in one common coordinate.  F and stability
components remain observable in the pair even though their primary weights are zero.
-/
def preregisteredP0RiskPair
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (coordinate : CommonRiskCoordinate)
    (populationRaw empiricalRaw : RiskComponent → Candidate → ℝ) :
    CompositeRiskPair Candidate where
  weight := preregisteredP0IdentificationWeight
  populationComponent := fun component candidate =>
    coordinate.transform component (populationRaw component candidate)
  empiricalComponent := fun component candidate =>
    coordinate.transform component (empiricalRaw component candidate)

/-- Auxiliary F and stability values are reported as a pair and are not collapsed into selection. -/
def preregisteredP0AuxiliaryReport
    {Candidate : Type uCandidate}
    (componentRisk : RiskComponent → Candidate → ℝ) (candidate : Candidate) : ℝ × ℝ :=
  (componentRisk .covariate candidate, componentRisk .stability candidate)

/-- The primary population risk is exactly the mean of common-coordinate Y and X risks. -/
theorem populationCompositeRisk_preregisteredP0RiskPair_eq
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (coordinate : CommonRiskCoordinate)
    (populationRaw empiricalRaw : RiskComponent → Candidate → ℝ)
    (candidate : Candidate) :
    populationCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw empiricalRaw) candidate =
      (coordinate.transform .outcome (populationRaw .outcome candidate) +
        coordinate.transform .treatment (populationRaw .treatment candidate)) / 2 := by
  classical
  unfold populationCompositeRisk preregisteredP0RiskPair
  rw [riskComponent_univ]
  simp [Finset.sum_insert, preregisteredP0IdentificationWeight]
  ring

/-- The primary empirical risk uses the identical mean of common-coordinate Y and X risks. -/
theorem empiricalCompositeRisk_preregisteredP0RiskPair_eq
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (coordinate : CommonRiskCoordinate)
    (populationRaw empiricalRaw : RiskComponent → Candidate → ℝ)
    (candidate : Candidate) :
    empiricalCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw empiricalRaw) candidate =
      (coordinate.transform .outcome (empiricalRaw .outcome candidate) +
        coordinate.transform .treatment (empiricalRaw .treatment candidate)) / 2 := by
  classical
  unfold empiricalCompositeRisk preregisteredP0RiskPair
  rw [riskComponent_univ]
  simp [Finset.sum_insert, preregisteredP0IdentificationWeight]
  ring

/-- Arbitrary changes to population F/stability components cannot alter the primary risk. -/
theorem populationCompositeRisk_preregisteredP0_invariant_of_primary_eq
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (coordinate : CommonRiskCoordinate)
    (populationRaw₁ populationRaw₂ empiricalRaw : RiskComponent → Candidate → ℝ)
    (candidate : Candidate)
    (hOutcome : populationRaw₁ .outcome candidate = populationRaw₂ .outcome candidate)
    (hTreatment : populationRaw₁ .treatment candidate = populationRaw₂ .treatment candidate) :
    populationCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw₁ empiricalRaw) candidate =
      populationCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw₂ empiricalRaw) candidate := by
  rw [populationCompositeRisk_preregisteredP0RiskPair_eq,
    populationCompositeRisk_preregisteredP0RiskPair_eq, hOutcome, hTreatment]

/-- Arbitrary changes to empirical F/stability components cannot alter the primary risk. -/
theorem empiricalCompositeRisk_preregisteredP0_invariant_of_primary_eq
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (coordinate : CommonRiskCoordinate)
    (populationRaw empiricalRaw₁ empiricalRaw₂ : RiskComponent → Candidate → ℝ)
    (candidate : Candidate)
    (hOutcome : empiricalRaw₁ .outcome candidate = empiricalRaw₂ .outcome candidate)
    (hTreatment : empiricalRaw₁ .treatment candidate = empiricalRaw₂ .treatment candidate) :
    empiricalCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw empiricalRaw₁) candidate =
      empiricalCompositeRisk
        (preregisteredP0RiskPair coordinate populationRaw empiricalRaw₂) candidate := by
  rw [empiricalCompositeRisk_preregisteredP0RiskPair_eq,
    empiricalCompositeRisk_preregisteredP0RiskPair_eq, hOutcome, hTreatment]

/--
Only Y/X deviations enter the preregistered primary-risk threshold.  No F or stability deviation
hypothesis is present.
-/
theorem abs_preregisteredP0CompositeRisk_sub_le
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (coordinate : CommonRiskCoordinate)
    (populationRaw empiricalRaw : RiskComponent → Candidate → ℝ)
    (candidate : Candidate) (outcomeThreshold treatmentThreshold : ℝ)
    (hOutcome :
      |coordinate.transform .outcome (empiricalRaw .outcome candidate) -
        coordinate.transform .outcome (populationRaw .outcome candidate)| ≤ outcomeThreshold)
    (hTreatment :
      |coordinate.transform .treatment (empiricalRaw .treatment candidate) -
        coordinate.transform .treatment (populationRaw .treatment candidate)| ≤ treatmentThreshold) :
    |empiricalCompositeRisk
          (preregisteredP0RiskPair coordinate populationRaw empiricalRaw) candidate -
        populationCompositeRisk
          (preregisteredP0RiskPair coordinate populationRaw empiricalRaw) candidate| ≤
      (outcomeThreshold + treatmentThreshold) / 2 := by
  rw [empiricalCompositeRisk_preregisteredP0RiskPair_eq,
    populationCompositeRisk_preregisteredP0RiskPair_eq]
  calc
    |(coordinate.transform .outcome (empiricalRaw .outcome candidate) +
          coordinate.transform .treatment (empiricalRaw .treatment candidate)) / 2 -
        (coordinate.transform .outcome (populationRaw .outcome candidate) +
          coordinate.transform .treatment (populationRaw .treatment candidate)) / 2| =
      |(coordinate.transform .outcome (empiricalRaw .outcome candidate) -
          coordinate.transform .outcome (populationRaw .outcome candidate)) +
        (coordinate.transform .treatment (empiricalRaw .treatment candidate) -
          coordinate.transform .treatment (populationRaw .treatment candidate))| / 2 := by
        rw [← abs_of_nonneg (show (0 : ℝ) ≤ 2 by norm_num), ← abs_div]
        congr 1
        ring
    _ ≤
      (|coordinate.transform .outcome (empiricalRaw .outcome candidate) -
          coordinate.transform .outcome (populationRaw .outcome candidate)| +
        |coordinate.transform .treatment (empiricalRaw .treatment candidate) -
          coordinate.transform .treatment (populationRaw .treatment candidate)|) / 2 := by
        exact div_le_div_of_nonneg_right (abs_add_le _ _) (by norm_num)
    _ ≤ (outcomeThreshold + treatmentThreshold) / 2 := by
      exact div_le_div_of_nonneg_right (add_le_add hOutcome hTreatment) (by norm_num)

/--
All records may retain direction-specific fitted contexts while sharing one projected risk
coordinate.  This is the correct compatibility requirement; equality of full fit contexts is not.
-/
def guardedPoolHasCommonRiskCoordinate
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    (pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context))
    (coordinateOf : Context → CommonRiskCoordinate)
    (commonCoordinate : CommonRiskCoordinate) : Prop :=
  ∀ record ∈ pool, coordinateOf record.context = commonCoordinate

end

end BiostatFormal.HighDimensional.Multiview
