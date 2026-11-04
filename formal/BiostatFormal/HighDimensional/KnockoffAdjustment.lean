import Mathlib

namespace BiostatFormal.HighDimensional.KnockoffAdjustment

open Filter MeasureTheory Topology

universe uOmega uVariable

variable {Omega : Type uOmega} {Variable : Type uVariable}

def BaselineOnly [DecidableEq Variable]
    (baseline selected : Finset Variable) : Prop :=
  selected ⊆ baseline

/-- A candidate pool is drawn entirely from the declared baseline universe. -/
def CandidatePoolBaseline [DecidableEq Variable]
    (baseline candidates : Finset Variable) : Prop :=
  candidates ⊆ baseline

/-- Any subset selected from a baseline-only candidate pool remains baseline-only. -/
theorem baselineOnly_of_selected_subset_candidates [DecidableEq Variable]
    {baseline candidates selected : Finset Variable}
    (hCandidates : CandidatePoolBaseline baseline candidates)
    (hSelected : selected ⊆ candidates) :
    BaselineOnly baseline selected := by
  exact hSelected.trans hCandidates

def dualBranchSelection [DecidableEq Variable]
    (treatmentSelected outcomeSelected : Finset Variable) : Finset Variable :=
  treatmentSelected ∪ outcomeSelected

def SureScreens [DecidableEq Variable]
    (core selected : Finset Variable) : Prop :=
  core ⊆ selected

def CompleteAdjustment [DecidableEq Variable]
    (adjustmentCore selected : Finset Variable) : Prop :=
  adjustmentCore ⊆ selected

def SafeFamilyValid [DecidableEq Variable]
    (safeFamily : Finset Variable → Prop) (selected : Finset Variable) : Prop :=
  safeFamily selected

def ValidAdjustment [DecidableEq Variable]
    (adjustmentCore : Finset Variable) (safeFamily : Finset Variable → Prop)
    (selected : Finset Variable) : Prop :=
  CompleteAdjustment adjustmentCore selected ∧ SafeFamilyValid safeFamily selected

def FDRControlled (expectedFDP q : ℝ) : Prop := expectedFDP ≤ q

theorem baselineOnly_dualBranchSelection [DecidableEq Variable]
    {baseline treatmentSelected outcomeSelected : Finset Variable}
    (hTreatment : BaselineOnly baseline treatmentSelected)
    (hOutcome : BaselineOnly baseline outcomeSelected) :
    BaselineOnly baseline
      (dualBranchSelection treatmentSelected outcomeSelected) := by
  intro v hVariable
  simp only [BaselineOnly] at hTreatment hOutcome ⊢
  simp only [dualBranchSelection, Finset.mem_union] at hVariable
  rcases hVariable with hVariable | hVariable
  · exact hTreatment hVariable
  · exact hOutcome hVariable

theorem sureScreens_dualBranchSelection [DecidableEq Variable]
    {treatmentCore outcomeCore treatmentSelected outcomeSelected : Finset Variable}
    (hTreatment : SureScreens treatmentCore treatmentSelected)
    (hOutcome : SureScreens outcomeCore outcomeSelected) :
    SureScreens (treatmentCore ∪ outcomeCore)
      (dualBranchSelection treatmentSelected outcomeSelected) := by
  intro v hVariable
  simp only [SureScreens] at hTreatment hOutcome ⊢
  simp only [dualBranchSelection, Finset.mem_union] at hVariable ⊢
  rcases hVariable with hVariable | hVariable
  · exact Or.inl (hTreatment hVariable)
  · exact Or.inr (hOutcome hVariable)

theorem adjustmentCore_complete_of_branchwise_sureScreening
    [DecidableEq Variable]
    {adjustmentCore treatmentCore outcomeCore treatmentSelected outcomeSelected :
      Finset Variable}
    (hCore : adjustmentCore ⊆ treatmentCore ∪ outcomeCore)
    (hTreatment : SureScreens treatmentCore treatmentSelected)
    (hOutcome : SureScreens outcomeCore outcomeSelected) :
    CompleteAdjustment adjustmentCore
      (dualBranchSelection treatmentSelected outcomeSelected) := by
  exact hCore.trans (sureScreens_dualBranchSelection hTreatment hOutcome)

theorem validAdjustment_of_complete_of_safe [DecidableEq Variable]
    {adjustmentCore selected : Finset Variable}
    {safeFamily : Finset Variable → Prop}
    (hComplete : CompleteAdjustment adjustmentCore selected)
    (hSafe : SafeFamilyValid safeFamily selected) :
    ValidAdjustment adjustmentCore safeFamily selected :=
  ⟨hComplete, hSafe⟩

def variableSelectedEvent [DecidableEq Variable]
    (selected : Omega → Finset Variable) (v : Variable) : Set Omega :=
  {omega | v ∈ selected omega}

def variableMissEvent [DecidableEq Variable]
    (selected : Omega → Finset Variable) (v : Variable) : Set Omega :=
  {omega | v ∉ selected omega}

def sureScreeningEvent [DecidableEq Variable]
    (core : Finset Variable) (selected : Omega → Finset Variable) : Set Omega :=
  {omega | SureScreens core (selected omega)}

theorem sureScreeningEvent_compl_eq_iUnion_miss
    [DecidableEq Variable]
    (core : Finset Variable) (selected : Omega → Finset Variable) :
    (sureScreeningEvent core selected)ᶜ =
      ⋃ v ∈ core, variableMissEvent selected v := by
  classical
  ext omega
  simp [sureScreeningEvent, SureScreens, variableMissEvent, Finset.subset_iff]

theorem measure_sureScreeningFailure_le_sum_miss
    [MeasurableSpace Omega] [DecidableEq Variable]
    (mu : Measure Omega) (core : Finset Variable)
    (selected : Omega → Finset Variable) :
    mu (sureScreeningEvent core selected)ᶜ ≤
      ∑ v ∈ core, mu (variableMissEvent selected v) := by
  rw [sureScreeningEvent_compl_eq_iUnion_miss]
  exact measure_biUnion_finset_le core (variableMissEvent selected)

noncomputable def averagePower [DecidableEq Variable]
    (core : Finset Variable) (selectionProbability : Variable → ℝ) : ℝ :=
  (∑ v ∈ core, selectionProbability v) / core.card

theorem sum_one_sub_selectionProbability_eq_card_mul_one_sub_averagePower
    [DecidableEq Variable]
    (core : Finset Variable) (selectionProbability : Variable → ℝ)
    (hCore : core.Nonempty) :
    (∑ v ∈ core, (1 - selectionProbability v)) =
      core.card * (1 - averagePower core selectionProbability) := by
  rw [Finset.sum_sub_distrib]
  have hCard : (core.card : ℝ) ≠ 0 := by
    exact_mod_cast hCore.card_ne_zero
  have hSumOne : (∑ _x ∈ core, (1 : ℝ)) = core.card := by
    simp
  rw [hSumOne]
  rw [averagePower]
  field_simp [hCard]

theorem sureScreeningFailure_le_card_mul_one_sub_averagePower
    [DecidableEq Variable]
    {core : Finset Variable} {failureProbability : ℝ}
    {selectionProbability : Variable → ℝ}
    (hCore : core.Nonempty)
    (hUnion : failureProbability ≤
      ∑ v ∈ core, (1 - selectionProbability v)) :
    failureProbability ≤
      core.card * (1 - averagePower core selectionProbability) := by
  rw [← sum_one_sub_selectionProbability_eq_card_mul_one_sub_averagePower
    core selectionProbability hCore]
  exact hUnion

theorem sureScreeningFailure_le_card_mul_one_sub_averagePower_of_eventProbabilities
    [MeasurableSpace Omega] [DecidableEq Variable]
    (mu : Measure Omega) (core : Finset Variable)
    (selected : Omega → Finset Variable) (selectionProbability : Variable → ℝ)
    (failureProbability : ℝ)
    (hCore : core.Nonempty)
    (hFailure : failureProbability ≤
      ∑ v ∈ core, (mu (variableMissEvent selected v)).toReal)
    (hSelected : ∀ v ∈ core,
      selectionProbability v = (mu (variableSelectedEvent selected v)).toReal)
    (hMissComplement : ∀ v ∈ core,
      (mu (variableMissEvent selected v)).toReal =
        1 - (mu (variableSelectedEvent selected v)).toReal) :
    failureProbability ≤
      core.card * (1 - averagePower core selectionProbability) := by
  apply sureScreeningFailure_le_card_mul_one_sub_averagePower hCore
  calc
    failureProbability ≤ ∑ v ∈ core, (mu (variableMissEvent selected v)).toReal := hFailure
    _ = ∑ v ∈ core, (1 - selectionProbability v) := by
      apply Finset.sum_congr rfl
      intro v hv
      rw [hMissComplement v hv, ← hSelected v hv]

theorem sureScreeningFailure_tendsto_zero_of_rate
    (failureProbability supportPowerDeficit : ℕ → ℝ)
    (hNonneg : ∀ n, 0 ≤ failureProbability n)
    (hBound : ∀ n, failureProbability n ≤ supportPowerDeficit n)
    (hRate : Tendsto supportPowerDeficit atTop (nhds 0)) :
    Tendsto failureProbability atTop (nhds 0) := by
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hRate hNonneg hBound

end BiostatFormal.HighDimensional.KnockoffAdjustment
