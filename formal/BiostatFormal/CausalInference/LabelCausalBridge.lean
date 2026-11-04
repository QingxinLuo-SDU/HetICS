import BiostatFormal.HighDimensional.Multiview.SelectorContract

/-!
# Exact-label causal bridge

This bridge transfers classwise deterministic adjustments only after exact recovery modulo one
global label permutation.  It deliberately makes no stochastic claim: pairwise-disjoint split
sets do not establish independence, exchangeability, or Model-X validity.
-/

namespace BiostatFormal.CausalInference

open BiostatFormal.HighDimensional.Multiview

universe uUnit uCenter uLabel uIndividual uVariable uSlot uContext uOmega uReason

/--
Three unit sets used for label learning, knockoff construction, and effect estimation.
This structure proves only pairwise set disjointness; it does not prove stochastic independence,
exchangeability, or Model-X validity.
-/
structure ThreeStageSplit (Unit : Type uUnit) [DecidableEq Unit] where
  labelLearning : Finset Unit
  knockoff : Finset Unit
  effectEstimation : Finset Unit
  label_knockoff_disjoint : Disjoint labelLearning knockoff
  label_effect_disjoint : Disjoint labelLearning effectEstimation
  knockoff_effect_disjoint : Disjoint knockoff effectEstimation

/-- Centers assigned a given class label. -/
def classMembers {Center : Type uCenter} {Label : Type uLabel}
    (labels : Partition Center Label) (label : Label) : Set Center :=
  {center | labels center = label}

/-- Lift center-level labels to individuals through the explicitly supplied center map. -/
def individualLabels {Individual : Type uIndividual} {Center : Type uCenter} {Label : Type uLabel}
    (centerOf : Individual → Center) (labels : Partition Center Label) : Partition Individual Label :=
  fun individual => labels (centerOf individual)

/-- Apply a frozen selector to members of one label class. -/
def classwiseAdjustment {Center : Type uCenter} {Label : Type uLabel} {Variable : Type uVariable}
    [DecidableEq Variable] (selector : Set Center → Finset Variable)
    (labels : Partition Center Label) (label : Label) : Finset Variable :=
  selector (classMembers labels label)

/--
Exact recovery identifies learned classes with true classes through one global permutation, not
coordinatewise equality under the original label names.
-/
theorem classMembers_eq_under_exact
    {Center : Type uCenter} {Label : Type uLabel}
    {estimated truth : Partition Center Label}
    (hExact : Exact estimated truth) :
    ∃ sigma : LabelPermutation Label,
      ∀ label, classMembers estimated label = classMembers truth (sigma label) := by
  rcases hExact with ⟨sigma, hSigma⟩
  refine ⟨sigma, fun label => ?_⟩
  ext center
  have hAt := congrFun hSigma center
  constructor
  · intro hLabel
    change truth center = sigma label
    calc
      truth center = sigma (estimated center) := hAt.symm
      _ = sigma label := congrArg sigma hLabel
  · intro hLabel
    change estimated center = label
    apply sigma.injective
    calc
      sigma (estimated center) = truth center := hAt
      _ = sigma label := hLabel

/-- Exact center labels lift through the explicitly supplied `centerOf` map. -/
theorem individualLabels_exact
    {Individual : Type uIndividual} {Center : Type uCenter} {Label : Type uLabel}
    {estimated truth : Partition Center Label}
    (centerOf : Individual → Center) (hExact : Exact estimated truth) :
    Exact (individualLabels centerOf estimated) (individualLabels centerOf truth) := by
  rcases hExact with ⟨sigma, hSigma⟩
  refine ⟨sigma, ?_⟩
  funext individual
  exact congrFun hSigma (centerOf individual)

/-- A frozen selector agrees on corresponding exact-recovery classes. -/
theorem classwiseAdjustment_eq_under_exact
    {Center : Type uCenter} {Label : Type uLabel} {Variable : Type uVariable}
    {estimated truth : Partition Center Label}
    [DecidableEq Variable] (selector : Set Center → Finset Variable)
    (hExact : Exact estimated truth) :
    ∃ sigma : LabelPermutation Label,
      ∀ label,
        classwiseAdjustment selector estimated label =
          classwiseAdjustment selector truth (sigma label) := by
  rcases classMembers_eq_under_exact hExact with ⟨sigma, hMembers⟩
  refine ⟨sigma, fun label => ?_⟩
  unfold classwiseAdjustment
  rw [hMembers label]

/-- Extract the selected candidate's label, returning `none` for abstention. -/
def selectedLabel?
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot} {Context : Type uContext}
    {Reason : Type uReason}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    {pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context)}
    (output : AlgorithmOutput pool Reason) : Option (Partition Center Label) :=
  match output with
  | .selected selected => some selected.1.label
  | .abstain _ => none

@[simp] theorem selectedLabel?_abstain
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot} {Context : Type uContext}
    {Reason : Type uReason}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    {pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context)}
    (reason : Reason) :
    selectedLabel? (AlgorithmOutput.abstain reason : AlgorithmOutput pool Reason) = none :=
  rfl

@[simp] theorem selectedLabel?_selected
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot} {Context : Type uContext}
    {Reason : Type uReason}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    {pool : GuardedCandidatePool (CandidateRecord Center Label Slot Context)}
    (selected : GuardedPoolMember pool) :
    selectedLabel? (AlgorithmOutput.selected selected : AlgorithmOutput pool Reason) =
      some selected.1.label :=
  rfl

/--
An exact selected record transfers every deterministic classwise adjustment along the same global
permutation.  The existing `exactSelectedEvent` fixes the abstention-reason type to
`SelectorAbstainReason`.
-/
theorem exactSelected_classwiseAdjustment_transfer
    {Omega : Type uOmega} {Center : Type uCenter} {Label : Type uLabel}
    {Variable : Type uVariable} {Slot : Type uSlot} {Context : Type uContext}
    [DecidableEq (CandidateRecord Center Label Slot Context)]
    {pool : Omega → GuardedCandidatePool (CandidateRecord Center Label Slot Context)}
    (output : (omega : Omega) → AlgorithmOutput (pool omega) SelectorAbstainReason)
    (truth : Partition Center Label)
    [DecidableEq Variable]
    (selector : Set Center → Finset Variable)
    {omega : Omega}
    (hExact : omega ∈ exactSelectedEvent output truth) :
    ∃ (selected : GuardedPoolMember (pool omega)) (sigma : LabelPermutation Label),
      output omega = .selected selected ∧
      ∀ label,
        classwiseAdjustment selector selected.1.label label =
          classwiseAdjustment selector truth (sigma label) := by
  rcases hExact with ⟨selected, hSelected, hExactSelected⟩
  have hLabels : Exact selected.1.label truth := by
    simpa [exactCandidateSet] using hExactSelected
  rcases classwiseAdjustment_eq_under_exact selector hLabels with ⟨sigma, hAdjustment⟩
  exact ⟨selected, sigma, hSelected, hAdjustment⟩

end BiostatFormal.CausalInference
