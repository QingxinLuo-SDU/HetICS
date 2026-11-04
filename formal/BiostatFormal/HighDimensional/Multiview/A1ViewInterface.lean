import BiostatFormal.HighDimensional.Multiview.States
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.MeasureTheory.Measure.GiryMonad

/-!
# Generic A1 active-view interface

A1 is indexed by a finite configuration of the seven candidate-generation views.  It does not
require X, F, and Y (or any other fixed collection) all to be active.  Its strong event says that
split/balance succeeds and at least one slot in the declared pool is an active, view-backed slot
whose AB/BA pair is exact.

The family `pairExact` supplies the already-defined same-slot, two-direction estimand.  This file
only organizes which slots are allowed to witness it; it does not assume or prove a generator-level
coverage probability.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uSample uCenter uLabel uSlot uFitParam uNormalization uFrozen uFailureReason

/-- A view or slot can be unused, active in the algorithm, or diagnostic-only. -/
inductive A1UseStatus where
  | inactive
  | active
  | diagnostic
  deriving DecidableEq, Fintype, Repr

/-- A finite status assignment on the fixed seven-view universe. -/
structure A1ViewConfiguration where
  status : View → A1UseStatus

/-- The configured subset `V_active ⊆ {X,Y,F,XY,XF,YF,XYF}`. -/
def A1ViewConfiguration.activeViews (configuration : A1ViewConfiguration) : Finset View :=
  Finset.univ.filter fun view ↦ configuration.status view = .active

/-- Diagnostic-only views are recorded separately and cannot witness A1. -/
def A1ViewConfiguration.diagnosticViews (configuration : A1ViewConfiguration) : Finset View :=
  Finset.univ.filter fun view ↦ configuration.status view = .diagnostic

/-- The configured active views are a subset of the fixed seven-view universe. -/
theorem A1ViewConfiguration.activeViews_subset_univ
    (configuration : A1ViewConfiguration) :
    configuration.activeViews ⊆ (Finset.univ : Finset View) :=
  Finset.subset_univ _

/-- Active and diagnostic view sets are disjoint by construction. -/
theorem A1ViewConfiguration.activeViews_disjoint_diagnosticViews
    (configuration : A1ViewConfiguration) :
    Disjoint configuration.activeViews configuration.diagnosticViews := by
  rw [Finset.disjoint_left]
  intro view hActive hDiagnostic
  have hActiveStatus : configuration.status view = .active :=
    (Finset.mem_filter.1 hActive).2
  have hDiagnosticStatus : configuration.status view = .diagnostic :=
    (Finset.mem_filter.1 hDiagnostic).2
  have hImpossible : A1UseStatus.active = A1UseStatus.diagnostic :=
    hActiveStatus.symm.trans hDiagnosticStatus
  exact A1UseStatus.noConfusion hImpossible

/-- Slot metadata is separate from the seven-view status because some active extras have no view. -/
structure A1SlotInterface (Slot : Type uSlot) where
  view : Slot → Option View
  status : Slot → A1UseStatus

/--
A slot is eligible to witness generic A1 exactly when it is active and is backed by an active view.
Viewless algorithmic extras and diagnostic slots cannot satisfy this predicate.
-/
def A1SlotInterface.IsActiveWitnessSlot
    {Slot : Type uSlot}
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (slot : Slot) : Prop :=
  slotInterface.status slot = .active ∧
    ∃ view, slotInterface.view slot = some view ∧ configuration.status view = .active

/-- A diagnostic slot cannot be an active A1 witness. -/
theorem A1SlotInterface.not_isActiveWitnessSlot_of_diagnostic
    {Slot : Type uSlot}
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (slot : Slot)
    (hDiagnostic : slotInterface.status slot = .diagnostic) :
    ¬slotInterface.IsActiveWitnessSlot configuration slot := by
  intro hActive
  have hImpossible : A1UseStatus.diagnostic = A1UseStatus.active :=
    hDiagnostic.symm.trans hActive.1
  exact A1UseStatus.noConfusion hImpossible

/--
Typed AB/BA pre-outputs for every slot.  A failed state has no candidate payload, while a valid
state retains slot, direction, stage, legality, and frozen context through `CandidateState`.
-/
structure A1SlotOutputs
    (Sample : Type uSample) (Center : Type uCenter) (Label : Type uLabel)
    (Slot : Type uSlot) (FitParam : Type uFitParam)
    (NormalizationContext : Type uNormalization) (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop)
    (FailureReason : Type uFailureReason) where
  ab : (slot : Slot) → Sample →
    CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
      FailureReason slot .AB .pre
  ba : (slot : Slot) → Sample →
    CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
      FailureReason slot .BA .pre

/-- One slot/direction returns a legal exact pre-candidate; failure sentinels cannot satisfy it. -/
def a1SlotExactPreEvent
    {Sample : Type uSample} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {FitParam : Type uFitParam}
    {NormalizationContext : Type uNormalization} {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    {slot : Slot} {direction : Direction}
    (truth : Partition Center Label)
    (output : Sample →
      CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot direction .pre) : Set Sample :=
  {sample | ∃ candidate,
    (output sample).toOption = some candidate ∧ Exact candidate.1.label truth}

/-- Same-slot AB/BA exactness with separate global label permutations in the two directions. -/
def a1SlotPairExactEvent
    {Sample : Type uSample} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {FitParam : Type uFitParam}
    {NormalizationContext : Type uNormalization} {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    (truth : Partition Center Label)
    (outputs : A1SlotOutputs Sample Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason)
    (slot : Slot) : Set Sample :=
  a1SlotExactPreEvent truth (outputs.ab slot) ∩
    a1SlotExactPreEvent truth (outputs.ba slot)

/-- Event that some slot in the finite pool is an active witness and supplies an exact AB/BA pair. -/
def activeSlotProvidesWitnessEvent
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairExact : Slot → Set Sample) : Set Sample :=
  {sample | ∃ slot, slot ∈ pool ∧
    slotInterface.IsActiveWitnessSlot configuration slot ∧ sample ∈ pairExact slot}

/--
The generic strong A1 estimand.  Its event number and meaning remain
`E_cover^A = E_sb ∩ {some active slot supplies PairExact}`.
-/
def a1CoverageEvent
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairExact : Slot → Set Sample) : Set Sample :=
  splitBalance ∩ activeSlotProvidesWitnessEvent configuration slotInterface pool pairExact

/-- Typed form of the generic A1 event, built from legal indexed AB/BA pre-output states. -/
def a1TypedCoverageEvent
    {Sample : Type uSample} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} [DecidableEq Slot]
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (truth : Partition Center Label)
    (outputs : A1SlotOutputs Sample Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason) : Set Sample :=
  a1CoverageEvent splitBalance configuration slotInterface pool
    (a1SlotPairExactEvent truth outputs)

/-- A retained active exact slot is sufficient for membership in the generic A1 event. -/
theorem mem_a1CoverageEvent_of_active_witness
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairExact : Slot → Set Sample)
    (sample : Sample) (slot : Slot)
    (hSplitBalance : sample ∈ splitBalance)
    (hSlot : slot ∈ pool)
    (hActive : slotInterface.IsActiveWitnessSlot configuration slot)
    (hPairExact : sample ∈ pairExact slot) :
    sample ∈ a1CoverageEvent splitBalance configuration slotInterface pool pairExact :=
  ⟨hSplitBalance, ⟨slot, hSlot, hActive, hPairExact⟩⟩

/-- Enlarging a finite pool cannot destroy an active witness. -/
theorem activeSlotProvidesWitnessEvent_mono
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (smaller larger : Finset Slot)
    (pairExact : Slot → Set Sample)
    (hSubset : smaller ⊆ larger) :
    activeSlotProvidesWitnessEvent configuration slotInterface smaller pairExact ⊆
      activeSlotProvidesWitnessEvent configuration slotInterface larger pairExact := by
  intro sample hSample
  rcases hSample with ⟨slot, hSmaller, hActive, hExact⟩
  exact ⟨slot, hSubset hSmaller, hActive, hExact⟩

/--
Deleting slots that are structurally unable to be active witnesses leaves the witness event
unchanged.  This is stronger than preserving one realized witness and does not inspect truth or
the data.
-/
theorem activeSlotProvidesWitnessEvent_eq_of_removed_not_activeWitness
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (original reduced : Finset Slot)
    (pairExact : Slot → Set Sample)
    (hReduced : reduced ⊆ original)
    (hRetained : ∀ slot ∈ original,
      slotInterface.IsActiveWitnessSlot configuration slot → slot ∈ reduced) :
    activeSlotProvidesWitnessEvent configuration slotInterface original pairExact =
      activeSlotProvidesWitnessEvent configuration slotInterface reduced pairExact := by
  apply Set.Subset.antisymm
  · intro sample hSample
    rcases hSample with ⟨slot, hOriginal, hActive, hExact⟩
    exact ⟨slot, hRetained slot hOriginal hActive, hActive, hExact⟩
  · exact activeSlotProvidesWitnessEvent_mono configuration slotInterface reduced original
      pairExact hReduced

/-- The same deletion invariance for the numbered A1 coverage event. -/
theorem a1CoverageEvent_eq_of_removed_not_activeWitness
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (original reduced : Finset Slot)
    (pairExact : Slot → Set Sample)
    (hReduced : reduced ⊆ original)
    (hRetained : ∀ slot ∈ original,
      slotInterface.IsActiveWitnessSlot configuration slot → slot ∈ reduced) :
    a1CoverageEvent splitBalance configuration slotInterface original pairExact =
      a1CoverageEvent splitBalance configuration slotInterface reduced pairExact := by
  unfold a1CoverageEvent
  rw [activeSlotProvidesWitnessEvent_eq_of_removed_not_activeWitness configuration slotInterface
    original reduced pairExact hReduced hRetained]

/-- Typed A1 coverage is invariant under the same structural deletion condition. -/
theorem a1TypedCoverageEvent_eq_of_removed_not_activeWitness
    {Sample : Type uSample} {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} [DecidableEq Slot]
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (original reduced : Finset Slot)
    (truth : Partition Center Label)
    (outputs : A1SlotOutputs Sample Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason)
    (hReduced : reduced ⊆ original)
    (hRetained : ∀ slot ∈ original,
      slotInterface.IsActiveWitnessSlot configuration slot → slot ∈ reduced) :
    a1TypedCoverageEvent splitBalance configuration slotInterface original truth outputs =
      a1TypedCoverageEvent splitBalance configuration slotInterface reduced truth outputs := by
  exact a1CoverageEvent_eq_of_removed_not_activeWitness splitBalance configuration slotInterface
    original reduced (a1SlotPairExactEvent truth outputs) hReduced hRetained

/-- Union event over the actual finite candidate envelope after a pool change. -/
def candidatePoolUnionEvent
    {Sample : Type uSample} {Candidate : Type uSlot} [DecidableEq Candidate]
    (pool : Finset Candidate) (failure : Candidate → Set Sample) : Set Sample :=
  ⋃ candidate ∈ pool, failure candidate

/-- Candidate-specific finite-pool union bound; no cross-candidate independence is used. -/
theorem measure_candidatePoolUnionEvent_le_sum
    {Sample : Type uSample} {Candidate : Type uSlot} [DecidableEq Candidate]
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (pool : Finset Candidate)
    (failure : Candidate → Set Sample)
    (tailBudget : Candidate → ENNReal)
    (hTail : ∀ candidate ∈ pool, μ (failure candidate) ≤ tailBudget candidate) :
    μ (candidatePoolUnionEvent pool failure) ≤ ∑ candidate ∈ pool, tailBudget candidate := by
  calc
    μ (candidatePoolUnionEvent pool failure)
        ≤ ∑ candidate ∈ pool, μ (failure candidate) := by
          exact measure_biUnion_finset_le pool _
    _ ≤ ∑ candidate ∈ pool, tailBudget candidate := by
      exact Finset.sum_le_sum fun candidate hCandidate ↦ hTail candidate hCandidate

/-- Common-tail specialization with the cardinality of the changed pool exposed exactly. -/
theorem measure_candidatePoolUnionEvent_le_card_mul
    {Sample : Type uSample} {Candidate : Type uSlot} [DecidableEq Candidate]
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (pool : Finset Candidate)
    (failure : Candidate → Set Sample)
    (commonTail : ENNReal)
    (hTail : ∀ candidate ∈ pool, μ (failure candidate) ≤ commonTail) :
    μ (candidatePoolUnionEvent pool failure) ≤ pool.card * commonTail := by
  calc
    μ (candidatePoolUnionEvent pool failure)
        ≤ ∑ candidate ∈ pool, commonTail :=
      measure_candidatePoolUnionEvent_le_sum μ pool failure (fun _ ↦ commonTail) hTail
    _ = pool.card * commonTail := by simp

end BiostatFormal.HighDimensional.Multiview
