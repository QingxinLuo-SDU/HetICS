import BiostatFormal.HighDimensional.Multiview.A1ViewInterface
import Mathlib.Tactic.DeriveFintype

/-!
# Preregistered B2 v3 active and diagnostic slot types

This is a new-version specification.  It does not alter the frozen v2 implementation or evidence.
Pure-F KMeans and Ward are represented by a disjoint diagnostic-only type; they cannot inhabit the
v3 active-candidate type used by deduplication, risk normalization, selection, guards, or labels.
F-containing fusion slots remain active.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

/-- Algorithms appearing in the v3 active candidate pool. -/
inductive B2V3CandidateAlgorithm where
  | kmeans
  | ward
  | balancedRandom
  deriving DecidableEq, Fintype, Repr

/--
Frozen v3 active-slot list: the v2 list with only `covariate__kmeans` and `covariate__ward`
removed.  The four F-containing fusion families remain present.
-/
inductive B2V3ActiveSlot where
  | outcomeKMeans
  | outcomeWard
  | treatmentKMeans
  | fusionXYKMeans
  | fusionXYWard
  | fusionXFKMeans
  | fusionXFWard
  | fusionYFKMeans
  | fusionYFWard
  | fusionFXYKMeans
  | fusionFXYWard
  | balancedRandom01
  | balancedRandom02
  | balancedRandom03
  | balancedRandom04
  deriving DecidableEq, Fintype, Repr

/-- Frozen active-slot order; this is also the candidate tie order for v3. -/
def b2V3ActiveSlotOrder : List B2V3ActiveSlot :=
  [.outcomeKMeans, .outcomeWard, .treatmentKMeans,
    .fusionXYKMeans, .fusionXYWard,
    .fusionXFKMeans, .fusionXFWard,
    .fusionYFKMeans, .fusionYFWard,
    .fusionFXYKMeans, .fusionFXYWard,
    .balancedRandom01, .balancedRandom02, .balancedRandom03, .balancedRandom04]

/-- Every active slot occurs in the frozen order. -/
theorem B2V3ActiveSlot.mem_activeSlotOrder (slot : B2V3ActiveSlot) :
    slot ∈ b2V3ActiveSlotOrder := by
  cases slot <;> simp [b2V3ActiveSlotOrder]

/-- No active slot is repeated. -/
theorem b2V3ActiveSlotOrder_nodup : b2V3ActiveSlotOrder.Nodup := by
  decide

/-- View used by each active slot.  No constructor maps to the pure covariate view. -/
def B2V3ActiveSlot.view : B2V3ActiveSlot → Option View
  | .outcomeKMeans | .outcomeWard => some .outcome
  | .treatmentKMeans => some .treatment
  | .fusionXYKMeans | .fusionXYWard => some .treatmentOutcome
  | .fusionXFKMeans | .fusionXFWard => some .treatmentCovariate
  | .fusionYFKMeans | .fusionYFWard => some .outcomeCovariate
  | .fusionFXYKMeans | .fusionFXYWard => some .treatmentOutcomeCovariate
  | .balancedRandom01 | .balancedRandom02 | .balancedRandom03 | .balancedRandom04 => none

/-- Algorithm used by each active slot. -/
def B2V3ActiveSlot.algorithm : B2V3ActiveSlot → B2V3CandidateAlgorithm
  | .outcomeKMeans | .treatmentKMeans | .fusionXYKMeans | .fusionXFKMeans |
      .fusionYFKMeans | .fusionFXYKMeans => .kmeans
  | .outcomeWard | .fusionXYWard | .fusionXFWard | .fusionYFWard | .fusionFXYWard => .ward
  | .balancedRandom01 | .balancedRandom02 | .balancedRandom03 | .balancedRandom04 => .balancedRandom

/-- Every active view-backed slot is eligible for the generic existential A1 event. -/
def B2V3ActiveSlot.strongWitnessEligible : B2V3ActiveSlot → Prop
  | .balancedRandom01 | .balancedRandom02 | .balancedRandom03 | .balancedRandom04 => False
  | _ => True

/-- Ordered active view-backed witness slots; viewless balanced-random extras are excluded. -/
def b2V3StrongWitnessSlotOrder : List B2V3ActiveSlot :=
  [.outcomeKMeans, .outcomeWard, .treatmentKMeans,
    .fusionXYKMeans, .fusionXYWard,
    .fusionXFKMeans, .fusionXFWard,
    .fusionYFKMeans, .fusionYFWard,
    .fusionFXYKMeans, .fusionFXYWard]

theorem B2V3ActiveSlot.mem_strongWitnessSlotOrder_iff (slot : B2V3ActiveSlot) :
    slot ∈ b2V3StrongWitnessSlotOrder ↔ slot.strongWitnessEligible := by
  cases slot <;> simp [b2V3StrongWitnessSlotOrder, B2V3ActiveSlot.strongWitnessEligible]

/--
Narrower set for the existing ordinary-KMeans generator proof.  Membership here is not required by
the generic A1 event; Ward slots need their own probability theorem before receiving a bound.
-/
def B2V3ActiveSlot.ordinaryKMeansCoverageEligible : B2V3ActiveSlot → Prop
  | .outcomeKMeans | .treatmentKMeans | .fusionXYKMeans | .fusionXFKMeans |
      .fusionYFKMeans | .fusionFXYKMeans => True
  | _ => False

def b2V3OrdinaryKMeansCoverageSlotOrder : List B2V3ActiveSlot :=
  [.outcomeKMeans, .treatmentKMeans, .fusionXYKMeans, .fusionXFKMeans,
    .fusionYFKMeans, .fusionFXYKMeans]

theorem B2V3ActiveSlot.mem_ordinaryKMeansCoverageSlotOrder_iff (slot : B2V3ActiveSlot) :
    slot ∈ b2V3OrdinaryKMeansCoverageSlotOrder ↔ slot.ordinaryKMeansCoverageEligible := by
  cases slot <;>
    simp [b2V3OrdinaryKMeansCoverageSlotOrder,
      B2V3ActiveSlot.ordinaryKMeansCoverageEligible]

/-- Every view-backed v3 active slot is non-pure-F. -/
theorem B2V3ActiveSlot.view_ne_covariate
    (slot : B2V3ActiveSlot) (view : View) (hView : slot.view = some view) :
    view ≠ .covariate := by
  cases slot <;> simp [B2V3ActiveSlot.view] at hView
  all_goals
    subst view
    decide

/-- The two removed pure-F slots remain available only as diagnostic outputs. -/
inductive B2V3PureFDiagnosticSlot where
  | covariateKMeans
  | covariateWard
  deriving DecidableEq, Fintype, Repr

def B2V3PureFDiagnosticSlot.view (_slot : B2V3PureFDiagnosticSlot) : View := .covariate

/--
Unified historical slot universe.  The left summand is the 15-slot v3 active pool and the right
summand is the two pure-F slots that remain active in v2 but become diagnostic in v3.
-/
abbrev B2CandidateSlot := Sum B2V3ActiveSlot B2V3PureFDiagnosticSlot

/-- View metadata for the unified 17-slot universe.  Balanced-random slots have no source view. -/
def B2CandidateSlot.view : B2CandidateSlot → Option View
  | .inl slot => slot.view
  | .inr slot => some slot.view

/-- Frozen B2-v2 view instance: every one of the seven source views is active. -/
def b2V2ViewConfiguration : A1ViewConfiguration where
  status _ := .active

/--
B2-v3 view instance: pure F is diagnostic, while X, Y, XY, XF, YF, and XYF remain active.
-/
def b2V3ViewConfiguration : A1ViewConfiguration where
  status
    | .covariate => .diagnostic
    | _ => .active

/-- Frozen B2-v2 slot instance: all original 17 slots remain active. -/
def b2V2SlotInterface : A1SlotInterface B2CandidateSlot where
  view := B2CandidateSlot.view
  status _ := .active

/-- B2-v3 slot instance: exactly the two pure-F slots are diagnostic-only. -/
def b2V3SlotInterface : A1SlotInterface B2CandidateSlot where
  view := B2CandidateSlot.view
  status
    | .inr _ => .diagnostic
    | .inl _ => .active

/-- Exact historical v2 logical-slot/tie order from the frozen source. -/
def b2V2CandidateSlotOrder : List B2CandidateSlot :=
  [.inl .outcomeKMeans, .inl .outcomeWard, .inl .treatmentKMeans,
    .inr .covariateKMeans, .inr .covariateWard,
    .inl .fusionXYKMeans, .inl .fusionXYWard,
    .inl .fusionXFKMeans, .inl .fusionXFWard,
    .inl .fusionYFKMeans, .inl .fusionYFWard,
    .inl .fusionFXYKMeans, .inl .fusionFXYWard,
    .inl .balancedRandom01, .inl .balancedRandom02,
    .inl .balancedRandom03, .inl .balancedRandom04]

/-- Historical v2 retains the complete 17-slot universe. -/
def b2V2CandidatePool : Finset B2CandidateSlot := Finset.univ

/-- v3 active pool obtained from the same universe by the declared slot status. -/
def b2V3CandidatePool : Finset B2CandidateSlot :=
  Finset.univ.filter fun slot ↦ b2V3SlotInterface.status slot = .active

/-- v3 diagnostic pool contains the two pure-F slots. -/
def b2V3DiagnosticPool : Finset B2CandidateSlot :=
  Finset.univ.filter fun slot ↦ b2V3SlotInterface.status slot = .diagnostic

theorem B2CandidateSlot.mem_b2V2CandidateSlotOrder (slot : B2CandidateSlot) :
    slot ∈ b2V2CandidateSlotOrder := by
  rcases slot with slot | slot <;> cases slot <;> simp [b2V2CandidateSlotOrder]

theorem b2V2CandidateSlotOrder_nodup : b2V2CandidateSlotOrder.Nodup := by
  decide

theorem b2V2CandidatePool_card : b2V2CandidatePool.card = 17 := by
  decide

theorem b2V3CandidatePool_card : b2V3CandidatePool.card = 15 := by
  decide

theorem b2V3DiagnosticPool_card : b2V3DiagnosticPool.card = 2 := by
  decide

@[simp]
theorem b2V3_covariate_not_active :
    View.covariate ∉ b2V3ViewConfiguration.activeViews := by
  decide

@[simp]
theorem b2V3_covariate_diagnostic :
    View.covariate ∈ b2V3ViewConfiguration.diagnosticViews := by
  decide

theorem b2V2_every_view_active (view : View) :
    view ∈ b2V2ViewConfiguration.activeViews := by
  cases view <;> decide

@[simp]
theorem b2V3_mem_candidatePool_iff_active (slot : B2CandidateSlot) :
    slot ∈ b2V3CandidatePool ↔ b2V3SlotInterface.status slot = .active := by
  simp [b2V3CandidatePool]

@[simp]
theorem b2V3_mem_diagnosticPool_iff_diagnostic (slot : B2CandidateSlot) :
    slot ∈ b2V3DiagnosticPool ↔ b2V3SlotInterface.status slot = .diagnostic := by
  simp [b2V3DiagnosticPool]

/--
Under the v3 status assignment, physically removing the two diagnostic pure-F slots from an
enumerated superset cannot change the event that some active slot supplies the A1 witness.
This does not equate the historical v2 and v3 status assignments.
-/
theorem b2V3_removing_diagnosticF_preserves_activeWitnessEvent
    {Sample : Type*}
    (pairExact : B2CandidateSlot → Set Sample) :
    activeSlotProvidesWitnessEvent b2V3ViewConfiguration b2V3SlotInterface
        b2V2CandidatePool pairExact =
      activeSlotProvidesWitnessEvent b2V3ViewConfiguration b2V3SlotInterface
        b2V3CandidatePool pairExact := by
  apply activeSlotProvidesWitnessEvent_eq_of_removed_not_activeWitness
  · intro slot hSlot
    simp [b2V2CandidatePool]
  · intro slot _ hActive
    exact b2V3_mem_candidatePool_iff_active slot |>.2 hActive.1

/-- The numbered A1 coverage event is likewise unchanged by deleting those diagnostic slots. -/
theorem b2V3_removing_diagnosticF_preserves_a1CoverageEvent
    {Sample : Type*}
    (splitBalance : Set Sample)
    (pairExact : B2CandidateSlot → Set Sample) :
    a1CoverageEvent splitBalance b2V3ViewConfiguration b2V3SlotInterface
        b2V2CandidatePool pairExact =
      a1CoverageEvent splitBalance b2V3ViewConfiguration b2V3SlotInterface
        b2V3CandidatePool pairExact := by
  apply a1CoverageEvent_eq_of_removed_not_activeWitness
  · intro slot hSlot
    simp [b2V2CandidatePool]
  · intro slot _ hActive
    exact b2V3_mem_candidatePool_iff_active slot |>.2 hActive.1

/-- Candidate record index used by the two directions and pre/post stages. -/
abbrev B2CandidateRecordIndex := B2CandidateSlot × (Direction × CandidateStage)

def b2V2CandidateRecordEnvelope : Finset B2CandidateRecordIndex :=
  b2V2CandidatePool.product
    ((Finset.univ : Finset Direction).product (Finset.univ : Finset CandidateStage))

def b2V3CandidateRecordEnvelope : Finset B2CandidateRecordIndex :=
  b2V3CandidatePool.product
    ((Finset.univ : Finset Direction).product (Finset.univ : Finset CandidateStage))

theorem b2V2CandidateRecordEnvelope_card : b2V2CandidateRecordEnvelope.card = 68 := by
  decide

theorem b2V3CandidateRecordEnvelope_card : b2V3CandidateRecordEnvelope.card = 60 := by
  decide

/-- Historical v2 common-tail union bound over `17 × 2 × 2 = 68` records. -/
theorem measure_b2V2CandidateRecordUnionEvent_le_68_mul
    {Sample : Type*} [MeasurableSpace Sample]
    (μ : Measure Sample)
    (failure : B2CandidateRecordIndex → Set Sample)
    (commonTail : ENNReal)
    (hTail : ∀ candidate ∈ b2V2CandidateRecordEnvelope,
      μ (failure candidate) ≤ commonTail) :
    μ (candidatePoolUnionEvent b2V2CandidateRecordEnvelope failure) ≤ 68 * commonTail := by
  simpa [b2V2CandidateRecordEnvelope_card] using
    measure_candidatePoolUnionEvent_le_card_mul μ b2V2CandidateRecordEnvelope failure commonTail
      hTail

/-- Revised v3 common-tail union bound over `15 × 2 × 2 = 60` records. -/
theorem measure_b2V3CandidateRecordUnionEvent_le_60_mul
    {Sample : Type*} [MeasurableSpace Sample]
    (μ : Measure Sample)
    (failure : B2CandidateRecordIndex → Set Sample)
    (commonTail : ENNReal)
    (hTail : ∀ candidate ∈ b2V3CandidateRecordEnvelope,
      μ (failure candidate) ≤ commonTail) :
    μ (candidatePoolUnionEvent b2V3CandidateRecordEnvelope failure) ≤ 60 * commonTail := by
  simpa [b2V3CandidateRecordEnvelope_card] using
    measure_candidatePoolUnionEvent_le_card_mul μ b2V3CandidateRecordEnvelope failure commonTail
      hTail

/-- An active candidate can enter v3 downstream candidate operations. -/
structure B2V3ActiveCandidate (Payload : Type*) where
  slot : B2V3ActiveSlot
  payload : Payload

/-- A pure-F result has a disjoint type and is diagnostic-only. -/
structure B2V3PureFDiagnostic (Payload : Type*) where
  slot : B2V3PureFDiagnosticSlot
  payload : Payload

end BiostatFormal.HighDimensional.Multiview
