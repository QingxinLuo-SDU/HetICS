import BiostatFormal.HighDimensional.Multiview.A1ViewInterface

/-!
# Exact and attraction targets for A1

The active-slot event constructor is independent of the candidate-quality predicate.  The exact
and attraction targets below are deliberately separate named specializations.  No probability
bound, positive attraction radius, or A3 contraction theorem is assumed here.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uSample uSlot

/-- A neutral name for A1 coverage with an explicitly supplied paired-slot quality event. -/
def a1CoverageEventOf
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairGood : Slot → Set Sample) : Set Sample :=
  a1CoverageEvent splitBalance configuration slotInterface pool pairGood

/-- The frozen strong A1 estimand: some active slot is exact in both directions. -/
def a1ExactCoverageEvent
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairExact : Slot → Set Sample) : Set Sample :=
  a1CoverageEventOf splitBalance configuration slotInterface pool pairExact

/--
The weaker A1 attraction target.  `pairWithinRadius` must be instantiated from a declared metric
and radius; this definition does not assert that such a radius is an A3 basin.
-/
def a1AttractionCoverageEvent
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairWithinRadius : Slot → Set Sample) : Set Sample :=
  a1CoverageEventOf splitBalance configuration slotInterface pool pairWithinRadius

/-- Strengthening every paired-slot quality event can only shrink the resulting A1 event. -/
theorem a1CoverageEventOf_mono_quality
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (strong weak : Slot → Set Sample)
    (hStrongWeak : ∀ slot, strong slot ⊆ weak slot) :
    a1CoverageEventOf splitBalance configuration slotInterface pool strong ⊆
      a1CoverageEventOf splitBalance configuration slotInterface pool weak := by
  intro sample hSample
  rcases hSample with ⟨hSplitBalance, slot, hSlot, hActive, hStrong⟩
  exact ⟨hSplitBalance, slot, hSlot, hActive, hStrongWeak slot hStrong⟩

/--
Exact coverage implies attraction coverage only after the slot-level zero-error implication has
been supplied explicitly.  This is not an A3 recovery theorem.
-/
theorem a1ExactCoverageEvent_subset_attraction
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairExact pairWithinRadius : Slot → Set Sample)
    (hExactWithinRadius : ∀ slot, pairExact slot ⊆ pairWithinRadius slot) :
    a1ExactCoverageEvent splitBalance configuration slotInterface pool pairExact ⊆
      a1AttractionCoverageEvent splitBalance configuration slotInterface pool
        pairWithinRadius :=
  a1CoverageEventOf_mono_quality splitBalance configuration slotInterface pool
    pairExact pairWithinRadius hExactWithinRadius

/-- The quality event for one slot chosen before the sample is observed. -/
def a1FixedSlotCoverageEvent
    {Sample : Type uSample} {Slot : Type uSlot}
    (splitBalance : Set Sample)
    (pairGood : Slot → Set Sample)
    (witness : Slot) : Set Sample :=
  splitBalance ∩ pairGood witness

/-- A preregistered active slot injects its fixed-slot event into the generic active-pool event. -/
theorem a1FixedSlotCoverageEvent_subset
    {Sample : Type uSample} {Slot : Type uSlot} [DecidableEq Slot]
    (splitBalance : Set Sample)
    (configuration : A1ViewConfiguration)
    (slotInterface : A1SlotInterface Slot)
    (pool : Finset Slot)
    (pairGood : Slot → Set Sample)
    (witness : Slot)
    (hWitnessPool : witness ∈ pool)
    (hWitnessActive : slotInterface.IsActiveWitnessSlot configuration witness) :
    a1FixedSlotCoverageEvent splitBalance pairGood witness ⊆
      a1CoverageEventOf splitBalance configuration slotInterface pool pairGood := by
  intro sample hSample
  exact mem_a1CoverageEvent_of_active_witness splitBalance configuration slotInterface pool
    pairGood sample witness hSample.1 hWitnessPool hWitnessActive hSample.2

end BiostatFormal.HighDimensional.Multiview
