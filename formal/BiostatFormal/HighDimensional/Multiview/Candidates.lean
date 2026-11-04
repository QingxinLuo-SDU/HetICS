import BiostatFormal.HighDimensional.Multiview.Labels
import BiostatFormal.HighDimensional.Multiview.Views
import Mathlib.Data.Set.Basic

/-!
# Contextual candidate records

A risk input is a record rather than a bare partition.  Its slot, direction, companion partition,
and frozen context remain visible even when two records have the same label projection.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uCenter uLabel uSlot uContext

/-- Informal split direction `b ∈ {AB,BA}` ↔ Lean: a two-constructor finite type. -/
inductive Direction where
  | AB
  | BA
  deriving DecidableEq, Fintype, Repr

/-- Informal `b̄` ↔ Lean: the complementary label-splitting direction. -/
def Direction.flip : Direction → Direction
  | .AB => .BA
  | .BA => .AB

@[simp]
theorem Direction.flip_flip (direction : Direction) : direction.flip.flip = direction := by
  cases direction <;> rfl

/--
Informal `(m,b; z_guard, z_companion, T)` ↔ Lean: one contextual candidate record.  `context`
is abstract because fitted models, normalization tables, validation maps, and numerical rules are
not yet stable Lean objects.
-/
structure CandidateRecord
    (Center : Type uCenter) (Label : Type uLabel)
    (Slot : Type uSlot) (Context : Type uContext) where
  slot : Slot
  direction : Direction
  label : Partition Center Label
  companion : Partition Center Label
  context : Context

/-- Informal `Z_exact` ↔ Lean: records whose label projection is exact relative to truth. -/
def exactCandidateSet
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    (truth : Partition Center Label) :
    Set (CandidateRecord Center Label Slot Context) :=
  {record | Exact record.label truth}

/-- Informal `Z_bad` ↔ Lean: records whose label projection is not exact relative to truth. -/
def badCandidateSet
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    (truth : Partition Center Label) :
    Set (CandidateRecord Center Label Slot Context) :=
  {record | ¬Exact record.label truth}

theorem badCandidateSet_eq_compl_exactCandidateSet
    {Center : Type uCenter} {Label : Type uLabel}
    {Slot : Type uSlot} {Context : Type uContext}
    (truth : Partition Center Label) :
    badCandidateSet (Slot := Slot) (Context := Context) truth =
      (exactCandidateSet (Slot := Slot) (Context := Context) truth)ᶜ := by
  rfl

end BiostatFormal.HighDimensional.Multiview
