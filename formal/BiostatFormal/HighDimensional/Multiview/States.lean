import BiostatFormal.HighDimensional.Multiview.Guards
import BiostatFormal.HighDimensional.Multiview.Risks

/-!
# Shared multiview pipeline states

This file makes the source's state boundaries structural:

* a failed candidate has no record that could be passed to a risk;
* a paired state uses one slot and one stage with fixed `AB` and `BA` fields;
* an atomic guard can accept both post records, jointly fall back to both pre records, or reject;
* a final selected output carries membership in the realized guarded pool;
* every successful record retains its fit, normalization, and remaining frozen risk context.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uCenter uLabel uSlot uFitParam uNormalization uFrozen
  uFailureReason uRejectReason

/-- Informal refinement stage `pre` or `post` ↔ Lean: an explicit finite tag. -/
inductive CandidateStage where
  | pre
  | post
  deriving DecidableEq, Fintype, Repr

/--
Informal frozen risk context `T_{m,b}` ↔ Lean: the stage, fitted parameter, normalization context,
and all other frozen validation/replay/numerical information.  The direction and slot remain fields
of the surrounding `CandidateRecord`.
-/
structure RiskContext
    (FitParam : Type uFitParam)
    (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen) where
  stage : CandidateStage
  fitParameter : FitParam
  normalization : NormalizationContext
  frozen : FrozenContext

/-- A complete record whose context retains stage, fit, normalization, and auxiliary frozen data. -/
abbrev RiskRecord
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen) :=
  CandidateRecord Center Label Slot (RiskContext FitParam NormalizationContext FrozenContext)

/--
A successful risk input is a complete record bundled with evidence for the caller's legality
predicate.  Composite risks should use this type, never `CandidateState`.
-/
abbrev RiskCandidate
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop) :=
  {record // Legal record}

/--
Informal legal candidate or failure sentinel ↔ Lean: an indexed state.  The successful constructor
contains a legal complete record tied to the declared slot, direction, and stage.  The failed
constructor contains only a reason; it cannot fabricate a label or risk input.
-/
inductive CandidateState
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop)
    (FailureReason : Type uFailureReason)
    (slot : Slot) (direction : Direction) (stage : CandidateStage) where
  | valid
      (candidate : RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)
      (slot_eq : candidate.1.slot = slot)
      (direction_eq : candidate.1.direction = direction)
      (stage_eq : candidate.1.context.stage = stage)
  | failed (reason : FailureReason)

/-- A failed state maps to `none`; only a successful state exposes a legal risk candidate. -/
def CandidateState.toOption
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    {slot : Slot} {direction : Direction} {stage : CandidateStage} :
    CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot direction stage →
      Option (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)
  | .valid candidate _ _ _ => some candidate
  | .failed _ => none

@[simp]
theorem CandidateState.toOption_valid
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    {slot : Slot} {direction : Direction} {stage : CandidateStage}
    (candidate : RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)
    (hslot : candidate.1.slot = slot) (hdirection : candidate.1.direction = direction)
    (hstage : candidate.1.context.stage = stage) :
    CandidateState.toOption
        (CandidateState.valid candidate hslot hdirection hstage :
          CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
            FailureReason slot direction stage) = some candidate :=
  rfl

@[simp]
theorem CandidateState.toOption_failed
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    {slot : Slot} {direction : Direction} {stage : CandidateStage}
    (reason : FailureReason) :
    CandidateState.toOption
        (CandidateState.failed reason :
          CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
            FailureReason slot direction stage) = none :=
  rfl

/--
One slot's two directional states at one common stage.  The indices make cross-slot pairs,
direction swaps, and mixed pre/post pairs unrepresentable.
-/
structure PairedCandidateState
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop)
    (FailureReason : Type uFailureReason)
    (slot : Slot) (stage : CandidateStage) where
  ab : CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
    FailureReason slot .AB stage
  ba : CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
    FailureReason slot .BA stage

/--
A successful same-slot AB/BA pair.  This is the payload accepted by the atomic guard, so a failed
candidate cannot enter either accepted branch.
-/
structure PairedRiskCandidates
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop)
    (slot : Slot) (stage : CandidateStage) where
  ab : RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal
  ba : RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal
  ab_slot : ab.1.slot = slot
  ba_slot : ba.1.slot = slot
  ab_direction : ab.1.direction = .AB
  ba_direction : ba.1.direction = .BA
  ab_stage : ab.1.context.stage = stage
  ba_stage : ba.1.context.stage = stage

/-- A successful pair can be viewed as a paired state with two `valid` constructors. -/
def PairedRiskCandidates.toPairedCandidateState
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {FailureReason : Type uFailureReason}
    {slot : Slot} {stage : CandidateStage}
    (pair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot stage) :
    PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
      FailureReason slot stage where
  ab := .valid pair.ab pair.ab_slot pair.ab_direction pair.ab_stage
  ba := .valid pair.ba pair.ba_slot pair.ba_direction pair.ba_stage

/--
The atomic per-slot guard has exactly three outcomes.  There is no constructor for accepting one
post record and one pre record.
-/
inductive AtomicGuardResult
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop)
    (RejectReason : Type uRejectReason)
    (slot : Slot) where
  | acceptPost
      (candidates : PairedRiskCandidates Center Label Slot FitParam NormalizationContext
        FrozenContext Legal slot .post)
  | fallbackPre
      (candidates : PairedRiskCandidates Center Label Slot FitParam NormalizationContext
        FrozenContext Legal slot .pre)
  | reject (reason : RejectReason)

/--
The only risk type exported by this state layer has successful legal records as its domain.
`CandidateState.failed` is not an inhabitant of this domain.
-/
abbrev CandidateCompositeRiskPair
    (Center : Type uCenter) (Label : Type uLabel) (Slot : Type uSlot)
    (FitParam : Type uFitParam) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop) :=
  CompositeRiskPair
    (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)

/-- Pair exactness uses one independent global-permutation existential per direction. -/
def PairedRiskCandidates.Exact
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {slot : Slot} {stage : CandidateStage}
    (pair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot stage)
    (truth : Partition Center Label) : Prop :=
  Multiview.Exact pair.ab.1.label truth ∧ Multiview.Exact pair.ba.1.label truth

/--
AB and BA may be exact under different global label permutations.  Each permutation is nevertheless
global across all centers in its own direction.
-/
theorem PairedRiskCandidates.exact_iff_exists_directionPermutations
    {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
    {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext → Prop}
    {slot : Slot} {stage : CandidateStage}
    (pair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot stage)
    (truth : Partition Center Label) :
    pair.Exact truth ↔
      (∃ σAB : LabelPermutation Label, relabel σAB pair.ab.1.label = truth) ∧
      (∃ σBA : LabelPermutation Label, relabel σBA pair.ba.1.label = truth) := by
  rfl

end BiostatFormal.HighDimensional.Multiview
