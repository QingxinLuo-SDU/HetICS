import BiostatFormal.HighDimensional.Multiview.B2V3A3PrimitiveTails
import BiostatFormal.HighDimensional.Multiview.B2V3SourceA2

/-!
# Actual B2-v3 A3/A2 connections

This file derives the source-survival premise used by the primitive-tail theorem from the actual
ordinary-call certificates, candidate-local terminal refit/reassignment, atomic guard, and
canonical provenance trace.  It then reuses the verified outer equal-weight Y/X selector.  The
post-selection fixed-label refit is intentionally absent because it does not change frozen labels.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uOmega uFit uNorm uFrozen uEval uReject uKey uOwner

namespace B2V3ActualPrimitiveLedger

/-- No raw source atom fails anywhere in the fixed witness ledger. -/
def rawGood {Omega : Type uOmega} {T : Nat}
    (ledger : B2V3ActualPrimitiveLedger Omega T) (omega : Omega) : Prop :=
  ∀ q kind, omega ∉ ledger.failure kind q

end B2V3ActualPrimitiveLedger

/-! ## Fixed-slot actual-source binding -/

/--
One realization of the actual deterministic source bindings for a fixed active logical slot. AB
and BA use the same `witnessSlot`, but their global label permutations and physical owners may
differ.
-/
structure B2V3ActualGoodRealizationFor
    (witnessSlot : B2V3ActiveSlot)
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    {Key : Type uKey} {Owner : Type uOwner}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega)
    (provenance : B2V3WitnessBinding witnessSlot Key Owner)
    (Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop)
    (RejectReason : Type uReject)
    (T : Nat) (maxSteps : Nat)
    (referenceAB referenceBA : Fit) (rThetaAB rThetaBA : ℝ) where
  paired : B2V3ActualPairedRun (Fin E) Fit T
  abCertificate :
    B2V3ActualDirectionCertificate (Fin E) Fit T paired.ab ctx.truth referenceAB
  baCertificate :
    B2V3ActualDirectionCertificate (Fin E) Fit T paired.ba ctx.truth referenceBA
  abSuccess :
    B2V3ActualTerminalSuccess (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal T paired.ab ctx.truth
      referenceAB rThetaAB witnessSlot .AB maxSteps
  baSuccess :
    B2V3ActualTerminalSuccess (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal T paired.ba ctx.truth
      referenceBA rThetaBA witnessSlot .BA maxSteps
  outcome : B2V3PairedSourceOutcome (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal ctx.truth
    referenceAB referenceBA rThetaAB rThetaBA RejectReason witnessSlot maxSteps
  outcome_eq : outcome = paired.terminalize abCertificate baCertificate
    (.success abSuccess) (.success baSuccess)
  postOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
    witnessSlot .post → Prop
  preOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
    witnessSlot .pre → Prop
  rejectReason : RejectReason
  preState : PairedCandidateState (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
    (B2V3AdapterFailure RejectReason) witnessSlot .pre
  guard_postOK : ∀ postPair,
    outcome.toPairedCandidateState.toOption = some postPair → postOK postPair
  trace : B2V3LocalToGlobalTrace ctx omega
    (atomicGuard postOK preOK rejectReason preState outcome.toPairedCandidateState).toGuardedPool
  trace_witnessSlot_provenance : ∀ candidate,
    candidate ∈
        (atomicGuard postOK preOK rejectReason preState
          outcome.toPairedCandidateState).toGuardedPool →
      ∃ provenanceEntry ∈ (trace.canonicalRecord candidate).provenance,
        ∃ direction,
          provenanceEntry.logicalId = (witnessSlot, direction, CandidateStage.post)

namespace B2V3ActualGoodRealizationFor

variable
    {witnessSlot : B2V3ActiveSlot}
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    {omega : Omega}
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {RejectReason : Type uReject}
    {Key : Type uKey} {Owner : Type uOwner}
    {provenance : B2V3WitnessBinding witnessSlot Key Owner}
    {T : Nat} {maxSteps : Nat}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}

/-- A fixed-slot realization retains an exact record in the guarded pool. -/
theorem exactSurvival_of_rawGood
    (realization : B2V3ActualGoodRealizationFor witnessSlot ctx omega provenance Legal RejectReason
      T maxSteps referenceAB referenceBA rThetaAB rThetaBA) :
    omega ∈ ctx.exactSurvival := by
  have hCertified : realization.outcome.BothCertified := by
    rw [realization.outcome_eq]
    refine ⟨?_, ?_⟩
    · exact ⟨realization.abSuccess.toCertifiedPayload realization.abCertificate, rfl⟩
    · exact ⟨realization.baSuccess.toCertifiedPayload realization.baCertificate, rfl⟩
  have hGuardExact := b2V3CoarseGuard_exact_of_certified realization.postOK realization.preOK
    realization.rejectReason realization.preState realization.outcome hCertified
    realization.guard_postOK
  obtain ⟨candidate, hLocal, hExact⟩ :=
    AtomicGuardResult.exists_exact_mem_toGuardedPool _ ctx.truth hGuardExact
  obtain ⟨_provenanceEntry, _hEntry, _direction, _hProvenance⟩ :=
    realization.trace_witnessSlot_provenance candidate hLocal
  exact realization.trace.exactSurvival candidate hLocal hExact

end B2V3ActualGoodRealizationFor

/--
One realization of all actual deterministic source bindings.  `outcome_eq` prevents an unrelated
exact terminal outcome from being substituted for the AB/BA runs certified above it.
-/
structure B2V3ActualGoodRealization
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    {Key : Type uKey} {Owner : Type uOwner}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega)
    (provenance : B2V3OutcomeWitnessBinding Key Owner)
    (Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop)
    (RejectReason : Type uReject)
    (T : Nat) (maxSteps : Nat)
    (referenceAB referenceBA : Fit) (rThetaAB rThetaBA : ℝ) where
  paired : B2V3ActualPairedRun (Fin E) Fit T
  abCertificate :
    B2V3ActualDirectionCertificate (Fin E) Fit T paired.ab ctx.truth referenceAB
  baCertificate :
    B2V3ActualDirectionCertificate (Fin E) Fit T paired.ba ctx.truth referenceBA
  abSuccess :
    B2V3ActualTerminalSuccess (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal T paired.ab ctx.truth
      referenceAB rThetaAB B2V3ActiveSlot.outcomeKMeans .AB maxSteps
  baSuccess :
    B2V3ActualTerminalSuccess (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal T paired.ba ctx.truth
      referenceBA rThetaBA B2V3ActiveSlot.outcomeKMeans .BA maxSteps
  outcome : B2V3PairedSourceOutcome (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal ctx.truth
    referenceAB referenceBA rThetaAB rThetaBA RejectReason B2V3ActiveSlot.outcomeKMeans maxSteps
  outcome_eq : outcome = paired.terminalize abCertificate baCertificate
    (.success abSuccess) (.success baSuccess)
  postOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
    B2V3ActiveSlot.outcomeKMeans .post → Prop
  preOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
    B2V3ActiveSlot.outcomeKMeans .pre → Prop
  rejectReason : RejectReason
  preState : PairedCandidateState (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
    (B2V3AdapterFailure RejectReason) B2V3ActiveSlot.outcomeKMeans .pre
  guard_postOK : ∀ postPair,
    outcome.toPairedCandidateState.toOption = some postPair → postOK postPair
  trace : B2V3LocalToGlobalTrace ctx omega
    (atomicGuard postOK preOK rejectReason preState outcome.toPairedCandidateState).toGuardedPool
  /-- The retained canonical record carries the literal source provenance of the witness slot. -/
  trace_outcomeKMeans_provenance : ∀ candidate,
    candidate ∈
        (atomicGuard postOK preOK rejectReason preState
          outcome.toPairedCandidateState).toGuardedPool →
      ∃ provenanceEntry ∈ (trace.canonicalRecord candidate).provenance,
        ∃ direction,
          provenanceEntry.logicalId =
            (B2V3ActiveSlot.outcomeKMeans, direction, CandidateStage.post)

namespace B2V3ActualGoodRealization

variable
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    {omega : Omega}
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {RejectReason : Type uReject}
    {Key : Type uKey} {Owner : Type uOwner}
    {provenance : B2V3OutcomeWitnessBinding Key Owner}
    {T : Nat} {maxSteps : Nat}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}

/-- All source bindings imply survival of an exact record in the current guarded pool. -/
theorem exactSurvival_of_rawGood
    (realization : B2V3ActualGoodRealization ctx omega provenance Legal RejectReason T maxSteps
      referenceAB referenceBA rThetaAB rThetaBA) :
    omega ∈ ctx.exactSurvival := by
  have hCertified : realization.outcome.BothCertified := by
    rw [realization.outcome_eq]
    refine ⟨?_, ?_⟩
    · exact ⟨realization.abSuccess.toCertifiedPayload realization.abCertificate, rfl⟩
    · exact ⟨realization.baSuccess.toCertifiedPayload realization.baCertificate, rfl⟩
  have hGuardExact := b2V3CoarseGuard_exact_of_certified realization.postOK realization.preOK
    realization.rejectReason realization.preState realization.outcome hCertified
    realization.guard_postOK
  obtain ⟨candidate, hLocal, hExact⟩ :=
    AtomicGuardResult.exists_exact_mem_toGuardedPool _ ctx.truth hGuardExact
  obtain ⟨_provenanceEntry, _hEntry, _direction, _hProvenance⟩ :=
    realization.trace_outcomeKMeans_provenance candidate hLocal
  exact realization.trace.exactSurvival candidate hLocal hExact

end B2V3ActualGoodRealization

/-- Fixed-slot adapter from the seven raw-good atoms to concrete certificate data. -/
structure B2V3ActualSourceBindingFor
    (witnessSlot : B2V3ActiveSlot)
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (CAttr : Set Omega) (T : Nat) (ledger : B2V3ActualPrimitiveLedger Omega T)
    (Key : Type uKey) (Owner : Type uOwner)
    (Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop)
    (RejectReason : Type uReject)
    (maxSteps : Nat)
    (referenceAB referenceBA : Fit) (rThetaAB rThetaBA : ℝ) where
  provenance : B2V3WitnessBinding witnessSlot Key Owner
  realize : ∀ omega, omega ∈ CAttr → ledger.rawGood omega →
    B2V3ActualGoodRealizationFor witnessSlot ctx omega provenance Legal RejectReason T maxSteps
      referenceAB referenceBA rThetaAB rThetaBA

namespace B2V3ActualSourceBindingFor

variable
    {witnessSlot : B2V3ActiveSlot}
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    {CAttr : Set Omega} {T : Nat} {ledger : B2V3ActualPrimitiveLedger Omega T}
    {Key : Type uKey} {Owner : Type uOwner}
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {RejectReason : Type uReject}
    {maxSteps : Nat}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}

/-- The fixed-slot raw-good adapter implies exact source survival. -/
theorem exactSurvival_of_rawGood
    (binding : B2V3ActualSourceBindingFor witnessSlot ctx CAttr T ledger Key Owner Legal
      RejectReason maxSteps referenceAB referenceBA rThetaAB rThetaBA)
    {omega : Omega} (hAttr : omega ∈ CAttr) (hRaw : ledger.rawGood omega) :
    omega ∈ ctx.exactSurvival :=
  (binding.realize omega hAttr hRaw).exactSurvival_of_rawGood

end B2V3ActualSourceBindingFor

/--
Source adapter from the seven raw-good atoms to a concrete, source-bound realization.  Its target
is certificate data, not the exact-survival conclusion.
-/
structure B2V3ActualSourceBinding
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (CAttr : Set Omega) (T : Nat) (ledger : B2V3ActualPrimitiveLedger Omega T)
    (Key : Type uKey) (Owner : Type uOwner)
    (Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop)
    (RejectReason : Type uReject)
    (maxSteps : Nat)
    (referenceAB referenceBA : Fit) (rThetaAB rThetaBA : ℝ) where
  provenance : B2V3OutcomeWitnessBinding Key Owner
  realize : ∀ omega, omega ∈ CAttr → ledger.rawGood omega →
    B2V3ActualGoodRealization ctx omega provenance Legal RejectReason T maxSteps referenceAB
      referenceBA rThetaAB rThetaBA

namespace B2V3ActualSourceBinding

variable
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    {CAttr : Set Omega} {T : Nat} {ledger : B2V3ActualPrimitiveLedger Omega T}
    {Key : Type uKey} {Owner : Type uOwner}
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {RejectReason : Type uReject}
    {maxSteps : Nat}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}

/-- The formerly abstract `allGoodSurvives` premise is derived from the actual adapter. -/
theorem exactSurvival_of_rawGood
    (binding : B2V3ActualSourceBinding ctx CAttr T ledger Key Owner Legal RejectReason maxSteps
      referenceAB referenceBA rThetaAB rThetaBA)
    {omega : Omega} (hAttr : omega ∈ CAttr) (hRaw : ledger.rawGood omega) :
    omega ∈ ctx.exactSurvival :=
  (binding.realize omega hAttr hRaw).exactSurvival_of_rawGood

end B2V3ActualSourceBinding

/-- The fixed-slot adapter discharges the seven-tail source-survival premise. -/
theorem measure_b2V3ActualSourceFailure_le_deltaR_of_sourceBindingFor
    {witnessSlot : B2V3ActiveSlot}
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (CAttr : Set Omega) {T : Nat} (ledger : B2V3ActualPrimitiveLedger Omega T)
    (budgets : B2V3ActualRawTailBudgets ctx.jointLaw CAttr ledger)
    {Key : Type uKey} {Owner : Type uOwner}
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {RejectReason : Type uReject}
    {maxSteps : Nat}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}
    (binding : B2V3ActualSourceBindingFor witnessSlot ctx CAttr T ledger Key Owner Legal
      RejectReason maxSteps referenceAB referenceBA rThetaAB rThetaBA) :
    ctx.jointLaw (CAttr ∩ ctx.exactSurvivalᶜ) ≤ budgets.deltaR := by
  apply measure_b2V3ActualSourceFailure_le_deltaR ctx.jointLaw CAttr ctx.exactSurvival ledger budgets
  intro omega hAttr hRaw
  exact binding.exactSurvival_of_rawGood hAttr hRaw

/-- The actual adapter discharges the survival premise of the seven-tail union bound. -/
theorem measure_b2V3ActualSourceFailure_le_deltaR_of_sourceBinding
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} [SeminormedAddCommGroup Fit]
    {Norm : Type uNorm} {Frozen : Type uFrozen} {EvaluationContext : Type uEval}
    [Nonempty (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (CAttr : Set Omega) {T : Nat} (ledger : B2V3ActualPrimitiveLedger Omega T)
    (budgets : B2V3ActualRawTailBudgets ctx.jointLaw CAttr ledger)
    {Key : Type uKey} {Owner : Type uOwner}
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {RejectReason : Type uReject}
    {maxSteps : Nat}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}
    (binding : B2V3ActualSourceBinding ctx CAttr T ledger Key Owner Legal RejectReason maxSteps
      referenceAB referenceBA rThetaAB rThetaBA) :
    ctx.jointLaw (CAttr ∩ ctx.exactSurvivalᶜ) ≤ budgets.deltaR := by
  apply measure_b2V3ActualSourceFailure_le_deltaR ctx.jointLaw CAttr ctx.exactSurvival ledger budgets
  intro omega hAttr hRaw
  exact binding.exactSurvival_of_rawGood hAttr hRaw

/-- The actual outer source selector has no approximation tolerance. -/
def b2V3ActualSelectorTolerance : ℝ := selectionTolerance 0 0 0

theorem b2V3ActualSelectorTolerance_eq_zero : b2V3ActualSelectorTolerance = 0 := by
  simp [b2V3ActualSelectorTolerance, selectionTolerance]

/-- No additional numerical selector-failure budget is charged by the verified source adapter. -/
def b2V3ActualSelectorDeltaO : ENNReal := 0

/-- Reuse of the verified canonical outer selector gives `δ_O = 0`. -/
theorem measure_b2V3ActualSelectorFailure_le_deltaO
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    {ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext}
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (hConformance : B2V3CanonicalConformance source 0 0 0) :
    ctx.jointLaw (ctx.selectorSuccess source.outputInContext 0 0 0)ᶜ ≤
      b2V3ActualSelectorDeltaO := by
  rw [measure_selectorFailure_eq_zero_of_canonicalConformance source 0 0 0 hConformance]
  rfl

/--
The corrected theorem-two chain: actual A3 exact survival is followed by the existing A2
deviation, population-gap, and actual outer-selector events.
-/
theorem b2V3ActualA3A2Chain
    {E : Nat} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uFit} {Norm : Type uNorm} {Frozen : Type uFrozen}
    {EvaluationContext : Type uEval}
    [Fintype (Fin E)] [DecidableEq (Fin E)]
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    [MeasurableSpace (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (source : B2V3CanonicalSelectorSourceAdapter ctx)
    (CAttr : Set Omega) (u : ℝ) :
    CAttr ∩ ctx.exactSurvival ∩ ctx.uniformDeviation u ∩
        ctx.strictPopulationGap u 0 0 0 ∩
        ctx.selectorSuccess source.outputInContext 0 0 0 ⊆
      ctx.exactSelected source.outputInContext := by
  exact ctx.semanticWithCoverage source.outputInContext u 0 0 0 CAttr

end BiostatFormal.HighDimensional.Multiview
