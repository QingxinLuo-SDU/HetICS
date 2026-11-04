import BiostatFormal.HighDimensional.Multiview.ProbabilityStitching
import BiostatFormal.HighDimensional.Multiview.States

/-!
# Exact-start multiview refinement and atomic guarding

This module formalizes only the strong-witness route used by Appendix A3.  It does not state a
positive-radius basin of attraction, coarse-start contraction, global convergence, or recovery
from arbitrary initialization.

The deterministic layer first proves that a typed hard failure cannot be inserted into the risk
pool: if either post direction is a failure sentinel, the atomic guard jointly falls back to the
paired pre state.  The probability layer then charges exactly four contract-failure events:

* uniform score/RSC failure;
* fit-certificate failure;
* assignment margin/regret failure;
* terminal-call contract failure.

A typed terminal hard failure satisfies the terminal-call contract only when accompanied by the
fallback certificate obtained from the atomic-guard theorem.  It is then not charged again as a
separate failure event.
-/

open MeasureTheory

namespace BiostatFormal.HighDimensional.Multiview

universe uOmega uCenter uLabel uSlot uFitParam uNormalization uFrozen
  uFailureReason uRejectReason

section TypedAtomicGuard

variable {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
  {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext -> Prop}
  {FailureReason : Type uFailureReason} {RejectReason : Type uRejectReason}
  {slot : Slot} {stage : CandidateStage}

/-- A paired state exposes a pair of legal records only when both directions succeeded. -/
def PairedCandidateState.toOption
    (state : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot stage) :
    Option (PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot stage) :=
  match state.ab, state.ba with
  | .valid ab abSlot abDirection abStage, .valid ba baSlot baDirection baStage =>
      some {
        ab := ab
        ba := ba
        ab_slot := abSlot
        ba_slot := baSlot
        ab_direction := abDirection
        ba_direction := baDirection
        ab_stage := abStage
        ba_stage := baStage
      }
  | _, _ => none

@[simp]
theorem PairedCandidateState.toOption_toPairedCandidateState
    (pair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot stage) :
    (pair.toPairedCandidateState :
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot stage).toOption = some pair := by
  rfl

/-- A successful state must be exact; a failure sentinel contains no inexact candidate. -/
def CandidateState.ExactOrFailure
    {direction : Direction}
    (state : CandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
      FailureReason slot direction stage)
    (truth : Partition Center Label) : Prop :=
  match state with
  | .valid candidate _ _ _ => Exact candidate.1.label truth
  | .failed _ => True

/-- Both directional post states are either exact successful records or failure sentinels. -/
def PairedCandidateState.ExactOrFailure
    (state : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot stage)
    (truth : Partition Center Label) : Prop :=
  state.ab.ExactOrFailure truth /\ state.ba.ExactOrFailure truth

/-- Exactness of the pair returned by an atomic guard; rejection is not exact. -/
def AtomicGuardResult.Exact
    (result : AtomicGuardResult Center Label Slot FitParam NormalizationContext FrozenContext Legal
      RejectReason slot)
    (truth : Partition Center Label) : Prop :=
  match result with
  | .acceptPost pair => pair.Exact truth
  | .fallbackPre pair => pair.Exact truth
  | .reject _ => False

/-- The joint pre fallback used by both post-rejection and post-failure paths. -/
noncomputable def atomicPreFallback
    (preOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (pre : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot .pre) :
    AtomicGuardResult Center Label Slot FitParam NormalizationContext FrozenContext Legal
      RejectReason slot := by
  classical
  exact match pre.toOption with
    | some pair => if preOK pair then .fallbackPre pair else .reject rejectReason
    | none => .reject rejectReason

/--
Atomic same-slot guard.  It accepts both post records, jointly falls back to both pre records, or
rejects the whole slot.  There is no mixed pre/post output branch.
-/
noncomputable def atomicGuard
    (postOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (pre : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot .pre)
    (post : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot .post) :
    AtomicGuardResult Center Label Slot FitParam NormalizationContext FrozenContext Legal
      RejectReason slot := by
  classical
  exact match post.toOption with
    | some pair =>
        if postOK pair then .acceptPost pair
        else atomicPreFallback preOK rejectReason pre
    | none => atomicPreFallback preOK rejectReason pre

/-- If either post direction failed, an admissible paired pre state is jointly restored. -/
theorem atomicGuard_eq_fallbackPre_of_post_failure
    (postOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (prePair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot .pre)
    (post : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot .post)
    (hPostFailure : post.toOption = none)
    (hPreOK : preOK prePair) :
    atomicGuard postOK preOK rejectReason
        (prePair.toPairedCandidateState :
          PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
            FailureReason slot .pre)
        post = .fallbackPre prePair := by
  classical
  simp [atomicGuard, atomicPreFallback, hPostFailure, hPreOK]

/-- Successful extraction from an exact-or-failure paired state yields an exact pair. -/
theorem PairedCandidateState.exact_of_toOption_eq_some
    (state : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot stage)
    (truth : Partition Center Label)
    (pair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot stage)
    (hSafe : state.ExactOrFailure truth)
    (hSome : state.toOption = some pair) :
    pair.Exact truth := by
  cases state with
  | mk ab ba =>
      cases ab <;> cases ba <;>
        simp [PairedCandidateState.toOption, PairedCandidateState.ExactOrFailure,
          CandidateState.ExactOrFailure, PairedRiskCandidates.Exact] at hSome hSafe ⊢
      subst pair
      exact hSafe

/--
Typed atomic-guard safety.  Starting from an exact legal paired pre state, an exact successful post
is accepted and every typed hard failure falls back to the exact pre pair.  This theorem is purely
exact-start safety; it is not a contraction or attraction-basin theorem.
-/
theorem atomicGuard_exact_of_exact_pre_of_post_exactOrFailure
    (postOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK :
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (prePair : PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext
      Legal slot .pre)
    (post : PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext
      Legal FailureReason slot .post)
    (truth : Partition Center Label)
    (hPreExact : prePair.Exact truth)
    (hPreOK : preOK prePair)
    (hPostSafe : post.ExactOrFailure truth) :
    (atomicGuard postOK preOK rejectReason
      (prePair.toPairedCandidateState :
        PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
          FailureReason slot .pre)
      post).Exact truth := by
  classical
  cases hPostOption : post.toOption with
  | none =>
      rw [atomicGuard_eq_fallbackPre_of_post_failure postOK preOK rejectReason prePair post
        hPostOption hPreOK]
      exact hPreExact
  | some postPair =>
      have hPostExact : postPair.Exact truth :=
        post.exact_of_toOption_eq_some truth postPair hPostSafe hPostOption
      by_cases hPostOK : postOK postPair
      · simp [atomicGuard, hPostOption, hPostOK, AtomicGuardResult.Exact, hPostExact]
      · simp [atomicGuard, hPostOption, hPostOK, atomicPreFallback, hPreOK,
          AtomicGuardResult.Exact, hPreExact]

/-- The realized guarded pool contains both records of the accepted or fallback pair. -/
def AtomicGuardResult.toGuardedPool
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    (result : AtomicGuardResult Center Label Slot FitParam NormalizationContext FrozenContext Legal
      RejectReason slot) :
    GuardedCandidatePool
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal) :=
  match result with
  | .acceptPost pair => {pair.ab, pair.ba}
  | .fallbackPre pair => {pair.ab, pair.ba}
  | .reject _ => ∅

/-- Every exact atomic-guard result contributes an exact record to its guarded pool. -/
theorem AtomicGuardResult.exists_exact_mem_toGuardedPool
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    (result : AtomicGuardResult Center Label Slot FitParam NormalizationContext FrozenContext Legal
      RejectReason slot)
    (truth : Partition Center Label)
    (hExact : result.Exact truth) :
    exists candidate, candidate ∈ result.toGuardedPool /\
      Multiview.Exact candidate.1.label truth := by
  cases result with
  | acceptPost pair =>
      exact ⟨pair.ab, by simp [AtomicGuardResult.toGuardedPool], hExact.1⟩
  | fallbackPre pair =>
      exact ⟨pair.ab, by simp [AtomicGuardResult.toGuardedPool], hExact.1⟩
  | reject reason =>
      exact False.elim hExact

end TypedAtomicGuard

section CallLedger

universe uCall uParameter uCandidate

/-- A finite ledger entry identifies an ordinary or terminal fit/assignment call. -/
inductive RefinementCallKind (iterations : Nat) where
  | fit (iteration : Fin iterations)
  | assignment (iteration : Fin iterations)
  | terminalFit
  | terminalAssignment
  deriving DecidableEq, Fintype, Repr

/-- Same-slot and fixed-direction identity carried by every ledger call. -/
structure RefinementCall (Slot : Type uSlot) (iterations : Nat) where
  slot : Slot
  direction : Direction
  kind : RefinementCallKind iterations
  deriving DecidableEq, Repr

/-- The finite, realized collection of fit/assignment/terminal calls. -/
abbrev RefinementCallLedger (Slot : Type uSlot) (iterations : Nat) :=
  Finset (RefinementCall Slot iterations)

/-- Uniform contract event over every entry in a finite call ledger. -/
def uniformCallContractEvent
    {Omega : Type uOmega} {Call : Type uCall}
    (ledger : Finset Call) (contract : Call -> Omega -> Prop) : Set Omega :=
  {omega | ∀ call ∈ ledger, contract call omega}

/--
Complete `K = 3` reference margin.  It quantifies over every center and both wrong-class
competitors; a binary or averaged contrast does not inhabit this contract.
-/
structure Fin3AssignmentMargin
    (Center : Type uCenter) (Parameter : Type uParameter)
    (truth : Partition Center (Fin 3))
    (score : Center -> Fin 3 -> Parameter -> Real)
    (reference : Parameter) (gamma0 : Real) : Prop where
  gamma0_positive : 0 < gamma0
  true_minus_wrong : ∀ center wrongClass,
    wrongClass ≠ truth center ->
      gamma0 <= score center (truth center) reference - score center wrongClass reference

/--
Auditable total-objective regret certificate for a constrained `K = 3` assignment.  The bound is
against every feasible assignment, not merely a coordinatewise or one-vs-rest optimum.
-/
structure Fin3AssignmentRegretCertificate
    (Center : Type uCenter) [Fintype Center]
    (feasible : Partition Center (Fin 3) -> Prop)
    (objective : Partition Center (Fin 3) -> Real)
    (returned : Partition Center (Fin 3)) (etaAssignment : Real) : Prop where
  eta_nonnegative : 0 <= etaAssignment
  returned_feasible : feasible returned
  global_regret : ∀ candidate, feasible candidate ->
    objective candidate - objective returned <= etaAssignment

/-- Auditable optimization-error certificate for one fit return. -/
structure FitCertificate
    (Parameter : Type uParameter) (objective : Parameter -> Real)
    (reference estimate : Parameter) (epsilonFit : Real) : Prop where
  epsilon_nonnegative : 0 <= epsilonFit
  residual_bound : objective estimate <= objective reference + epsilonFit

/--
A terminal success binds the reassignment to the exact parameter returned by the terminal refit.
The relation may additionally encode normalization and frozen-context equality.
-/
structure TerminalSuccess
    (Parameter : Type uParameter) (Candidate : Type uCandidate)
    (assignedFrom : Parameter -> Candidate -> Prop) where
  parameter : Parameter
  candidate : Candidate
  reassignment_uses_returned_parameter : assignedFrom parameter candidate

/-- A terminal call either returns a bound success payload or a typed hard-failure sentinel. -/
inductive TerminalCallResult
    (Parameter : Type uParameter) (Candidate : Type uCandidate)
    (FailureReason : Type uFailureReason)
    (assignedFrom : Parameter -> Candidate -> Prop) where
  | success (payload : TerminalSuccess Parameter Candidate assignedFrom)
  | failed (reason : FailureReason)

/--
Terminal-call contract.  A successful call must satisfy its supplied certificate.  A hard failure
is contract-safe only when the caller supplies a fallback certificate; it is not free merely
because it uses the `.failed` constructor.
-/
def TerminalCallResult.ContractHolds
    {Parameter : Type uParameter} {Candidate : Type uCandidate}
    {FailureReason : Type uFailureReason}
    {assignedFrom : Parameter -> Candidate -> Prop}
    (successCertificate : TerminalSuccess Parameter Candidate assignedFrom -> Prop)
    (failureFallbackCertificate : FailureReason -> Prop) :
    TerminalCallResult Parameter Candidate FailureReason assignedFrom -> Prop
  | .success payload => successCertificate payload
  | .failed reason => failureFallbackCertificate reason

@[simp]
theorem TerminalCallResult.contractHolds_failed
    {Parameter : Type uParameter} {Candidate : Type uCandidate}
    {FailureReason : Type uFailureReason}
    {assignedFrom : Parameter -> Candidate -> Prop}
    (successCertificate : TerminalSuccess Parameter Candidate assignedFrom -> Prop)
    (failureFallbackCertificate : FailureReason -> Prop)
    (reason : FailureReason)
    (hFallback : failureFallbackCertificate reason) :
    (TerminalCallResult.failed reason :
      TerminalCallResult Parameter Candidate FailureReason assignedFrom).ContractHolds
        successCertificate failureFallbackCertificate := by
  exact hFallback

/--
Uniform terminal-call event.  A typed hard failure belongs only through its supplied fallback
certificate, which the strong-witness client obtains from the atomic guard theorem.
-/
def uniformTerminalCallContractEvent
    {Omega : Type uOmega} {Call : Type uCall}
    {Parameter : Type uParameter} {Candidate : Type uCandidate}
    {FailureReason : Type uFailureReason}
    {assignedFrom : Parameter -> Candidate -> Prop}
    (ledger : Finset Call)
    (result : Call -> Omega ->
      TerminalCallResult Parameter Candidate FailureReason assignedFrom)
    (successCertificate : Call -> Omega ->
      TerminalSuccess Parameter Candidate assignedFrom -> Prop)
    (failureFallbackCertificate : Call -> Omega -> FailureReason -> Prop) : Set Omega :=
  uniformCallContractEvent ledger fun call omega =>
    (result call omega).ContractHolds (successCertificate call omega)
      (failureFallbackCertificate call omega)

end CallLedger

section StrongWitnessProbability

variable {Omega : Type uOmega}
  {Center : Type uCenter} {Label : Type uLabel} {Slot : Type uSlot}
  {FitParam : Type uFitParam} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center Label Slot FitParam NormalizationContext FrozenContext -> Prop}
  {FailureReason : Type uFailureReason} {RejectReason : Type uRejectReason}
  {slot : Slot}

/--
The four, and only four, A3 cross-failure sources charged on strong-witness coverage.  Coverage
itself is a premise supplied by A1 and is not charged again inside `deltaR`.
-/
def refinementFailureSourceUnion
    (coverage scoreRSC fitCertificate assignmentContract terminalContract : Set Omega) :
    Set Omega :=
  (coverage ∩ scoreRSCᶜ) ∪ (coverage ∩ fitCertificateᶜ) ∪
    (coverage ∩ assignmentContractᶜ) ∪ (coverage ∩ terminalContractᶜ)

/-- Sum of the four A3 failure budgets. -/
def refinementFailureBudget
    (deltaScoreRSC deltaFit deltaAssignment deltaTerminal : ENNReal) : ENNReal :=
  deltaScoreRSC + deltaFit + deltaAssignment + deltaTerminal

/-- Realized guarded pool for one strong exact witness. -/
noncomputable def strongWitnessGuardedPool
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    (prePair : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre)
    (post : Omega ->
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (postOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : Omega -> RejectReason)
    (omega : Omega) :
    GuardedCandidatePool
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal) :=
  (atomicGuard (postOK omega) (preOK omega) (rejectReason omega)
    ((prePair omega).toPairedCandidateState :
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot .pre)
    (post omega)).toGuardedPool

/-- Event that the strong witness contributes an exact record to the guarded pool. -/
def strongWitnessRefinementEvent
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    (truth : Partition Center Label)
    (prePair : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre)
    (post : Omega ->
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (postOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : Omega -> RejectReason) : Set Omega :=
  {omega | exists candidate,
    candidate ∈ strongWitnessGuardedPool prePair post postOK preOK rejectReason omega /\
    Exact candidate.1.label truth}

/--
Deterministic strong-witness implication used by the probability theorem.  The four contracts only
have to imply exact-or-failure post safety; typed failure sentinels are then discharged by the
atomic guard theorem rather than by a fifth probability charge.
-/
theorem strongExactWitness_mem_refinementEvent
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    (truth : Partition Center Label)
    (coverage scoreRSC fitCertificate assignmentContract terminalContract : Set Omega)
    (prePair : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre)
    (post : Omega ->
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (postOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : Omega -> RejectReason)
    (hPreExact : ∀ omega ∈ coverage, (prePair omega).Exact truth)
    (hPreOK : ∀ omega ∈ coverage, preOK omega (prePair omega))
    (hPostSafe : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentContract ->
      omega ∈ terminalContract -> (post omega).ExactOrFailure truth)
    {omega : Omega}
    (hCoverage : omega ∈ coverage)
    (hScoreRSC : omega ∈ scoreRSC)
    (hFit : omega ∈ fitCertificate)
    (hAssignment : omega ∈ assignmentContract)
    (hTerminal : omega ∈ terminalContract) :
    omega ∈ strongWitnessRefinementEvent truth prePair post postOK preOK rejectReason := by
  have hGuardExact := atomicGuard_exact_of_exact_pre_of_post_exactOrFailure
    (postOK omega) (preOK omega) (rejectReason omega) (prePair omega) (post omega) truth
    (hPreExact omega hCoverage) (hPreOK omega hCoverage)
    (hPostSafe omega hCoverage hScoreRSC hFit hAssignment hTerminal)
  exact AtomicGuardResult.exists_exact_mem_toGuardedPool _ truth hGuardExact

/--
Set-level A3 interface: on strong-witness coverage, refinement failure is contained in the union
of the four declared contract failures.
-/
theorem strongExactWitness_refinementFailure_subset
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    (truth : Partition Center Label)
    (coverage scoreRSC fitCertificate assignmentContract terminalContract : Set Omega)
    (prePair : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre)
    (post : Omega ->
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (postOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : Omega -> RejectReason)
    (hPreExact : ∀ omega ∈ coverage, (prePair omega).Exact truth)
    (hPreOK : ∀ omega ∈ coverage, preOK omega (prePair omega))
    (hPostSafe : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentContract ->
      omega ∈ terminalContract -> (post omega).ExactOrFailure truth) :
    coverage ∩
        (strongWitnessRefinementEvent truth prePair post postOK preOK rejectReason)ᶜ ⊆
      refinementFailureSourceUnion coverage scoreRSC fitCertificate assignmentContract
        terminalContract := by
  intro omega hFailure
  by_cases hScoreRSC : omega ∈ scoreRSC
  · by_cases hFit : omega ∈ fitCertificate
    · by_cases hAssignment : omega ∈ assignmentContract
      · by_cases hTerminal : omega ∈ terminalContract
        · exact False.elim (hFailure.2 (strongExactWitness_mem_refinementEvent truth coverage
            scoreRSC fitCertificate assignmentContract terminalContract prePair post postOK preOK
            rejectReason hPreExact hPreOK hPostSafe hFailure.1 hScoreRSC hFit hAssignment
            hTerminal))
        · simp [refinementFailureSourceUnion, hFailure.1, hTerminal]
      · simp [refinementFailureSourceUnion, hFailure.1, hAssignment]
    · simp [refinementFailureSourceUnion, hFailure.1, hFit]
  · simp [refinementFailureSourceUnion, hFailure.1, hScoreRSC]

/-- Four-event union bound for the A3 refinement failure source union. -/
theorem measure_refinementFailureSourceUnion_le
    [MeasurableSpace Omega]
    (mu : Measure Omega)
    (coverage scoreRSC fitCertificate assignmentContract terminalContract : Set Omega)
    {deltaScoreRSC deltaFit deltaAssignment deltaTerminal : ENNReal}
    (hScoreRSC : mu (coverage ∩ scoreRSCᶜ) <= deltaScoreRSC)
    (hFit : mu (coverage ∩ fitCertificateᶜ) <= deltaFit)
    (hAssignment : mu (coverage ∩ assignmentContractᶜ) <= deltaAssignment)
    (hTerminal : mu (coverage ∩ terminalContractᶜ) <= deltaTerminal) :
    mu (refinementFailureSourceUnion coverage scoreRSC fitCertificate assignmentContract
      terminalContract) <=
        refinementFailureBudget deltaScoreRSC deltaFit deltaAssignment deltaTerminal := by
  calc
    mu (refinementFailureSourceUnion coverage scoreRSC fitCertificate assignmentContract
        terminalContract) <=
        mu (coverage ∩ scoreRSCᶜ) + mu (coverage ∩ fitCertificateᶜ) +
          mu (coverage ∩ assignmentContractᶜ) + mu (coverage ∩ terminalContractᶜ) := by
      unfold refinementFailureSourceUnion
      calc
        mu ((coverage ∩ scoreRSCᶜ) ∪ (coverage ∩ fitCertificateᶜ) ∪
            (coverage ∩ assignmentContractᶜ) ∪ (coverage ∩ terminalContractᶜ)) <=
            mu ((coverage ∩ scoreRSCᶜ) ∪ (coverage ∩ fitCertificateᶜ) ∪
              (coverage ∩ assignmentContractᶜ)) +
                mu (coverage ∩ terminalContractᶜ) := measure_union_le _ _
        _ <= (mu ((coverage ∩ scoreRSCᶜ) ∪ (coverage ∩ fitCertificateᶜ)) +
              mu (coverage ∩ assignmentContractᶜ)) +
                mu (coverage ∩ terminalContractᶜ) := by
          gcongr
          exact measure_union_le _ _
        _ <= ((mu (coverage ∩ scoreRSCᶜ) + mu (coverage ∩ fitCertificateᶜ)) +
              mu (coverage ∩ assignmentContractᶜ)) +
                mu (coverage ∩ terminalContractᶜ) := by
          gcongr
          exact measure_union_le _ _
    _ <= refinementFailureBudget deltaScoreRSC deltaFit deltaAssignment deltaTerminal :=
      add_le_add (add_le_add (add_le_add hScoreRSC hFit) hAssignment) hTerminal

/--
Final A3 probability interface for the strong exact witness:
`P(C ∩ Rᶜ) ≤ deltaR`.  No independence is assumed.  There is no separate charge for typed hard
failure because `atomicGuard_eq_fallbackPre_of_post_failure` proves exact-pre fallback first.
-/
theorem measure_strongExactWitness_refinementFailure_le
    [DecidableEq
      (RiskCandidate Center Label Slot FitParam NormalizationContext FrozenContext Legal)]
    [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    (truth : Partition Center Label)
    (coverage scoreRSC fitCertificate assignmentContract terminalContract : Set Omega)
    (prePair : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre)
    (post : Omega ->
      PairedCandidateState Center Label Slot FitParam NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (postOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK : Omega ->
      PairedRiskCandidates Center Label Slot FitParam NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : Omega -> RejectReason)
    {deltaScoreRSC deltaFit deltaAssignment deltaTerminal deltaR : ENNReal}
    (hPreExact : ∀ omega ∈ coverage, (prePair omega).Exact truth)
    (hPreOK : ∀ omega ∈ coverage, preOK omega (prePair omega))
    (hPostSafe : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentContract ->
      omega ∈ terminalContract -> (post omega).ExactOrFailure truth)
    (hScoreRSC : mu (coverage ∩ scoreRSCᶜ) <= deltaScoreRSC)
    (hFit : mu (coverage ∩ fitCertificateᶜ) <= deltaFit)
    (hAssignment : mu (coverage ∩ assignmentContractᶜ) <= deltaAssignment)
    (hTerminal : mu (coverage ∩ terminalContractᶜ) <= deltaTerminal)
    (hBudget :
      refinementFailureBudget deltaScoreRSC deltaFit deltaAssignment deltaTerminal <= deltaR) :
    mu (coverage ∩
      (strongWitnessRefinementEvent truth prePair post postOK preOK rejectReason)ᶜ) <= deltaR := by
  calc
    mu (coverage ∩
        (strongWitnessRefinementEvent truth prePair post postOK preOK rejectReason)ᶜ) <=
        mu (refinementFailureSourceUnion coverage scoreRSC fitCertificate assignmentContract
          terminalContract) :=
      measure_mono (strongExactWitness_refinementFailure_subset truth coverage scoreRSC
        fitCertificate assignmentContract terminalContract prePair post postOK preOK rejectReason
        hPreExact hPreOK hPostSafe)
    _ <= refinementFailureBudget deltaScoreRSC deltaFit deltaAssignment deltaTerminal :=
      measure_refinementFailureSourceUnion_le mu coverage scoreRSC fitCertificate
        assignmentContract terminalContract hScoreRSC hFit hAssignment hTerminal
    _ <= deltaR := hBudget

end StrongWitnessProbability

end BiostatFormal.HighDimensional.Multiview
