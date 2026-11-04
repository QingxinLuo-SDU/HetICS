import BiostatFormal.HighDimensional.Multiview.B2V3CoarseRefinement
import BiostatFormal.HighDimensional.Multiview.A3Terminal
import BiostatFormal.HighDimensional.Multiview.B2V3Connections

/-!
# Source-faithful B2-v3 refinement adapter

This module maps every realized frozen-v3 refinement branch to a mathematical output.  Ordinary
eligible maximum-iteration output remains an ordinary status only.  Terminal maximum-iteration
output is a typed failure and carries no parameter or label payload.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uOmega uCenter uParameter uSlot uNormalization uFrozen uReject

/-- Failures emitted before or during one ordinary fixed-label fit. -/
inductive B2V3FitFailure where
  | noFeasibleStart
  | mStepBacktrackingFailed
  | nonfiniteObjective
  deriving DecidableEq, Fintype, Repr

/-- The constrained three-class assignment has one source failure status. -/
inductive B2V3AssignmentFailure where
  | assignmentInfeasible
  deriving DecidableEq, Fintype, Repr

/-- Exhaustive failure reasons exported by the candidate and terminal source paths. -/
inductive B2V3PipelineFailure where
  | fit (reason : B2V3FitFailure)
  | assignment (reason : B2V3AssignmentFailure)
  | monotonicityViolation
  | noEligibleCandidate
  | fullRefitFailed
  | fullRefitMaxIterUncertified
  deriving DecidableEq, Repr

/-- The two eligible ordinary-loop statuses.  Neither is a terminal correctness certificate. -/
inductive B2V3OrdinaryStatus where
  | converged
  | maxOuterIterEligible
  deriving DecidableEq, Fintype, Repr

/-- Frozen-v3 terminal statuses after the fail-closed maximum-iteration correction. -/
inductive B2V3TerminalStatus where
  | success
  | fullRefitFailed
  | fullRefitMaxIterUncertified
  deriving DecidableEq, Fintype, Repr

/-- One source call returns a payload, a typed rejection, or a typed failure—never a mixed value. -/
inductive B2V3DirectionalSourceOutcome
    (Success : Type*) (Failure : Type*) (RejectReason : Type*) where
  | accepted (payload : Success)
  | rejected (reason : RejectReason)
  | failed (failure : Failure)

/-- A finite ordinary-loop success retains its source status and stopping witness. -/
structure B2V3OrdinarySuccess (State : Type*) (maxSteps : ℕ) where
  post : State
  status : B2V3OrdinaryStatus
  stop : FiniteStopWitness maxSteps

/-- A source failure retains a reason and finite stopping metadata but no successful payload. -/
structure B2V3TypedSourceFailure (maxSteps : ℕ) where
  reason : B2V3PipelineFailure
  stop : FiniteStopWitness maxSteps

/-- Every branch returned by one actual ordinary CEM iteration. -/
inductive B2V3OuterStepBranch (State : Type*) (maxSteps : ℕ) where
  | fitFailed (reason : B2V3FitFailure) (stop : FiniteStopWitness maxSteps)
  | assignmentFailed (reason : B2V3AssignmentFailure) (stop : FiniteStopWitness maxSteps)
  | monotonicityViolation (stop : FiniteStopWitness maxSteps)
  | converged (post : State) (stop : FiniteStopWitness maxSteps)
  | maxOuterIterEligible (post : State) (stop : FiniteStopWitness maxSteps)

/-- Candidate selection either returns one eligible result or reports that none exists. -/
inductive B2V3CandidateSelectionBranch (State : Type*) (maxSteps : ℕ) where
  | selected (result : B2V3OrdinarySuccess State maxSteps)
  | noEligibleCandidate (stop : FiniteStopWitness maxSteps)

/-- Terminal refit has exactly one payload branch and two payload-free failure branches. -/
inductive B2V3TerminalBranch (Payload : Type*) (maxSteps : ℕ) where
  | success (payload : Payload)
  | fullRefitFailed (stop : FiniteStopWitness maxSteps)
  | fullRefitMaxIterUncertified (stop : FiniteStopWitness maxSteps)

/-- The single frozen-v3 control-flow map. -/
inductive B2V3SourceMap where
  | v3

namespace B2V3SourceMap

/-- Exhaustive mathematical map for one ordinary CEM step. -/
def outerStep
    (source : B2V3SourceMap)
    {State RejectReason : Type*} {maxSteps : ℕ}
    (branch : B2V3OuterStepBranch State maxSteps) :
    B2V3DirectionalSourceOutcome (B2V3OrdinarySuccess State maxSteps)
      (B2V3TypedSourceFailure maxSteps) RejectReason :=
  match source, branch with
  | .v3, .fitFailed reason stop => .failed ⟨.fit reason, stop⟩
  | .v3, .assignmentFailed reason stop => .failed ⟨.assignment reason, stop⟩
  | .v3, .monotonicityViolation stop => .failed ⟨.monotonicityViolation, stop⟩
  | .v3, .converged post stop => .accepted ⟨post, .converged, stop⟩
  | .v3, .maxOuterIterEligible post stop =>
      .accepted ⟨post, .maxOuterIterEligible, stop⟩

/-- Exhaustive mathematical map for eligible-candidate selection. -/
def candidateSelection
    (source : B2V3SourceMap)
    {State RejectReason : Type*} {maxSteps : ℕ}
    (branch : B2V3CandidateSelectionBranch State maxSteps) :
    B2V3DirectionalSourceOutcome (B2V3OrdinarySuccess State maxSteps)
      (B2V3TypedSourceFailure maxSteps) RejectReason :=
  match source, branch with
  | .v3, .selected result => .accepted result
  | .v3, .noEligibleCandidate stop => .failed ⟨.noEligibleCandidate, stop⟩

/--
Exhaustive terminal map.  Only `.success` carries `Payload`; both failure statuses are fail-closed.
-/
def terminal
    (source : B2V3SourceMap)
    {Payload RejectReason : Type*} {maxSteps : ℕ}
    (branch : B2V3TerminalBranch Payload maxSteps) :
    B2V3DirectionalSourceOutcome Payload (B2V3TypedSourceFailure maxSteps) RejectReason :=
  match source, branch with
  | .v3, .success payload => .accepted payload
  | .v3, .fullRefitFailed stop => .failed ⟨.fullRefitFailed, stop⟩
  | .v3, .fullRefitMaxIterUncertified stop =>
      .failed ⟨.fullRefitMaxIterUncertified, stop⟩

end B2V3SourceMap

/--
A successful terminal payload binds the frozen outer labels and terminal parameter to the exact
legal post record exported to the guard.  Frozen v3 performs no assignment after this refit.
-/
structure B2V3CertifiedTerminalPayload
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [PseudoMetricSpace Parameter]
    (Slot : Type uSlot) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop)
    (truth : Partition Center (Fin 3))
    (reference : Parameter) (rTheta : ℝ)
    (slot : Slot) (direction : Direction) (maxSteps : ℕ)
    (status : B2V3TerminalStatus) where
  frozenOuterLabels : Partition Center (Fin 3)
  terminalParameter : Parameter
  post : RiskCandidate Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
  post_slot : post.1.slot = slot
  post_direction : post.1.direction = direction
  post_stage : post.1.context.stage = .post
  post_label_eq_frozen : post.1.label = frozenOuterLabels
  post_parameter_eq_terminal : post.1.context.fitParameter = terminalParameter
  outerLabels_exact : Exact frozenOuterLabels truth
  parameter_safe : dist terminalParameter reference ≤ rTheta
  sourceStatus_eq : status = .success
  stop : FiniteStopWitness maxSteps

namespace B2V3CertifiedTerminalPayload

/-- The exact post is the same label payload exported to the atomic guard. -/
theorem post_exact
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [PseudoMetricSpace Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
    {truth : Partition Center (Fin 3)} {reference : Parameter} {rTheta : ℝ}
    {slot : Slot} {direction : Direction} {maxSteps : ℕ} {status : B2V3TerminalStatus}
    (payload : B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext
      FrozenContext Legal truth reference rTheta slot direction maxSteps status) :
    Exact payload.post.1.label truth := by
  rw [payload.post_label_eq_frozen]
  exact payload.outerLabels_exact

/-- Optional downstream reassignment must consume this very terminal parameter. -/
def SameParameterAssignment
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [PseudoMetricSpace Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
    {truth : Partition Center (Fin 3)} {reference : Parameter} {rTheta : ℝ}
    {slot : Slot} {direction : Direction} {maxSteps : ℕ} {status : B2V3TerminalStatus}
    (payload : B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext
      FrozenContext Legal truth reference rTheta slot direction maxSteps status)
    (assignedFrom : Parameter → Partition Center (Fin 3) → Prop) : Prop :=
  assignedFrom payload.terminalParameter payload.post.1.label

end B2V3CertifiedTerminalPayload

/-- Failure state used when a source failure or guard rejection is converted to `CandidateState`. -/
inductive B2V3AdapterFailure (RejectReason : Type uReject) where
  | source (reason : B2V3PipelineFailure)
  | guardRejected (reason : RejectReason)

/-- One slot's AB/BA terminal source outcomes with distinct directional payload identities. -/
structure B2V3PairedSourceOutcome
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [PseudoMetricSpace Parameter]
    (Slot : Type uSlot) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop)
    (truth : Partition Center (Fin 3))
    (referenceAB referenceBA : Parameter) (rThetaAB rThetaBA : ℝ)
    (RejectReason : Type uReject) (slot : Slot) (maxSteps : ℕ) where
  ab : B2V3DirectionalSourceOutcome
    (B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext FrozenContext Legal
      truth referenceAB rThetaAB slot .AB maxSteps .success)
    (B2V3TypedSourceFailure maxSteps) RejectReason
  ba : B2V3DirectionalSourceOutcome
    (B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext FrozenContext Legal
      truth referenceBA rThetaBA slot .BA maxSteps .success)
    (B2V3TypedSourceFailure maxSteps) RejectReason

namespace B2V3PairedSourceOutcome

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [PseudoMetricSpace Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
  {truth : Partition Center (Fin 3)}
  {referenceAB referenceBA : Parameter} {rThetaAB rThetaBA : ℝ}
  {RejectReason : Type uReject} {slot : Slot} {maxSteps : ℕ}

/-- Both directions reached the payload-carrying terminal-success constructor. -/
def BothCertified
    (outcome : B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext
      Legal truth referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps) : Prop :=
  (∃ payload, outcome.ab = .accepted payload) ∧
    (∃ payload, outcome.ba = .accepted payload)

private def directionalState
    {direction : Direction} {reference : Parameter} {rTheta : ℝ}
    (outcome : B2V3DirectionalSourceOutcome
      (B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext FrozenContext Legal
        truth reference rTheta slot direction maxSteps .success)
      (B2V3TypedSourceFailure maxSteps) RejectReason) :
    CandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
      (B2V3AdapterFailure RejectReason) slot direction .post :=
  match outcome with
  | .accepted payload =>
      .valid payload.post payload.post_slot payload.post_direction payload.post_stage
  | .failed failure => .failed (.source failure.reason)
  | .rejected reason => .failed (.guardRejected reason)

/-- Convert both actual source outputs to the existing typed paired post state. -/
def toPairedCandidateState
    (outcome : B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext
      Legal truth referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps) :
    PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
      (B2V3AdapterFailure RejectReason) slot .post where
  ab := directionalState outcome.ab
  ba := directionalState outcome.ba

/-- Every accepted terminal payload is exact; all other branches become failure sentinels. -/
theorem exactOrFailure
    (outcome : B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext
      Legal truth referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps) :
    outcome.toPairedCandidateState.ExactOrFailure truth := by
  cases outcome with
  | mk ab ba =>
      cases ab with
      | accepted abPayload =>
          cases ba with
          | accepted baPayload => exact ⟨abPayload.post_exact, baPayload.post_exact⟩
          | rejected reason => exact ⟨abPayload.post_exact, trivial⟩
          | failed failure => exact ⟨abPayload.post_exact, trivial⟩
      | rejected reason =>
          cases ba with
          | accepted baPayload => exact ⟨trivial, baPayload.post_exact⟩
          | rejected reason => exact ⟨trivial, trivial⟩
          | failed failure => exact ⟨trivial, trivial⟩
      | failed failure =>
          cases ba with
          | accepted baPayload => exact ⟨trivial, baPayload.post_exact⟩
          | rejected reason => exact ⟨trivial, trivial⟩
          | failed failure => exact ⟨trivial, trivial⟩

end B2V3PairedSourceOutcome

/--
Coarse-start guard theorem.  Because both certified posts are present and pass `postOK`, the guard
accepts them; no exactness of the coarse pre state is assumed.
-/
theorem b2V3CoarseGuard_exact_of_certified
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [PseudoMetricSpace Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
    {truth : Partition Center (Fin 3)}
    {referenceAB referenceBA : Parameter} {rThetaAB rThetaBA : ℝ}
    {RejectReason : Type uReject} {slot : Slot} {maxSteps : ℕ}
    (postOK : PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext
      FrozenContext Legal slot .post → Prop)
    (preOK : PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext
      FrozenContext Legal slot .pre → Prop)
    (rejectReason : RejectReason)
    (preState : PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext
      FrozenContext Legal (B2V3AdapterFailure RejectReason) slot .pre)
    (outcome : B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext
      Legal truth referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps)
    (hCertified : outcome.BothCertified)
    (hPostOK : ∀ postPair,
      outcome.toPairedCandidateState.toOption = some postPair → postOK postPair) :
    (atomicGuard postOK preOK rejectReason preState outcome.toPairedCandidateState).Exact truth := by
  rcases hCertified with ⟨⟨abPayload, hAB⟩, ⟨baPayload, hBA⟩⟩
  cases outcome with
  | mk ab ba =>
      cases hAB
      cases hBA
      let postPair : PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext
          FrozenContext Legal slot .post := {
        ab := abPayload.post
        ba := baPayload.post
        ab_slot := abPayload.post_slot
        ba_slot := baPayload.post_slot
        ab_direction := abPayload.post_direction
        ba_direction := baPayload.post_direction
        ab_stage := abPayload.post_stage
        ba_stage := baPayload.post_stage
      }
      let certifiedOutcome :
          B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
            truth referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps := {
        ab := .accepted abPayload
        ba := .accepted baPayload
      }
      have hSome :
          certifiedOutcome.toPairedCandidateState.toOption = some postPair := rfl
      have hOK : postOK postPair := by
        apply hPostOK postPair
        simpa [certifiedOutcome] using hSome
      simp [atomicGuard, hSome, hOK, AtomicGuardResult.Exact,
        PairedRiskCandidates.Exact, postPair, certifiedOutcome,
        abPayload.post_exact, baPayload.post_exact]

/--
The exact candidate extracted from the same atomic guard pool is transported through the actual
local-to-global trace; no unrelated exact witness is introduced.
-/
theorem b2V3CoarseExactSurvival
    {E : ℕ} {Omega : Type uOmega} [MeasurableSpace Omega]
    {Fit : Type uParameter} [SeminormedAddCommGroup Fit]
    {Norm : Type uNormalization} {Frozen : Type uFrozen} {EvaluationContext : Type*}
    [DecidableEq (B2V3RecoveryRecord (Fin E) Fit Norm Frozen)]
    (ctx : B2V3RecoveryContext Omega (Fin E) Fit Norm Frozen EvaluationContext)
    (omega : Omega)
    {Legal : B2V3RecoveryRecord (Fin E) Fit Norm Frozen → Prop}
    {referenceAB referenceBA : Fit} {rThetaAB rThetaBA : ℝ}
    {RejectReason : Type uReject} {slot : B2V3ActiveSlot} {maxSteps : ℕ}
    (postOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      slot .post → Prop)
    (preOK : PairedRiskCandidates (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      slot .pre → Prop)
    (rejectReason : RejectReason)
    (preState : PairedCandidateState (Fin E) (Fin 3) B2V3ActiveSlot Fit Norm Frozen Legal
      (B2V3AdapterFailure RejectReason) slot .pre)
    (outcome : B2V3PairedSourceOutcome (Fin E) Fit B2V3ActiveSlot Norm Frozen Legal ctx.truth
      referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps)
    (trace : B2V3LocalToGlobalTrace ctx omega
      (atomicGuard postOK preOK rejectReason preState
        outcome.toPairedCandidateState).toGuardedPool)
    (hCertified : outcome.BothCertified)
    (hPostOK : ∀ postPair,
      outcome.toPairedCandidateState.toOption = some postPair → postOK postPair) :
    omega ∈ ctx.exactSurvival := by
  have hGuardExact := b2V3CoarseGuard_exact_of_certified postOK preOK rejectReason preState
    outcome hCertified hPostOK
  obtain ⟨candidate, hLocal, hExact⟩ :=
    AtomicGuardResult.exists_exact_mem_toGuardedPool _ ctx.truth hGuardExact
  exact trace.exactSurvival candidate hLocal hExact

section AdaptivePrimitiveLedger

open MeasureTheory

/-- The four primitive failure families charged by coarse-start A3. -/
inductive B2V3PrimitiveFailureKind where
  | scoreRSC
  | fit
  | assignment
  | terminal
  deriving DecidableEq, Fintype, Repr

/--
A finite ordered ledger of realized, potentially history-adaptive calls.  The event at call `q` may
depend on the complete preceding history; no independence field is present.
-/
structure B2V3AdaptiveCallLedger (Omega : Type*) (Q : ℕ) where
  failure : B2V3PrimitiveFailureKind → Fin Q → Set Omega

namespace B2V3AdaptiveCallLedger

/-- All earlier realized calls are primitive-good. -/
def prefixGood
    {Omega : Type*} {Q : ℕ}
    (ledger : B2V3AdaptiveCallLedger Omega Q) (q : Fin Q) : Set Omega :=
  {omega | ∀ p : Fin Q, p.val < q.val →
    ∀ kind : B2V3PrimitiveFailureKind, omega ∉ ledger.failure kind p}

end B2V3AdaptiveCallLedger

/-- Failure of primitive `kind` at call `q`, after a good realized prefix, inside attraction. -/
def firstPrimitiveFailure
    {Omega : Type*} {Q : ℕ}
    (CAttr : Set Omega) (ledger : B2V3AdaptiveCallLedger Omega Q)
    (kind : B2V3PrimitiveFailureKind) (q : Fin Q) : Set Omega :=
  CAttr ∩ ledger.prefixGood q ∩ ledger.failure kind q

/-- Union of the four first-failure families over the finite realized call order. -/
def allFirstPrimitiveFailures
    {Omega : Type*} {Q : ℕ}
    (CAttr : Set Omega) (ledger : B2V3AdaptiveCallLedger Omega Q) : Set Omega :=
  ⋃ q : Fin Q, ⋃ kind : B2V3PrimitiveFailureKind,
    firstPrimitiveFailure CAttr ledger kind q

/--
If every primitive-good realized ledger transforms attraction coverage into exact survival, then
every coarse-route failure has a least failing call and belongs to one first-failure event.
-/
theorem coarseFailure_subset_firstPrimitiveFailure
    {Omega : Type*} {Q : ℕ}
    (CAttr exactSurvival : Set Omega)
    (ledger : B2V3AdaptiveCallLedger Omega Q)
    (hAllGoodSurvives : ∀ omega, omega ∈ CAttr →
      (∀ q kind, omega ∉ ledger.failure kind q) → omega ∈ exactSurvival) :
    CAttr ∩ exactSurvivalᶜ ⊆ allFirstPrimitiveFailures CAttr ledger := by
  classical
  intro omega hCoarseFailure
  have hAttr : omega ∈ CAttr := hCoarseFailure.1
  have hNotSurvived : omega ∉ exactSurvival := hCoarseFailure.2
  have hSomeFailure :
      ∃ q : Fin Q, ∃ kind : B2V3PrimitiveFailureKind,
        omega ∈ ledger.failure kind q := by
    by_contra hNoFailure
    apply hNotSurvived
    apply hAllGoodSurvives omega hAttr
    intro q kind hFailure
    exact hNoFailure ⟨q, kind, hFailure⟩
  let failingCalls : Finset (Fin Q) :=
    Finset.univ.filter fun q =>
      ∃ kind : B2V3PrimitiveFailureKind, omega ∈ ledger.failure kind q
  have hFailingCallsNonempty : failingCalls.Nonempty := by
    rcases hSomeFailure with ⟨q, kind, hFailure⟩
    refine ⟨q, ?_⟩
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ q, ⟨kind, hFailure⟩⟩
  let firstCall : Fin Q := failingCalls.min' hFailingCallsNonempty
  have hFirstCallMem : firstCall ∈ failingCalls := by
    simpa [firstCall] using Finset.min'_mem failingCalls hFailingCallsNonempty
  have hFailureAtFirst :
      ∃ kind : B2V3PrimitiveFailureKind,
        omega ∈ ledger.failure kind firstCall :=
    (Finset.mem_filter.mp hFirstCallMem).2
  rcases hFailureAtFirst with ⟨firstKind, hFirstFailure⟩
  have hPrefixGood : omega ∈ ledger.prefixGood firstCall := by
    intro earlier hEarlier kind hEarlierFailure
    have hEarlierMem : earlier ∈ failingCalls := by
      exact Finset.mem_filter.mpr
        ⟨Finset.mem_univ earlier, ⟨kind, hEarlierFailure⟩⟩
    have hMin : firstCall ≤ earlier := by
      simpa [firstCall] using Finset.min'_le failingCalls earlier hEarlierMem
    exact (not_le_of_gt hEarlier) hMin
  rw [allFirstPrimitiveFailures]
  refine Set.mem_iUnion.2 ⟨firstCall, Set.mem_iUnion.2 ⟨firstKind, ?_⟩⟩
  exact ⟨⟨hAttr, hPrefixGood⟩, hFirstFailure⟩

/-- Integrated conditional-on-history primitive tails for all four failure families. -/
structure B2V3RawTailBudgets
    {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) {Q : ℕ}
    (CAttr : Set Omega) (ledger : B2V3AdaptiveCallLedger Omega Q) where
  delta : B2V3PrimitiveFailureKind → Fin Q → ENNReal
  firstFailure_le : ∀ kind (q : Fin Q),
    mu (firstPrimitiveFailure CAttr ledger kind q) ≤ delta kind q

namespace B2V3RawTailBudgets

/-- The actual refinement budget: four primitive tail families over the finite realized ledger. -/
noncomputable def deltaR
    {Omega : Type*} [MeasurableSpace Omega]
    {mu : Measure Omega} {Q : ℕ}
    {CAttr : Set Omega} {ledger : B2V3AdaptiveCallLedger Omega Q}
    (budgets : B2V3RawTailBudgets mu CAttr ledger) : ENNReal :=
  ∑ q : Fin Q, ∑ kind : B2V3PrimitiveFailureKind, budgets.delta kind q

end B2V3RawTailBudgets

/--
The coarse A3 probability interface.  It is a finite union bound over adaptive first-failure events
and therefore requires no independence between calls.
-/
theorem measure_coarseFailure_le_deltaR
    {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) {Q : ℕ}
    (CAttr exactSurvival : Set Omega)
    (ledger : B2V3AdaptiveCallLedger Omega Q)
    (budgets : B2V3RawTailBudgets mu CAttr ledger)
    (hCover : CAttr ∩ exactSurvivalᶜ ⊆ allFirstPrimitiveFailures CAttr ledger) :
    mu (CAttr ∩ exactSurvivalᶜ) ≤ budgets.deltaR := by
  calc
    mu (CAttr ∩ exactSurvivalᶜ) ≤
        mu (allFirstPrimitiveFailures CAttr ledger) := measure_mono hCover
    _ ≤ ∑ q : Fin Q,
        mu (⋃ kind : B2V3PrimitiveFailureKind,
          firstPrimitiveFailure CAttr ledger kind q) := by
      unfold allFirstPrimitiveFailures
      exact measure_iUnion_fintype_le mu _
    _ ≤ ∑ q : Fin Q, ∑ kind : B2V3PrimitiveFailureKind,
        mu (firstPrimitiveFailure CAttr ledger kind q) := by
      apply Finset.sum_le_sum
      intro q _
      exact measure_iUnion_fintype_le mu _
    _ ≤ budgets.deltaR := by
      rw [B2V3RawTailBudgets.deltaR]
      exact Finset.sum_le_sum fun q _ =>
        Finset.sum_le_sum fun kind _ => budgets.firstFailure_le kind q

/-- Direct corollary combining the deterministic adapter cover with the four raw tails. -/
theorem measure_coarseFailure_le_deltaR_of_allGoodSurvives
    {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) {Q : ℕ}
    (CAttr exactSurvival : Set Omega)
    (ledger : B2V3AdaptiveCallLedger Omega Q)
    (budgets : B2V3RawTailBudgets mu CAttr ledger)
    (hAllGoodSurvives : ∀ omega, omega ∈ CAttr →
      (∀ q kind, omega ∉ ledger.failure kind q) → omega ∈ exactSurvival) :
    mu (CAttr ∩ exactSurvivalᶜ) ≤ budgets.deltaR :=
  measure_coarseFailure_le_deltaR mu CAttr exactSurvival ledger budgets
    (coarseFailure_subset_firstPrimitiveFailure CAttr exactSurvival ledger hAllGoodSurvives)

end AdaptivePrimitiveLedger

end BiostatFormal.HighDimensional.Multiview
