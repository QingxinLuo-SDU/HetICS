import BiostatFormal.HighDimensional.Multiview.B2Refinement

/-!
# A3 proof-carrying terminal interface

This module separates bounded stopping metadata from label/parameter safety.  A terminal output is
usable on the strong exact-witness route only when it carries the existing `Fin 3` R1/R2
certificate.  Otherwise it is a typed failure and the paired atomic guard falls back jointly.

Nothing here states coarse-start contraction, a positive-radius attraction basin, stationarity,
arbitrary-initialization recovery, or global IHT/CEM convergence.
-/

open MeasureTheory

namespace BiostatFormal.HighDimensional.Multiview

universe uOmega uCenter uParameter uSlot uNormalization uFrozen uFailure uReject

/--
Finite stopping metadata.  Both constructors certify only that the reported step is within the
preregistered limit.  In particular, `maxIterEligible` is not a stationarity, contraction, or label
safety certificate.
-/
inductive FiniteStopWitness (maxSteps : Nat) where
  | strictCoreConverged (step : Fin (maxSteps + 1))
  | maxIterEligible (step : Fin (maxSteps + 1))

/-- The realized number of completed steps recorded by a finite stopping witness. -/
def FiniteStopWitness.steps {maxSteps : Nat} : FiniteStopWitness maxSteps -> Nat
  | .strictCoreConverged step => step
  | .maxIterEligible step => step

/-- A finite stopping witness proves termination by the declared bound, and nothing stronger. -/
theorem FiniteStopWitness.steps_le {maxSteps : Nat}
    (stop : FiniteStopWitness maxSteps) : stop.steps <= maxSteps := by
  cases stop with
  | strictCoreConverged step => exact Nat.lt_succ_iff.mp step.isLt
  | maxIterEligible step => exact Nat.lt_succ_iff.mp step.isLt

/-- A terminal failure retains its reason and finite stopping metadata but contains no partition. -/
structure TypedTerminalFailure (FailureReason : Type uFailure) (maxSteps : Nat) where
  reason : FailureReason
  stop : FiniteStopWitness maxSteps

/--
Proof-carrying success for one terminal direction.  The exported post record is linked both to the
assignment and to the parameter of the same terminal payload.  The R1/R2 certificate supplies the
parameter-radius and full `Fin 3` multiclass assignment-safety arguments.
-/
structure Fin3TerminalSafetyCertificate
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    (Slot : Type uSlot) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop)
    (truth : Partition Center (Fin 3))
    (assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop)
    (slot : Slot) (direction : Direction) (maxSteps : Nat) where
  payload : TerminalSuccess Parameter (Partition Center (Fin 3)) assignedFrom
  callCertificate : Fin3SuccessfulCallCertificate truth payload
  post : RiskCandidate Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
  post_slot : post.1.slot = slot
  post_direction : post.1.direction = direction
  post_stage : post.1.context.stage = .post
  post_label_eq : post.1.label = payload.candidate
  post_parameter_eq : post.1.context.fitParameter = payload.parameter
  stop : FiniteStopWitness maxSteps

/-- Direct finite-step `K = 3` safety: a certified terminal post has exact labels. -/
theorem Fin3TerminalSafetyCertificate.post_exact
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop}
    {slot : Slot} {direction : Direction} {maxSteps : Nat}
    (certificate : Fin3TerminalSafetyCertificate Center Parameter Slot NormalizationContext
      FrozenContext Legal truth assignedFrom slot direction maxSteps) :
    Exact certificate.post.1.label truth := by
  rw [certificate.post_label_eq]
  exact certificate.callCertificate.exact

/--
One terminal direction is either a proof-carrying success or a typed failure.  There is no
constructor for an uncertified label-valued success.
-/
inductive DirectionalTerminalOutcome
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    (Slot : Type uSlot) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop)
    (truth : Partition Center (Fin 3))
    (assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop)
    (FailureReason : Type uFailure)
    (slot : Slot) (direction : Direction) (maxSteps : Nat) where
  | certified (certificate :
      Fin3TerminalSafetyCertificate Center Parameter Slot NormalizationContext FrozenContext
        Legal truth assignedFrom slot direction maxSteps)
  | failed (failure : TypedTerminalFailure FailureReason maxSteps)

/-- A typed directional terminal outcome exports either its certified post or a failure sentinel. -/
def DirectionalTerminalOutcome.toCandidateState
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure}
    {slot : Slot} {direction : Direction} {maxSteps : Nat}
    (outcome : DirectionalTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext
      Legal truth assignedFrom FailureReason slot direction maxSteps) :
    CandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
      FailureReason slot direction .post :=
  match outcome with
  | .certified certificate =>
      .valid certificate.post certificate.post_slot certificate.post_direction
        certificate.post_stage
  | .failed failure => .failed failure.reason

/-- One slot's two terminal directions, with independent same-parameter assignment relations. -/
structure PairedTerminalOutcome
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    (Slot : Type uSlot) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop)
    (truth : Partition Center (Fin 3))
    (assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop)
    (FailureReason : Type uFailure) (slot : Slot) (maxSteps : Nat) where
  ab : DirectionalTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
    truth assignedFromAB FailureReason slot .AB maxSteps
  ba : DirectionalTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
    truth assignedFromBA FailureReason slot .BA maxSteps

/-- Convert the two terminal directions atomically to the existing paired post-state type. -/
def PairedTerminalOutcome.toPairedCandidateState
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {slot : Slot} {maxSteps : Nat}
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps) :
    PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
      FailureReason slot .post where
  ab := outcome.ab.toCandidateState
  ba := outcome.ba.toCandidateState

/-- Both directions returned proof-carrying terminal certificates. -/
def PairedTerminalOutcome.BothCertified
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {slot : Slot} {maxSteps : Nat}
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps) : Prop :=
  Exists fun abCertificate :
      Fin3TerminalSafetyCertificate Center Parameter Slot NormalizationContext FrozenContext Legal
        truth assignedFromAB slot .AB maxSteps =>
    Exists fun baCertificate :
      Fin3TerminalSafetyCertificate Center Parameter Slot NormalizationContext FrozenContext Legal
        truth assignedFromBA slot .BA maxSteps =>
      outcome.ab = .certified abCertificate /\ outcome.ba = .certified baCertificate

/-- At least one terminal direction returned a typed failure sentinel. -/
def PairedTerminalOutcome.HasFailure
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {slot : Slot} {maxSteps : Nat}
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps) : Prop :=
  (Exists fun failure : TypedTerminalFailure FailureReason maxSteps =>
      outcome.ab = .failed failure) \/
    (Exists fun failure : TypedTerminalFailure FailureReason maxSteps =>
      outcome.ba = .failed failure)

/-- Every typed paired outcome exports exact labels or failure sentinels in both directions. -/
theorem PairedTerminalOutcome.exactOrFailure
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {slot : Slot} {maxSteps : Nat}
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps) :
    outcome.toPairedCandidateState.ExactOrFailure truth := by
  cases outcome with
  | mk ab ba =>
      cases ab with
      | certified abCertificate =>
          cases ba with
          | certified baCertificate =>
              exact ⟨abCertificate.post_exact, baCertificate.post_exact⟩
          | failed baFailure =>
              exact ⟨abCertificate.post_exact, trivial⟩
      | failed abFailure =>
          cases ba with
          | certified baCertificate =>
              exact ⟨trivial, baCertificate.post_exact⟩
          | failed baFailure =>
              exact ⟨trivial, trivial⟩

/-- A typed failure in either direction makes the paired post unavailable to the atomic guard. -/
theorem PairedTerminalOutcome.toOption_eq_none_of_hasFailure
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {slot : Slot} {maxSteps : Nat}
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps)
    (hFailure : outcome.HasFailure) :
    outcome.toPairedCandidateState.toOption = none := by
  cases outcome with
  | mk ab ba =>
      cases ab with
      | certified abCertificate =>
          cases ba with
          | certified baCertificate =>
              simp [PairedTerminalOutcome.HasFailure] at hFailure
          | failed baFailure => rfl
      | failed abFailure =>
          cases ba <;> rfl

/-- The two proof-carrying-success and typed-failure routes exhaust every paired outcome. -/
theorem PairedTerminalOutcome.bothCertified_or_hasFailure
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {slot : Slot} {maxSteps : Nat}
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps) :
    outcome.BothCertified \/ outcome.HasFailure := by
  cases outcome with
  | mk ab ba =>
      cases ab with
      | certified abCertificate =>
          cases ba with
          | certified baCertificate =>
              exact Or.inl ⟨abCertificate, baCertificate, rfl, rfl⟩
          | failed baFailure =>
              exact Or.inr (Or.inr ⟨baFailure, rfl⟩)
      | failed abFailure =>
          cases ba with
          | certified baCertificate =>
              exact Or.inr (Or.inl ⟨abFailure, rfl⟩)
          | failed baFailure =>
              exact Or.inr (Or.inl ⟨abFailure, rfl⟩)

/--
The requested real terminal dichotomy.  Two certificates form the strong route.  Any typed
directional failure makes the paired post unavailable and forces the atomic guard to restore both
pre records together.
-/
theorem pairedTerminal_certificate_or_typedFailureWithSafeFallback
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {RejectReason : Type uReject}
    {slot : Slot} {maxSteps : Nat}
    (postOK :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (prePair :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre)
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps)
    (hPreOK : preOK prePair) :
    outcome.BothCertified \/
      (outcome.HasFailure /\
        atomicGuard postOK preOK rejectReason
          (prePair.toPairedCandidateState :
            PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext
              Legal FailureReason slot .pre)
          outcome.toPairedCandidateState = .fallbackPre prePair) := by
  rcases outcome.bothCertified_or_hasFailure with hCertified | hFailure
  · exact Or.inl hCertified
  · refine Or.inr ⟨hFailure, ?_⟩
    exact atomicGuard_eq_fallbackPre_of_post_failure postOK preOK rejectReason prePair
      outcome.toPairedCandidateState (outcome.toOption_eq_none_of_hasFailure hFailure) hPreOK

/-- Exact pre plus a typed paired terminal outcome always gives an exact atomic-guard result. -/
theorem atomicGuard_exact_of_pairedTerminalOutcome
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {RejectReason : Type uReject}
    {slot : Slot} {maxSteps : Nat}
    (postOK :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (prePair :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre)
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps)
    (hPreExact : prePair.Exact truth)
    (hPreOK : preOK prePair) :
    (atomicGuard postOK preOK rejectReason
      (prePair.toPairedCandidateState :
        PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
          FailureReason slot .pre)
      outcome.toPairedCandidateState).Exact truth := by
  exact atomicGuard_exact_of_exact_pre_of_post_exactOrFailure postOK preOK rejectReason prePair
    outcome.toPairedCandidateState truth hPreExact hPreOK outcome.exactOrFailure

/-- Every typed terminal route leaves an exact record in the realized guarded pool. -/
theorem PairedTerminalOutcome.exists_exact_mem_guardedPool
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [SeminormedAddCommGroup Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
    {truth : Partition Center (Fin 3)}
    {assignedFromAB assignedFromBA : Parameter -> Partition Center (Fin 3) -> Prop}
    {FailureReason : Type uFailure} {RejectReason : Type uReject}
    {slot : Slot} {maxSteps : Nat}
    [DecidableEq
      (RiskCandidate Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal)]
    (postOK :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : RejectReason)
    (prePair :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre)
    (outcome : PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
      truth assignedFromAB assignedFromBA FailureReason slot maxSteps)
    (hPreExact : prePair.Exact truth)
    (hPreOK : preOK prePair) :
    Exists fun candidate =>
      candidate ∈
        (atomicGuard postOK preOK rejectReason
          (prePair.toPairedCandidateState :
            PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext
              Legal FailureReason slot .pre)
          outcome.toPairedCandidateState).toGuardedPool /\
      Exact candidate.1.label truth := by
  apply AtomicGuardResult.exists_exact_mem_toGuardedPool
  exact atomicGuard_exact_of_pairedTerminalOutcome postOK preOK rejectReason prePair outcome
    hPreExact hPreOK

section TerminalOutcomeProbability

variable {Omega : Type uOmega} [MeasurableSpace Omega]
  {Center : Type uCenter} [Fintype Center]
  {Parameter : Type uParameter} [SeminormedAddCommGroup Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
  {FailureReason : Type uFailure} {RejectReason : Type uReject}
  {slot : Slot} {maxSteps : Nat}

/--
B2-facing four-event composition through the proof-carrying terminal type.  The source-facing
premise binds the exported post to one typed paired outcome on the four good events.  A typed hard
failure inside that outcome is safe by the deterministic fallback theorem and incurs no fifth
probability charge.
-/
theorem measure_b2StrongExactWitness_refinementFailure_le_of_terminalOutcomes
    {assignedFromAB assignedFromBA :
      Parameter -> Partition Center (Fin 3) -> Prop}
    [DecidableEq
      (RiskCandidate Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal)]
    (jointLaw : Measure Omega) [IsProbabilityMeasure jointLaw]
    (truth : Partition Center (Fin 3))
    (coverage scoreRSC fitCertificate assignmentCertificate terminalCertificate : Set Omega)
    (prePair : Omega ->
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre)
    (post : Omega ->
      PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (postOK : Omega ->
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .post -> Prop)
    (preOK : Omega ->
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre -> Prop)
    (rejectReason : Omega -> RejectReason)
    (eventBudgets : B2RefinementEventBudgets jointLaw coverage scoreRSC fitCertificate
      assignmentCertificate terminalCertificate)
    (hPreExact : ∀ omega ∈ coverage, (prePair omega).Exact truth)
    (hPreOK : ∀ omega ∈ coverage, preOK omega (prePair omega))
    (hTerminalOutcome : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentCertificate ->
      omega ∈ terminalCertificate ->
        Exists fun outcome :
          PairedTerminalOutcome Center Parameter Slot NormalizationContext FrozenContext Legal
            truth assignedFromAB assignedFromBA FailureReason slot maxSteps =>
          post omega = outcome.toPairedCandidateState) :
    jointLaw (coverage ∩
      (strongWitnessRefinementEvent truth prePair post postOK preOK rejectReason)ᶜ) <=
        eventBudgets.deltaR := by
  have hPostSafe : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentCertificate ->
      omega ∈ terminalCertificate -> (post omega).ExactOrFailure truth := by
    intro omega hCoverage hScoreRSC hFit hAssignment hTerminal
    obtain ⟨outcome, hPost⟩ :=
      hTerminalOutcome omega hCoverage hScoreRSC hFit hAssignment hTerminal
    rw [hPost]
    exact outcome.exactOrFailure
  exact measure_strongExactWitness_refinementFailure_le jointLaw truth coverage scoreRSC
    fitCertificate assignmentCertificate terminalCertificate prePair post postOK preOK rejectReason
    hPreExact hPreOK hPostSafe eventBudgets.scoreRSC_bound eventBudgets.fit_bound
    eventBudgets.assignment_bound eventBudgets.terminal_bound le_rfl

end TerminalOutcomeProbability

end BiostatFormal.HighDimensional.Multiview
