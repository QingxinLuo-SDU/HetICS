import BiostatFormal.HighDimensional.Multiview.B2RiskDeviation
import BiostatFormal.HighDimensional.Multiview.Refinement

/-!
# B2 exact-start refinement contracts

This file supplies the deterministic Appendix A3 chain for the retained strong exact witness:

* R1: one-call restricted curvature, score control, and a fit certificate imply a parameter radius;
* R2: the complete `Fin 3` multiclass margin and strict constrained-assignment regret imply exact
  assignment;
* R3: a finite list of ordinary bound fit/assignment calls followed by one mandatory terminal call
  preserves exactness or returns a typed failure sentinel.

No declaration in this file states coarse-start contraction, a positive-radius attraction basin,
global IHT convergence, or recovery from arbitrary initialization.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uOmega uCenter uParameter uFailure uCandidate
  uSlot uNormalization uFrozen uReject

section ParameterRadius

variable {Parameter : Type uParameter} [SeminormedAddCommGroup Parameter]

/-- One-error-direction RSC inequality for a minimized fitting objective. -/
structure RestrictedCurvatureAt
    (objective : Parameter -> Real) (scoreLinear : Parameter -> Real)
    (reference estimate : Parameter) (kappa : Real) : Prop where
  kappa_positive : 0 < kappa
  lower_bound :
    objective reference + scoreLinear (estimate - reference) +
        (kappa / 2) * ‖estimate - reference‖ ^ 2 <= objective estimate

/-- Uniform score control specialized to the realized restricted error direction. -/
structure RestrictedScoreBoundAt
    (scoreLinear : Parameter -> Real) (reference estimate : Parameter)
    (restrictedDimension : Nat) (scoreBound : Real) : Prop where
  scoreBound_nonnegative : 0 <= scoreBound
  bound :
    |scoreLinear (estimate - reference)| <=
      Real.sqrt restrictedDimension * scoreBound * ‖estimate - reference‖

/-- The exact Appendix A3 radius, with optimization and score errors kept separate. -/
noncomputable def parameterRadius
    (restrictedDimension : Nat) (scoreBound kappa epsilonFit : Real) : Real :=
  (Real.sqrt restrictedDimension * scoreBound +
      Real.sqrt (restrictedDimension * scoreBound ^ 2 + 2 * kappa * epsilonFit)) / kappa

/--
R1. Restricted curvature, a restricted score bound, and an auditable objective-residual certificate
imply the declared one-call parameter-radius contract.
-/
theorem norm_sub_le_parameterRadius_of_rsc_score_fit
    (objective : Parameter -> Real) (scoreLinear : Parameter -> Real)
    (reference estimate : Parameter)
    (restrictedDimension : Nat) (scoreBound kappa epsilonFit : Real)
    (hRSC : RestrictedCurvatureAt objective scoreLinear reference estimate kappa)
    (hScore : RestrictedScoreBoundAt scoreLinear reference estimate
      restrictedDimension scoreBound)
    (hFit : FitCertificate Parameter objective reference estimate epsilonFit) :
    ‖estimate - reference‖ <=
      parameterRadius restrictedDimension scoreBound kappa epsilonFit := by
  let errorNorm : Real := ‖estimate - reference‖
  let linearEnvelope : Real := Real.sqrt restrictedDimension * scoreBound
  let discriminantRoot : Real :=
    Real.sqrt (restrictedDimension * scoreBound ^ 2 + 2 * kappa * epsilonFit)
  have hErrorNorm : 0 <= errorNorm := norm_nonneg _
  have hDimension : 0 <= (restrictedDimension : Real) := Nat.cast_nonneg _
  have hSqrtDimension : 0 <= Real.sqrt restrictedDimension := Real.sqrt_nonneg _
  have hLinearEnvelope : 0 <= linearEnvelope :=
    mul_nonneg hSqrtDimension hScore.scoreBound_nonnegative
  have hFitUpper : objective estimate <= objective reference + epsilonFit :=
    hFit.residual_bound
  have hCurvature :
      (kappa / 2) * errorNorm ^ 2 <=
        -scoreLinear (estimate - reference) + epsilonFit := by
    dsimp [errorNorm]
    linarith [hRSC.lower_bound, hFitUpper]
  have hNegativeScore :
      -scoreLinear (estimate - reference) <= linearEnvelope * errorNorm := by
    have hAbs := hScore.bound
    dsimp [linearEnvelope, errorNorm] at hAbs ⊢
    exact (neg_le_abs _).trans hAbs
  have hQuadratic :
      kappa * errorNorm ^ 2 <= 2 * linearEnvelope * errorNorm + 2 * epsilonFit := by
    nlinarith [hCurvature, hNegativeScore]
  have hInside :
      0 <= restrictedDimension * scoreBound ^ 2 + 2 * kappa * epsilonFit := by
    have hFitNonnegative := hFit.epsilon_nonnegative
    have hScoreSquare : 0 <= scoreBound ^ 2 := sq_nonneg scoreBound
    exact add_nonneg
      (mul_nonneg hDimension hScoreSquare)
      (mul_nonneg (mul_nonneg (by norm_num) hRSC.kappa_positive.le) hFitNonnegative)
  have hRootSquare :
      discriminantRoot ^ 2 =
        restrictedDimension * scoreBound ^ 2 + 2 * kappa * epsilonFit := by
    dsimp [discriminantRoot]
    exact Real.sq_sqrt hInside
  have hLinearSquare :
      linearEnvelope ^ 2 = restrictedDimension * scoreBound ^ 2 := by
    dsimp [linearEnvelope]
    rw [mul_pow]
    rw [Real.sq_sqrt hDimension]
  have hScaledQuadratic :
      kappa * (kappa * errorNorm ^ 2) <=
        kappa * (2 * linearEnvelope * errorNorm + 2 * epsilonFit) :=
    mul_le_mul_of_nonneg_left hQuadratic hRSC.kappa_positive.le
  have hSquareBound :
      (kappa * errorNorm - linearEnvelope) ^ 2 <= discriminantRoot ^ 2 := by
    nlinarith [hScaledQuadratic, hRootSquare, hLinearSquare]
  have hRootNonnegative : 0 <= discriminantRoot := by
    dsimp [discriminantRoot]
    exact Real.sqrt_nonneg _
  have hLinearRootBound :
      kappa * errorNorm <= linearEnvelope + discriminantRoot := by
    by_contra hNot
    have hGreater : discriminantRoot < kappa * errorNorm - linearEnvelope := by
      linarith
    have hRightPositive : 0 < kappa * errorNorm - linearEnvelope :=
      hRootNonnegative.trans_lt hGreater
    have hSquareStrict :
        discriminantRoot ^ 2 < (kappa * errorNorm - linearEnvelope) ^ 2 := by
      nlinarith [hRootNonnegative, hGreater]
    exact (not_lt_of_ge hSquareBound) hSquareStrict
  rw [parameterRadius]
  apply (le_div_iff₀ hRSC.kappa_positive).2
  dsimp [errorNorm, linearEnvelope, discriminantRoot] at hLinearRootBound ⊢
  simpa [mul_comm] using hLinearRootBound

end ParameterRadius

section Fin3Assignment

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [PseudoMetricSpace Parameter]

/-- Total constrained-assignment score evaluated at one common returned parameter. -/
noncomputable def fin3AssignmentObjective
    (score : Center -> Fin 3 -> Parameter -> Real)
    (parameter : Parameter) (assignment : Partition Center (Fin 3)) : Real :=
  ∑ center, score center (assignment center) parameter

/--
R2.  The full three-class reference margin, pairwise score-difference Lipschitz control, truth
feasibility, and a strict global assignment-regret certificate force coordinatewise truth and hence
exactness modulo a global permutation.
-/
theorem fin3_exact_of_assignment_regret_lt_multiclass_margin
    (truth returned : Partition Center (Fin 3))
    (feasible : Partition Center (Fin 3) -> Prop)
    (score : Center -> Fin 3 -> Parameter -> Real)
    (reference estimate : Parameter)
    (gamma0 lipschitz radius etaAssignment : Real)
    (hMargin : Fin3AssignmentMargin Center Parameter truth score reference gamma0)
    (hTruthFeasible : feasible truth)
    (hLipschitzNonnegative : 0 <= lipschitz)
    (hRadius : dist estimate reference <= radius)
    (hPairwiseLipschitz : ∀ center wrongClass,
      wrongClass ≠ truth center ->
        |(score center (truth center) estimate - score center wrongClass estimate) -
          (score center (truth center) reference - score center wrongClass reference)| <=
            lipschitz * dist estimate reference)
    (hRegret : Fin3AssignmentRegretCertificate Center feasible
      (fin3AssignmentObjective score estimate) returned etaAssignment)
    (hStrictRegret : etaAssignment < gamma0 - lipschitz * radius) :
    Exact returned truth := by
  classical
  have hEstimatedMargin : ∀ center wrongClass,
      wrongClass ≠ truth center ->
        gamma0 - lipschitz * radius <=
          score center (truth center) estimate - score center wrongClass estimate := by
    intro center wrongClass hWrong
    have hReference := hMargin.true_minus_wrong center wrongClass hWrong
    have hPerturbation := hPairwiseLipschitz center wrongClass hWrong
    have hDistanceScaled :
        lipschitz * dist estimate reference <= lipschitz * radius :=
      mul_le_mul_of_nonneg_left hRadius hLipschitzNonnegative
    have hLowerPerturbation :
        -(lipschitz * radius) <=
          (score center (truth center) estimate - score center wrongClass estimate) -
            (score center (truth center) reference - score center wrongClass reference) := by
      exact (neg_le_of_abs_le (hPerturbation.trans hDistanceScaled))
    linarith
  have hSafePositive : 0 < gamma0 - lipschitz * radius :=
    lt_of_le_of_lt hRegret.eta_nonnegative hStrictRegret
  have hCoordinateNonnegative : ∀ center,
      0 <= score center (truth center) estimate - score center (returned center) estimate := by
    intro center
    by_cases hEqual : returned center = truth center
    · simp [hEqual]
    · exact hSafePositive.le.trans
        (hEstimatedMargin center (returned center) hEqual)
  have hReturnedEq : returned = truth := by
    by_contra hNotEqual
    have hExists : ∃ center, returned center ≠ truth center := by
      by_contra hNoCenter
      apply hNotEqual
      funext center
      by_contra hDifferent
      exact hNoCenter ⟨center, hDifferent⟩
    obtain ⟨badCenter, hBadCenter⟩ := hExists
    have hBadMargin :
        gamma0 - lipschitz * radius <=
          score badCenter (truth badCenter) estimate -
            score badCenter (returned badCenter) estimate :=
      hEstimatedMargin badCenter (returned badCenter) hBadCenter
    have hMarginToSum :
        gamma0 - lipschitz * radius <=
          ∑ center,
            (score center (truth center) estimate -
              score center (returned center) estimate) := by
      calc
        gamma0 - lipschitz * radius <=
            score badCenter (truth badCenter) estimate -
              score badCenter (returned badCenter) estimate := hBadMargin
        _ <= ∑ center,
              (score center (truth center) estimate -
                score center (returned center) estimate) := by
          exact Finset.single_le_sum
            (fun center _ => hCoordinateNonnegative center)
            (Finset.mem_univ badCenter)
    have hObjectiveDifference :
        fin3AssignmentObjective score estimate truth -
            fin3AssignmentObjective score estimate returned =
          ∑ center,
            (score center (truth center) estimate -
              score center (returned center) estimate) := by
      simp [fin3AssignmentObjective, Finset.sum_sub_distrib]
    have hRegretAtTruth :
        fin3AssignmentObjective score estimate truth -
            fin3AssignmentObjective score estimate returned <= etaAssignment :=
      hRegret.global_regret truth hTruthFeasible
    rw [hObjectiveDifference] at hRegretAtTruth
    linarith
  rw [hReturnedEq]
  exact labelEquivalent_refl truth

end Fin3Assignment

section CertifiedCalls

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [SeminormedAddCommGroup Parameter]
  {assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop}

/--
All auditable evidence attached to one successful exact-start fit/assignment call.  The returned
assignment and returned parameter are the two fields of the same `TerminalSuccess` payload, and
`reassignment_uses_returned_parameter` records their binding.
-/
structure Fin3SuccessfulCallCertificate
    (truth : Partition Center (Fin 3))
    (payload : TerminalSuccess Parameter (Partition Center (Fin 3)) assignedFrom) where
  reference : Parameter
  objective : Parameter -> Real
  scoreLinear : Parameter -> Real
  restrictedDimension : Nat
  scoreBound : Real
  kappa : Real
  epsilonFit : Real
  rsc : RestrictedCurvatureAt objective scoreLinear reference payload.parameter kappa
  score_control : RestrictedScoreBoundAt scoreLinear reference payload.parameter
    restrictedDimension scoreBound
  fit_certificate :
    FitCertificate Parameter objective reference payload.parameter epsilonFit
  feasible : Partition Center (Fin 3) -> Prop
  assignmentScore : Center -> Fin 3 -> Parameter -> Real
  gamma0 : Real
  lipschitz : Real
  etaAssignment : Real
  margin : Fin3AssignmentMargin Center Parameter truth assignmentScore reference gamma0
  truth_feasible : feasible truth
  lipschitz_nonnegative : 0 <= lipschitz
  pairwise_lipschitz : ∀ center wrongClass,
    wrongClass ≠ truth center ->
      |(assignmentScore center (truth center) payload.parameter -
          assignmentScore center wrongClass payload.parameter) -
        (assignmentScore center (truth center) reference -
          assignmentScore center wrongClass reference)| <=
        lipschitz * dist payload.parameter reference
  assignment_regret : Fin3AssignmentRegretCertificate Center feasible
    (fin3AssignmentObjective assignmentScore payload.parameter)
    payload.candidate etaAssignment
  strict_regret :
    etaAssignment < gamma0 -
      lipschitz * parameterRadius restrictedDimension scoreBound kappa epsilonFit

/-- R1 followed by R2 certifies the assignment returned by one successful bound call. -/
theorem Fin3SuccessfulCallCertificate.exact
    {truth : Partition Center (Fin 3)}
    {payload : TerminalSuccess Parameter (Partition Center (Fin 3)) assignedFrom}
    (certificate : Fin3SuccessfulCallCertificate truth payload) :
    Exact payload.candidate truth := by
  have hNorm := norm_sub_le_parameterRadius_of_rsc_score_fit
    certificate.objective certificate.scoreLinear certificate.reference payload.parameter
    certificate.restrictedDimension certificate.scoreBound certificate.kappa
    certificate.epsilonFit certificate.rsc certificate.score_control certificate.fit_certificate
  have hDistance :
      dist payload.parameter certificate.reference <=
        parameterRadius certificate.restrictedDimension certificate.scoreBound
          certificate.kappa certificate.epsilonFit := by
    simpa [dist_eq_norm] using hNorm
  exact fin3_exact_of_assignment_regret_lt_multiclass_margin
    truth payload.candidate certificate.feasible certificate.assignmentScore
    certificate.reference payload.parameter certificate.gamma0 certificate.lipschitz
    (parameterRadius certificate.restrictedDimension certificate.scoreBound
      certificate.kappa certificate.epsilonFit)
    certificate.etaAssignment certificate.margin certificate.truth_feasible
    certificate.lipschitz_nonnegative hDistance certificate.pairwise_lipschitz
    certificate.assignment_regret certificate.strict_regret

end CertifiedCalls

section FiniteExactStartLedger

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [SeminormedAddCommGroup Parameter]
  {FailureReason : Type uFailure}
  {assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop}

/-- Active labels or a fail-closed sentinel during a finite exact-start path. -/
inductive ExactStartRunState (Candidate : Type uCandidate) (Failure : Type uFailure) where
  | active (candidate : Candidate)
  | failed (reason : Failure)

/-- A run state is exact when active; a failure sentinel carries no candidate. -/
def ExactStartRunState.ExactOrFailure
    {Candidate : Type uCandidate} {Failure : Type uFailure}
    (exactCandidate : Candidate -> Prop) : ExactStartRunState Candidate Failure -> Prop
  | .active candidate => exactCandidate candidate
  | .failed _ => True

/-- A fit/assignment call whose successful payload binds its assignment to its returned parameter. -/
abbrev BoundRefinementCall
    (Parameter : Type uParameter) (Candidate : Type uCandidate) (Failure : Type uFailure)
    (assignedFrom : Parameter -> Candidate -> Prop) :=
  Candidate -> TerminalCallResult Parameter Candidate Failure assignedFrom

/-- Apply one bound call, keeping a previous failure fail-closed. -/
def applyBoundRefinementCall
    {Parameter : Type uParameter} {Candidate : Type uCandidate} {Failure : Type uFailure}
    {assignedFrom : Parameter -> Candidate -> Prop}
    (call : BoundRefinementCall Parameter Candidate Failure assignedFrom) :
    ExactStartRunState Candidate Failure -> ExactStartRunState Candidate Failure
  | .failed reason => .failed reason
  | .active candidate =>
      match call candidate with
      | .success payload => .active payload.candidate
      | .failed reason => .failed reason

/-- A finite list, rather than an unbounded iteration or convergence assertion. -/
def runOrdinaryRefinementCalls
    {Parameter : Type uParameter} {Candidate : Type uCandidate} {Failure : Type uFailure}
    {assignedFrom : Parameter -> Candidate -> Prop} :
    List (BoundRefinementCall Parameter Candidate Failure assignedFrom) ->
      Candidate -> ExactStartRunState Candidate Failure
  | [], initial => .active initial
  | call :: remaining, initial =>
      match call initial with
      | .failed reason => .failed reason
      | .success payload => runOrdinaryRefinementCalls remaining payload.candidate

/-- Finite ordinary-call ledger plus one mandatory terminal refit/reassignment call. -/
structure Fin3ExactStartCallLedger
    (Parameter : Type uParameter) (Center : Type uCenter) (Failure : Type uFailure)
    (assignedFrom : Parameter -> Partition Center (Fin 3) -> Prop) where
  ordinary : List
    (BoundRefinementCall Parameter (Partition Center (Fin 3)) Failure assignedFrom)
  terminal : BoundRefinementCall Parameter (Partition Center (Fin 3)) Failure assignedFrom

/-- Execute all finite ordinary calls and then the terminal call if no hard failure occurred. -/
def Fin3ExactStartCallLedger.run
    (ledger : Fin3ExactStartCallLedger Parameter Center FailureReason assignedFrom)
    (initial : Partition Center (Fin 3)) :
    ExactStartRunState (Partition Center (Fin 3)) FailureReason :=
  applyBoundRefinementCall ledger.terminal
    (runOrdinaryRefinementCalls ledger.ordinary initial)

/-- Every successful branch of one call carries the R1/R2 certificate. -/
def Fin3BoundCallCertified
    (truth : Partition Center (Fin 3))
    (call : BoundRefinementCall Parameter (Partition Center (Fin 3)) FailureReason assignedFrom) :
    Prop :=
  ∀ current payload,
    Exact current truth ->
    call current = .success payload ->
    Nonempty (Fin3SuccessfulCallCertificate truth payload)

/-- Every ordinary call and the mandatory terminal call are certified. -/
def Fin3ExactStartCallLedger.Certified
    (truth : Partition Center (Fin 3))
    (ledger : Fin3ExactStartCallLedger Parameter Center FailureReason assignedFrom) : Prop :=
  (∀ call ∈ ledger.ordinary, Fin3BoundCallCertified truth call) /\
    Fin3BoundCallCertified truth ledger.terminal

/-- Successful application of a certified call preserves exactness; failure is safe. -/
theorem applyBoundRefinementCall_exactOrFailure
    (truth : Partition Center (Fin 3))
    (call : BoundRefinementCall Parameter (Partition Center (Fin 3)) FailureReason assignedFrom)
    (hCertified : Fin3BoundCallCertified truth call)
    (state : ExactStartRunState (Partition Center (Fin 3)) FailureReason)
    (hState : state.ExactOrFailure (fun candidate => Exact candidate truth)) :
    (applyBoundRefinementCall call state).ExactOrFailure
      (fun candidate => Exact candidate truth) := by
  cases state with
  | failed reason => simp [applyBoundRefinementCall, ExactStartRunState.ExactOrFailure]
  | active current =>
      cases hResult : call current with
      | failed reason =>
          simp [applyBoundRefinementCall, hResult, ExactStartRunState.ExactOrFailure]
      | success payload =>
          obtain ⟨certificate⟩ := hCertified current payload hState hResult
          simpa [applyBoundRefinementCall, hResult, ExactStartRunState.ExactOrFailure] using
            certificate.exact

/-- Finite ordinary-call induction from an exact initial assignment. -/
theorem runOrdinaryRefinementCalls_exactOrFailure
    (truth initial : Partition Center (Fin 3))
    (calls : List
      (BoundRefinementCall Parameter (Partition Center (Fin 3)) FailureReason assignedFrom))
    (hInitial : Exact initial truth)
    (hCertified : ∀ call ∈ calls, Fin3BoundCallCertified truth call) :
    (runOrdinaryRefinementCalls calls initial).ExactOrFailure
      (fun candidate => Exact candidate truth) := by
  induction calls generalizing initial with
  | nil => simpa [runOrdinaryRefinementCalls, ExactStartRunState.ExactOrFailure] using hInitial
  | cons call remaining inductionHypothesis =>
      cases hResult : call initial with
      | failed reason =>
          simp [runOrdinaryRefinementCalls, hResult, ExactStartRunState.ExactOrFailure]
      | success payload =>
          have hHeadCertified : Fin3BoundCallCertified truth call :=
            hCertified call (by simp)
          obtain ⟨certificate⟩ := hHeadCertified initial payload hInitial hResult
          have hPayloadExact : Exact payload.candidate truth :=
            certificate.exact
          simpa [runOrdinaryRefinementCalls, hResult] using
            (inductionHypothesis payload.candidate hPayloadExact
              (fun nextCall hMembership => hCertified nextCall (by simp [hMembership])))

/--
R3. A finite certified ordinary-call ledger followed by its mandatory same-returned-parameter
terminal call maps an exact pre assignment to an exact assignment or a failure sentinel.
-/
theorem Fin3ExactStartCallLedger.run_exactOrFailure
    (truth initial : Partition Center (Fin 3))
    (ledger : Fin3ExactStartCallLedger Parameter Center FailureReason assignedFrom)
    (hInitial : Exact initial truth)
    (hCertified : ledger.Certified truth) :
    (ledger.run initial).ExactOrFailure (fun candidate => Exact candidate truth) := by
  apply applyBoundRefinementCall_exactOrFailure truth ledger.terminal hCertified.2
  exact runOrdinaryRefinementCalls_exactOrFailure truth initial ledger.ordinary hInitial
    hCertified.1

end FiniteExactStartLedger

section PostRepresentation

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [SeminormedAddCommGroup Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
  {FailureReason : Type uFailure} {slot : Slot} {direction : Direction}

/--
Structural link between a label-level finite run and the exported typed post.  It records only
successful-label identity or failure-sentinel absence; it does not assume exactness.
-/
def CandidateState.RepresentsRunState
    (state : CandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
      FailureReason slot direction .post)
    (runState : ExactStartRunState (Partition Center (Fin 3)) FailureReason) : Prop :=
  match runState with
  | .active assignment =>
      ∃ candidate, state.toOption = some candidate /\ candidate.1.label = assignment
  | .failed _ => state.toOption = none

omit [Fintype Center] [SeminormedAddCommGroup Parameter] in
/-- A represented exact-or-failure run produces an exact-or-failure typed post. -/
theorem CandidateState.exactOrFailure_of_representsRunState
    (state : CandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
      FailureReason slot direction .post)
    (runState : ExactStartRunState (Partition Center (Fin 3)) FailureReason)
    (truth : Partition Center (Fin 3))
    (hRun : runState.ExactOrFailure (fun assignment => Exact assignment truth))
    (hRepresents : state.RepresentsRunState runState) :
    state.ExactOrFailure truth := by
  cases runState with
  | failed reason =>
      cases state <;>
        simp [CandidateState.RepresentsRunState, CandidateState.ExactOrFailure] at hRepresents ⊢
  | active assignment =>
      cases state with
      | failed reason =>
          simp [CandidateState.RepresentsRunState] at hRepresents
      | valid candidate hSlot hDirection hStage =>
          simp [CandidateState.RepresentsRunState] at hRepresents
          change Exact assignment truth at hRun
          change Exact candidate.1.label truth
          rw [hRepresents]
          exact hRun

/--
R1--R3 source-facing bridge for both directions.  The conclusion is exactly the `hPostSafe`
proposition required by the atomic-guard probability theorem, but it is derived from finite
certified ledgers and structural post representations rather than accepted as an unexplained input.
-/
theorem pairedPost_exactOrFailure_of_certifiedLedgers
    {assignedFromAB assignedFromBA :
      Parameter -> Partition Center (Fin 3) -> Prop}
    (truth : Partition Center (Fin 3))
    (prePair :
      PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        slot .pre)
    (post :
      PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
        FailureReason slot .post)
    (ledgerAB :
      Fin3ExactStartCallLedger Parameter Center FailureReason assignedFromAB)
    (ledgerBA :
      Fin3ExactStartCallLedger Parameter Center FailureReason assignedFromBA)
    (hPreExact : prePair.Exact truth)
    (hCertifiedAB : ledgerAB.Certified truth)
    (hCertifiedBA : ledgerBA.Certified truth)
    (hRepresentsAB : post.ab.RepresentsRunState (ledgerAB.run prePair.ab.1.label))
    (hRepresentsBA : post.ba.RepresentsRunState (ledgerBA.run prePair.ba.1.label)) :
    post.ExactOrFailure truth := by
  constructor
  · apply post.ab.exactOrFailure_of_representsRunState
      (ledgerAB.run prePair.ab.1.label) truth
    · exact ledgerAB.run_exactOrFailure truth prePair.ab.1.label hPreExact.1 hCertifiedAB
    · exact hRepresentsAB
  · apply post.ba.exactOrFailure_of_representsRunState
      (ledgerBA.run prePair.ba.1.label) truth
    · exact ledgerBA.run_exactOrFailure truth prePair.ba.1.label hPreExact.2 hCertifiedBA
    · exact hRepresentsBA

end PostRepresentation

section JointLawComposition

variable {Omega : Type uOmega} [MeasurableSpace Omega]
  {Center : Type uCenter} [Fintype Center]
  {Parameter : Type uParameter} [SeminormedAddCommGroup Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext -> Prop}
  {FailureReason : Type uFailure} {RejectReason : Type uReject} {slot : Slot}

/-- Four separately named B2-A3 cross-event budgets under one joint law. -/
structure B2RefinementEventBudgets
    (jointLaw : Measure Omega)
    (coverage scoreRSC fitCertificate assignmentCertificate terminalCertificate : Set Omega) where
  deltaScoreRSC : ENNReal
  deltaFit : ENNReal
  deltaAssignment : ENNReal
  deltaTerminal : ENNReal
  scoreRSC_bound : jointLaw (coverage ∩ scoreRSCᶜ) <= deltaScoreRSC
  fit_bound : jointLaw (coverage ∩ fitCertificateᶜ) <= deltaFit
  assignment_bound :
    jointLaw (coverage ∩ assignmentCertificateᶜ) <= deltaAssignment
  terminal_bound : jointLaw (coverage ∩ terminalCertificateᶜ) <= deltaTerminal

/-- The exact four-term A3 budget; no coverage or hard-failure term is added. -/
def B2RefinementEventBudgets.deltaR
    {jointLaw : Measure Omega}
    {coverage scoreRSC fitCertificate assignmentCertificate terminalCertificate : Set Omega}
    (budgets : B2RefinementEventBudgets jointLaw coverage scoreRSC fitCertificate
      assignmentCertificate terminalCertificate) : ENNReal :=
  budgets.deltaScoreRSC + budgets.deltaFit + budgets.deltaAssignment + budgets.deltaTerminal

/--
B2-A3 source-facing composition theorem.  R1/R2 certificates on every successful ordinary and
terminal call imply each ledger's R3 certification, which constructs `hPostSafe` internally before
calling the typed guard/probability theorem.

The four primitive joint-law bounds remain explicit in `eventBudgets`; this theorem composes them
but does not manufacture score/RSC or solver concentration from an unspecified B2 law.
-/
theorem measure_b2StrongExactWitness_refinementFailure_le_of_certifiedLedgers
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
    (ledgerAB : Omega ->
      Fin3ExactStartCallLedger Parameter Center FailureReason assignedFromAB)
    (ledgerBA : Omega ->
      Fin3ExactStartCallLedger Parameter Center FailureReason assignedFromBA)
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
    (hCertifiedLedgers : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentCertificate ->
        (ledgerAB omega).Certified truth /\ (ledgerBA omega).Certified truth)
    (hPostRepresentsAB : ∀ omega ∈ coverage,
      omega ∈ terminalCertificate ->
        (post omega).ab.RepresentsRunState
          ((ledgerAB omega).run (prePair omega).ab.1.label))
    (hPostRepresentsBA : ∀ omega ∈ coverage,
      omega ∈ terminalCertificate ->
        (post omega).ba.RepresentsRunState
          ((ledgerBA omega).run (prePair omega).ba.1.label)) :
    jointLaw (coverage ∩
      (strongWitnessRefinementEvent truth prePair post postOK preOK rejectReason)ᶜ) <=
        eventBudgets.deltaR := by
  have hPostSafe : ∀ omega ∈ coverage,
      omega ∈ scoreRSC -> omega ∈ fitCertificate -> omega ∈ assignmentCertificate ->
      omega ∈ terminalCertificate -> (post omega).ExactOrFailure truth := by
    intro omega hCoverage hScoreRSC hFit hAssignment hTerminal
    have hCertified := hCertifiedLedgers omega hCoverage hScoreRSC hFit hAssignment
    exact pairedPost_exactOrFailure_of_certifiedLedgers truth (prePair omega) (post omega)
      (ledgerAB omega) (ledgerBA omega) (hPreExact omega hCoverage)
      hCertified.1 hCertified.2
      (hPostRepresentsAB omega hCoverage hTerminal)
      (hPostRepresentsBA omega hCoverage hTerminal)
  exact measure_strongExactWitness_refinementFailure_le jointLaw truth coverage scoreRSC
    fitCertificate assignmentCertificate terminalCertificate prePair post postOK preOK rejectReason
    hPreExact hPreOK hPostSafe eventBudgets.scoreRSC_bound eventBudgets.fit_bound
    eventBudgets.assignment_bound eventBudgets.terminal_bound le_rfl

end JointLawComposition

end BiostatFormal.HighDimensional.Multiview
