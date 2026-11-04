import BiostatFormal.HighDimensional.Multiview.B2V3RefinementAdapter
import Mathlib.Tactic.DeriveFintype

/-!
# B2-v3 actual outer refinement source model

This module records the source-level objects needed to bind the 202608 outer pipeline to the
existing coarse-refinement mathematics.  The source constructs the historical 17-record roster,
then the theorem-two primary arm projects to the 15 non-pure-F active records.  Canonical owners
represent physical fits, so logical aliases do not create duplicate calls.

The fixed witness ledger contains only the two directional `outcome__kmeans` runs and their paired
guard.  No declaration in this file treats a finite stopping status as a correctness certificate.
-/

namespace BiostatFormal.HighDimensional.Multiview

universe uCenter uParameter uKey uOwner uState uPayload uReject uSlot uNormalization uFrozen

/-- The complete 17-record source roster, including the two pure-F diagnostic records. -/
abbrev B2V3HistoricalLogicalRecord := B2CandidateSlot

/-- Project a historical record to the 15-slot theorem-two primary pool. -/
def B2V3PrimaryActiveProjection :
    B2V3HistoricalLogicalRecord → Option B2V3ActiveSlot
  | .inl slot => some slot
  | .inr _ => none

@[simp]
theorem b2V3PrimaryActiveProjection_active (slot : B2V3ActiveSlot) :
    B2V3PrimaryActiveProjection (.inl slot) = some slot := rfl

@[simp]
theorem b2V3PrimaryActiveProjection_diagnostic (slot : B2V3PureFDiagnosticSlot) :
    B2V3PrimaryActiveProjection (.inr slot) = none := rfl

/--
Source-shaped logical-to-physical ownership.  `canonicalKey` is the mathematical image of the
Python canonical partition key; injectivity of `physicalOwner` prevents two keys from naming the
same physical call.
-/
structure B2V3CanonicalOwner (Key : Type uKey) (Owner : Type uOwner) where
  eligible : B2V3HistoricalLogicalRecord → Prop
  canonicalKey : B2V3HistoricalLogicalRecord → Option Key
  physicalOwner : Key → Owner
  key_present_iff_eligible : ∀ slot,
    (canonicalKey slot).isSome = true ↔ eligible slot
  physicalOwner_injective : Function.Injective physicalOwner

namespace B2V3CanonicalOwner

variable {Key : Type uKey} {Owner : Type uOwner}

/-- The physical owner of an eligible logical record, when its canonical key is present. -/
def logicalOwner
    (ownership : B2V3CanonicalOwner Key Owner)
    (slot : B2V3HistoricalLogicalRecord) : Option Owner :=
  (ownership.canonicalKey slot).map ownership.physicalOwner

/-- Every source-eligible logical record has one and only one physical owner. -/
theorem existsUnique_logicalOwner_of_eligible
    (ownership : B2V3CanonicalOwner Key Owner)
    (slot : B2V3HistoricalLogicalRecord)
    (hEligible : ownership.eligible slot) :
    ∃! owner, ownership.logicalOwner slot = some owner := by
  have hPresent : (ownership.canonicalKey slot).isSome = true :=
    (ownership.key_present_iff_eligible slot).2 hEligible
  cases hKey : ownership.canonicalKey slot with
  | none => simp [hKey] at hPresent
  | some key =>
      refine ⟨ownership.physicalOwner key, ?_, ?_⟩
      · simp [logicalOwner, hKey]
      · intro owner hOwner
        simpa [logicalOwner, hKey] using hOwner.symm

end B2V3CanonicalOwner

/--
The actual 202608 primary roster: source eligibility is exactly projection from the historical
17-record universe to the 15 active records, followed by canonical ownership.
-/
structure B2V3ActualPrimaryRoster (Key : Type uKey) (Owner : Type uOwner) where
  ownership : B2V3CanonicalOwner Key Owner
  eligible_iff_primary : ∀ slot,
    ownership.eligible slot ↔ ∃ active, B2V3PrimaryActiveProjection slot = some active

namespace B2V3ActualPrimaryRoster

variable {Key : Type uKey} {Owner : Type uOwner}

/-- The preregistered `outcome__kmeans` logical witness has one canonical physical owner. -/
theorem existsUnique_outcomeKMeansOwner
    (roster : B2V3ActualPrimaryRoster Key Owner) :
    ∃! owner,
      roster.ownership.logicalOwner (.inl B2V3ActiveSlot.outcomeKMeans) = some owner := by
  apply roster.ownership.existsUnique_logicalOwner_of_eligible
  apply (roster.eligible_iff_primary (.inl B2V3ActiveSlot.outcomeKMeans)).2
  exact ⟨B2V3ActiveSlot.outcomeKMeans, rfl⟩

end B2V3ActualPrimaryRoster

/--
The two directional physical canonical owners followed by one fixed active witness slot. AB and BA
canonicalize separate training-fold outputs, so they need not have the same owner.
-/
structure B2V3WitnessBinding
    (witnessSlot : B2V3ActiveSlot) (Key : Type uKey) (Owner : Type uOwner) where
  rosterAB : B2V3ActualPrimaryRoster Key Owner
  rosterBA : B2V3ActualPrimaryRoster Key Owner
  ownerAB : Owner
  ownerBA : Owner
  ownerAB_eq :
    rosterAB.ownership.logicalOwner (.inl witnessSlot) = some ownerAB
  ownerBA_eq :
    rosterBA.ownership.logicalOwner (.inl witnessSlot) = some ownerBA

/-- Backward-compatible name for the preregistered Y-only witness binding. -/
abbrev B2V3OutcomeWitnessBinding (Key : Type uKey) (Owner : Type uOwner) :=
  B2V3WitnessBinding B2V3ActiveSlot.outcomeKMeans Key Owner

/--
The exact fixed call envelope for the preregistered witness: ordinary calls and one terminal call
in each direction, followed by one paired guard.
-/
inductive B2V3ActualWitnessSite (T : Nat) where
  | ordinary (direction : Direction) (iteration : Fin T)
  | terminal (direction : Direction)
  | guard
  deriving DecidableEq, Fintype, Repr

/-- A transparent sum representation used only to count the fixed witness sites. -/
def b2V3ActualWitnessSiteEquiv (T : Nat) :
    B2V3ActualWitnessSite T ≃
      Sum (Direction × Fin T) (Sum Direction Unit) where
  toFun
    | .ordinary direction iteration => .inl (direction, iteration)
    | .terminal direction => .inr (.inl direction)
    | .guard => .inr (.inr ())
  invFun
    | .inl (direction, iteration) => .ordinary direction iteration
    | .inr (.inl direction) => .terminal direction
    | .inr (.inr _) => .guard
  left_inv site := by cases site <;> rfl
  right_inv site := by
    rcases site with site | site
    · rcases site with ⟨direction, iteration⟩
      rfl
    · rcases site with direction | guardToken
      · rfl
      · cases guardToken
        rfl

/-- The witness ledger has exactly `2(T+1)+1` sites. -/
theorem b2V3ActualWitnessSite_card (T : Nat) :
    Fintype.card (B2V3ActualWitnessSite T) = 2 * (T + 1) + 1 := by
  rw [Fintype.card_congr (b2V3ActualWitnessSiteEquiv T)]
  have hDirection : Fintype.card Direction = 2 := by decide
  simp [Fintype.card_sum, Fintype.card_prod, hDirection]
  omega

/--
Every candidate source branch is explicit.  Terminal success is parameterized by a proof-carrying
payload type; source failure carries no such payload.
-/
inductive B2V3ActualCandidateReturn
    (State : Type uState) (TerminalPayload : Type uPayload)
    (RejectReason : Type uReject) (maxSteps : Nat) where
  | ordinary (result : B2V3OrdinarySuccess State maxSteps)
  | terminal (result : B2V3DirectionalSourceOutcome TerminalPayload
      (B2V3TypedSourceFailure maxSteps) RejectReason)
  | sourceIneligible (failure : B2V3TypedSourceFailure maxSteps)

/-- Source-returned directional states before deterministic R1/R2 consequences are proved. -/
structure B2V3ActualDirectionRun
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] (T : Nat) where
  direction : Direction
  alignment : Equiv.Perm (Fin 3)
  labels : Nat → Partition Center (Fin 3)
  parameters : Fin T → Parameter
  active : Fin T → Prop
  inactive_labels : ∀ iteration, ¬ active iteration →
    labels (iteration.val + 1) = labels iteration.val

/--
One realized ordinary source call.  The fit component separates label contamination, sample-score
fluctuation, and finite optimization residual.  The assignment component is the complete `Fin 3`
margin/regret certificate for the very same returned parameter.
-/
structure B2V3ActualOrdinaryCallCertificate
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [PseudoMetricSpace Parameter]
    (alignment : Equiv.Perm (Fin 3))
    (truth current returned : Partition Center (Fin 3))
    (reference estimate : Parameter) where
  scoreMagnitude : ℝ
  fitResidual : ℝ
  kappa : ℝ
  bLabel : ℝ
  gStat : ℝ
  epsilonOpt : ℝ
  fit : B2V3FitPrimitiveCertificate
    (dist estimate reference) (alignedLabelError alignment current truth)
    scoreMagnitude fitResidual kappa bLabel gStat epsilonOpt
  assignment : B2V3Fin3AssignmentCertificate
    alignment truth returned reference estimate

namespace B2V3ActualOrdinaryCallCertificate

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [DecidableEq Center] [Nonempty Center]
  [PseudoMetricSpace Parameter]
  {alignment : Equiv.Perm (Fin 3)}
  {truth current returned : Partition Center (Fin 3)}
  {reference estimate : Parameter}

/-- Label-contamination coefficient obtained from the call's RSC/score certificate. -/
noncomputable def contaminationCoefficient
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      alignment truth current returned reference estimate) : ℝ :=
  v3ContaminationCoefficient certificate.bLabel certificate.kappa

/-- Statistical component of the parameter radius. -/
noncomputable def statisticalRadius
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      alignment truth current returned reference estimate) : ℝ :=
  v3StatisticalRadius certificate.gStat certificate.kappa

/-- Finite-optimization component of the parameter radius. -/
noncomputable def optimizationRadius
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      alignment truth current returned reference estimate) : ℝ :=
  v3OptimizationRadius certificate.epsilonOpt certificate.kappa

/-- The call-local contraction coefficient, computed from curvature, margin, and Lipschitz data. -/
noncomputable def rho
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      alignment truth current returned reference estimate) : ℝ :=
  v3Rho (Fintype.card Center) certificate.assignment.gamma0
    (v3AssignmentLipschitzSum certificate.assignment) certificate.contaminationCoefficient

/-- The call-local statistical/optimization/assignment floor. -/
noncomputable def xi
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      alignment truth current returned reference estimate) : ℝ :=
  v3Xi (Fintype.card Center) certificate.assignment.gamma0
    (v3AssignmentLipschitzSum certificate.assignment)
    certificate.statisticalRadius certificate.optimizationRadius
    certificate.assignment.etaAsn

end B2V3ActualOrdinaryCallCertificate

/--
A finite direction-level envelope for the actual ordinary-call ledger.  `rho` and `xi` are only
upper envelopes of call-local quantities already derived from the primitive certificates.
An inactive source iteration is admitted only with an exact-label early-stop certificate.
-/
structure B2V3ActualDirectionCertificate
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [PseudoMetricSpace Parameter]
    (T : Nat) (run : B2V3ActualDirectionRun Center Parameter T)
    (truth : Partition Center (Fin 3)) (reference : Parameter) where
  rho : ℝ
  xi : ℝ
  radii : B2V3AttractionRadii
  ordinary : ∀ iteration, run.active iteration →
    B2V3ActualOrdinaryCallCertificate Center Parameter run.alignment truth
      (run.labels iteration.val) (run.labels (iteration.val + 1)) reference
      (run.parameters iteration)
  rho_upper : ∀ iteration hActive, (ordinary iteration hActive).rho ≤ rho
  xi_upper : ∀ iteration hActive, (ordinary iteration hActive).xi ≤ xi
  rho_nonnegative : 0 ≤ rho
  xi_nonnegative : 0 ≤ xi
  rho_lt_one : rho < 1
  attraction_invariant_bound : rho * radii.rAttr + xi ≤ radii.rAttr
  start_in_attraction :
    alignedLabelError run.alignment (run.labels 0) truth ≤ radii.rAttr
  inactive_exact : ∀ iteration, ¬ run.active iteration →
    run.labels iteration.val = alignedTruth run.alignment truth
  exact_grid : radii.rExact ≤ 1 / (Fintype.card Center : ℝ)
  finite_exact_bound :
    rho ^ T * radii.rAttr + ((1 - rho ^ T) / (1 - rho)) * xi < radii.rExact

namespace B2V3ActualDirectionRun

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [DecidableEq Center] [Nonempty Center]
  [PseudoMetricSpace Parameter]
  {T : Nat} {run : B2V3ActualDirectionRun Center Parameter T}
  {truth current returned : Partition Center (Fin 3)}
  {reference estimate : Parameter}

omit [Nonempty Center] in
@[simp]
theorem alignedLabelError_alignedTruth (alignment : Equiv.Perm (Fin 3))
    (truth : Partition Center (Fin 3)) :
    alignedLabelError alignment (alignedTruth alignment truth) truth = 0 := by
  classical
  simp [alignedLabelError, alignedMismatchCenters]

theorem alignedLabelError_nonnegative (alignment : Equiv.Perm (Fin 3))
    (candidate truth : Partition Center (Fin 3)) :
    0 ≤ alignedLabelError alignment candidate truth := by
  rw [alignedLabelError]
  positivity

/-- R1 for one realized source call. -/
theorem parameter_error
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      run.alignment truth current returned reference estimate) :
    dist estimate reference ≤
      certificate.contaminationCoefficient *
          alignedLabelError run.alignment current truth +
        certificate.statisticalRadius + certificate.optimizationRadius := by
  exact parameterError_le_of_v3FitPrimitives certificate.fit

/-- R2 for one realized source call, including both wrong classes at every center. -/
theorem assignment_error
    (certificate : B2V3ActualOrdinaryCallCertificate Center Parameter
      run.alignment truth current returned reference estimate) :
    alignedLabelError run.alignment returned truth ≤
      certificate.rho * alignedLabelError run.alignment current truth + certificate.xi := by
  exact labelError_le_rho_mul_add_xi certificate.assignment
    certificate.contaminationCoefficient certificate.statisticalRadius
    certificate.optimizationRadius (parameter_error certificate)

/-- Every active or certified-inactive source step obeys the common finite-ledger envelope. -/
theorem step_contraction
    (certificate : B2V3ActualDirectionCertificate Center Parameter T run truth reference)
    (t : Nat) (hWithin : t < T) :
    alignedLabelError run.alignment (run.labels (t + 1)) truth ≤
      certificate.rho * alignedLabelError run.alignment (run.labels t) truth +
        certificate.xi := by
  let iteration : Fin T := ⟨t, hWithin⟩
  by_cases hActive : run.active iteration
  · have hLocal := assignment_error (certificate.ordinary iteration hActive)
    exact hLocal.trans <| add_le_add
      (mul_le_mul_of_nonneg_right (certificate.rho_upper iteration hActive)
        (alignedLabelError_nonnegative run.alignment (run.labels t) truth))
      (certificate.xi_upper iteration hActive)
  · have hCurrent : run.labels t = alignedTruth run.alignment truth :=
      certificate.inactive_exact iteration hActive
    have hNext : run.labels (t + 1) = alignedTruth run.alignment truth := by
      rw [run.inactive_labels iteration hActive]
      exact hCurrent
    rw [hCurrent, hNext, alignedLabelError_alignedTruth]
    simpa using certificate.xi_nonnegative

/-- The displayed basin inequality is invariant over the actual finite ledger. -/
theorem attraction_invariant
    (certificate : B2V3ActualDirectionCertificate Center Parameter T run truth reference) :
    ∀ n, n ≤ T →
      alignedLabelError run.alignment (run.labels n) truth ≤ certificate.radii.rAttr := by
  intro n hWithin
  induction n with
  | zero => exact certificate.start_in_attraction
  | succ n hInduction =>
      have hPrevious : n ≤ T := Nat.le_trans (Nat.le_succ n) hWithin
      have hStepWithin : n < T := Nat.lt_of_succ_le hWithin
      exact (step_contraction certificate n hStepWithin).trans <|
        (add_le_add
          (mul_le_mul_of_nonneg_left (hInduction hPrevious) certificate.rho_nonnegative)
          (le_refl certificate.xi)).trans certificate.attraction_invariant_bound

/-- Finite-ledger affine iteration; no behavior after `T` is assumed. -/
theorem iterate_affine_error_bound_finite
    (error : Nat → ℝ) (rho xi : ℝ) (T : Nat)
    (hRhoNonnegative : 0 ≤ rho)
    (hStep : ∀ t, t < T → error (t + 1) ≤ rho * error t + xi) :
    error T ≤ rho ^ T * error 0 +
      (∑ j ∈ Finset.range T, rho ^ j) * xi := by
  induction T with
  | zero => simp
  | succ T hInduction =>
      have hEarlier : ∀ t, t < T → error (t + 1) ≤ rho * error t + xi := by
        intro t ht
        exact hStep t (Nat.lt_succ_of_lt ht)
      calc
        error (T + 1) ≤ rho * error T + xi := hStep T (Nat.lt_succ_self T)
        _ ≤ rho *
              (rho ^ T * error 0 +
                (∑ j ∈ Finset.range T, rho ^ j) * xi) + xi :=
          add_le_add (mul_le_mul_of_nonneg_left (hInduction hEarlier) hRhoNonnegative)
            (le_refl xi)
        _ = rho ^ (T + 1) * error 0 +
              (∑ j ∈ Finset.range (T + 1), rho ^ j) * xi := by
          rw [geom_sum_succ]
          ring

/-- Closed form of the finite-ledger affine iteration. -/
theorem iterate_affine_closed_form_finite
    (error : Nat → ℝ) (rho xi : ℝ) (T : Nat)
    (hRhoNonnegative : 0 ≤ rho) (hRhoLessThanOne : rho < 1)
    (hStep : ∀ t, t < T → error (t + 1) ≤ rho * error t + xi) :
    error T ≤ rho ^ T * error 0 + ((1 - rho ^ T) / (1 - rho)) * xi := by
  calc
    error T ≤ rho ^ T * error 0 + ((rho ^ T - 1) / (rho - 1)) * xi := by
      rw [← geom_sum_eq (ne_of_lt hRhoLessThanOne)]
      exact iterate_affine_error_bound_finite error rho xi T hRhoNonnegative hStep
    _ = rho ^ T * error 0 + ((1 - rho ^ T) / (1 - rho)) * xi := by
      rw [show rho ^ T - 1 = -(1 - rho ^ T) by ring,
        show rho - 1 = -(1 - rho) by ring, neg_div_neg_eq]

/-- The actual finite ordinary-call ledger falls below the one-center error grid. -/
theorem terminal_pre_error_lt_grid
    (certificate : B2V3ActualDirectionCertificate Center Parameter T run truth reference) :
    alignedLabelError run.alignment (run.labels T) truth <
      1 / (Fintype.card Center : ℝ) := by
  let error : Nat → ℝ := fun t => alignedLabelError run.alignment (run.labels t) truth
  have hClosed := iterate_affine_closed_form_finite error certificate.rho certificate.xi T
    certificate.rho_nonnegative certificate.rho_lt_one
    (fun t ht => step_contraction certificate t ht)
  have hPowNonnegative : 0 ≤ certificate.rho ^ T :=
    pow_nonneg certificate.rho_nonnegative T
  have hAttractionBound :
      error T ≤ certificate.rho ^ T * certificate.radii.rAttr +
        ((1 - certificate.rho ^ T) / (1 - certificate.rho)) * certificate.xi :=
    hClosed.trans <| add_le_add
      (mul_le_mul_of_nonneg_left certificate.start_in_attraction hPowNonnegative)
      (le_refl (((1 - certificate.rho ^ T) / (1 - certificate.rho)) * certificate.xi))
  have hInsideExact : error T < certificate.radii.rExact :=
    hAttractionBound.trans_lt certificate.finite_exact_bound
  have hBelowGrid : error T < 1 / (Fintype.card Center : ℝ) :=
    hInsideExact.trans_le certificate.exact_grid
  exact hBelowGrid

/-- The finite ledger reaches the truth under the run's single fixed alignment. -/
theorem terminal_pre_aligned
    (certificate : B2V3ActualDirectionCertificate Center Parameter T run truth reference) :
    run.labels T = alignedTruth run.alignment truth := by
  by_contra hNe
  exact (not_le_of_gt (terminal_pre_error_lt_grid certificate))
    (labelError_ge_inv_card_of_ne run.alignment (run.labels T) truth hNe)

/-- The actual finite ordinary-call ledger reaches exact labels before candidate terminalization. -/
theorem terminal_pre_exact
    (certificate : B2V3ActualDirectionCertificate Center Parameter T run truth reference) :
    Exact (run.labels T) truth :=
  exact_of_labelError_lt_inv_card run.alignment (run.labels T) truth
    (terminal_pre_error_lt_grid certificate)

end B2V3ActualDirectionRun

/-- The two source directions for one logical witness slot. -/
structure B2V3ActualPairedRun
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] (T : Nat) where
  ab : B2V3ActualDirectionRun Center Parameter T
  ba : B2V3ActualDirectionRun Center Parameter T
  ab_direction : ab.direction = .AB
  ba_direction : ba.direction = .BA

/--
The proof-relevant payload of the actual candidate-local terminal refit and reassignment.  The
assignment certificate is indexed by `terminalParameter`, so a certificate for a different fit
cannot be substituted.  This is distinct from the later post-selection fixed-label refit.
-/
structure B2V3ActualTerminalSuccess
    (Center : Type uCenter) (Parameter : Type uParameter)
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [PseudoMetricSpace Parameter]
    (Slot : Type uSlot) (NormalizationContext : Type uNormalization)
    (FrozenContext : Type uFrozen)
    (Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop)
    (T : Nat) (run : B2V3ActualDirectionRun Center Parameter T)
    (truth : Partition Center (Fin 3)) (reference : Parameter) (rTheta : ℝ)
    (slot : Slot) (direction : Direction) (maxSteps : Nat) where
  terminalParameter : Parameter
  terminalLabels : Partition Center (Fin 3)
  call : B2V3ActualOrdinaryCallCertificate Center Parameter run.alignment truth
    (run.labels T) terminalLabels reference terminalParameter
  parameter_radius : call.statisticalRadius + call.optimizationRadius ≤ rTheta
  assignment_exact_radius : call.xi < 1 / (Fintype.card Center : ℝ)
  post : RiskCandidate Center (Fin 3) Slot Parameter NormalizationContext FrozenContext Legal
  post_slot : post.1.slot = slot
  post_direction : post.1.direction = direction
  source_direction : run.direction = direction
  post_stage : post.1.context.stage = .post
  post_label_eq_terminal : post.1.label = terminalLabels
  post_parameter_eq_terminal : post.1.context.fitParameter = terminalParameter
  stop : FiniteStopWitness maxSteps

namespace B2V3ActualTerminalSuccess

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [DecidableEq Center] [Nonempty Center]
  [PseudoMetricSpace Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
  {T : Nat} {run : B2V3ActualDirectionRun Center Parameter T}
  {truth : Partition Center (Fin 3)} {reference : Parameter} {rTheta : ℝ}
  {slot : Slot} {direction : Direction} {maxSteps : Nat}

/-- R1 applied to the terminal fit, using exact terminal-pre labels. -/
theorem parameter_safe
    (directionCertificate :
      B2V3ActualDirectionCertificate Center Parameter T run truth reference)
    (success : B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext
      Legal T run truth reference rTheta slot direction maxSteps) :
    dist success.terminalParameter reference ≤ rTheta := by
  have hParameter := B2V3ActualDirectionRun.parameter_error success.call
  have hCurrentError : alignedLabelError run.alignment (run.labels T) truth = 0 := by
    rw [B2V3ActualDirectionRun.terminal_pre_aligned directionCertificate]
    exact B2V3ActualDirectionRun.alignedLabelError_alignedTruth run.alignment truth
  rw [hCurrentError] at hParameter
  have hReduced : dist success.terminalParameter reference ≤
      success.call.statisticalRadius + success.call.optimizationRadius := by
    simpa using hParameter
  exact hReduced.trans success.parameter_radius

/-- The same-terminal-parameter Fin-3 margin/regret certificate makes reassigned labels exact. -/
theorem terminal_labels_exact
    (directionCertificate :
      B2V3ActualDirectionCertificate Center Parameter T run truth reference)
    (success : B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext
      Legal T run truth reference rTheta slot direction maxSteps) :
    Exact success.terminalLabels truth := by
  have hAssignment := B2V3ActualDirectionRun.assignment_error success.call
  have hCurrentError : alignedLabelError run.alignment (run.labels T) truth = 0 := by
    rw [B2V3ActualDirectionRun.terminal_pre_aligned directionCertificate]
    exact B2V3ActualDirectionRun.alignedLabelError_alignedTruth run.alignment truth
  rw [hCurrentError] at hAssignment
  have hSmall : alignedLabelError run.alignment success.terminalLabels truth <
      1 / (Fintype.card Center : ℝ) := by
    have hXi : alignedLabelError run.alignment success.terminalLabels truth ≤ success.call.xi := by
      simpa using hAssignment
    exact hXi.trans_lt success.assignment_exact_radius
  exact exact_of_labelError_lt_inv_card run.alignment success.terminalLabels truth hSmall

/-- Construct the old exact terminal payload from R1--R3 instead of assuming `hPostSafe`. -/
def toCertifiedPayload
    (directionCertificate :
      B2V3ActualDirectionCertificate Center Parameter T run truth reference)
    (success : B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext
      Legal T run truth reference rTheta slot direction maxSteps) :
    B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext FrozenContext Legal
      truth reference rTheta slot direction maxSteps .success where
  frozenOuterLabels := success.terminalLabels
  terminalParameter := success.terminalParameter
  post := success.post
  post_slot := success.post_slot
  post_direction := success.post_direction
  post_stage := success.post_stage
  post_label_eq_frozen := success.post_label_eq_terminal
  post_parameter_eq_terminal := success.post_parameter_eq_terminal
  outerLabels_exact := success.terminal_labels_exact directionCertificate
  parameter_safe := success.parameter_safe directionCertificate
  sourceStatus_eq := rfl
  stop := success.stop

end B2V3ActualTerminalSuccess

namespace B2V3ActualDirectionCertificate

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [DecidableEq Center] [Nonempty Center]
  [PseudoMetricSpace Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
  {T : Nat} {run : B2V3ActualDirectionRun Center Parameter T}
  {truth : Partition Center (Fin 3)} {reference : Parameter} {rTheta : ℝ}
  {slot : Slot} {direction : Direction} {maxSteps : Nat}
  {RejectReason : Type uReject}

/--
Exhaustive candidate-local terminal adapter: certified success carries the derived payload; fit
failure and uncertified maximum iteration are typed, payload-free failures.
-/
def terminalize
    (directionCertificate :
      B2V3ActualDirectionCertificate Center Parameter T run truth reference)
    (branch : B2V3TerminalBranch
      (B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T run truth reference rTheta slot direction maxSteps) maxSteps) :
    B2V3DirectionalSourceOutcome
      (B2V3CertifiedTerminalPayload Center Parameter Slot NormalizationContext FrozenContext Legal
        truth reference rTheta slot direction maxSteps .success)
      (B2V3TypedSourceFailure maxSteps) RejectReason :=
  match branch with
  | .success success => .accepted (success.toCertifiedPayload directionCertificate)
  | .fullRefitFailed stop => .failed ⟨.fullRefitFailed, stop⟩
  | .fullRefitMaxIterUncertified stop =>
      .failed ⟨.fullRefitMaxIterUncertified, stop⟩

end B2V3ActualDirectionCertificate

namespace B2V3ActualPairedRun

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [DecidableEq Center] [Nonempty Center]
  [PseudoMetricSpace Parameter]
  {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
  {FrozenContext : Type uFrozen}
  {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
  {T : Nat} {truth : Partition Center (Fin 3)}
  {referenceAB referenceBA : Parameter} {rThetaAB rThetaBA : ℝ}
  {slot : Slot} {maxSteps : Nat} {RejectReason : Type uReject}

/-- Map both actual terminal source branches to the existing paired typed-state adapter. -/
def terminalize
    (paired : B2V3ActualPairedRun Center Parameter T)
    (abCertificate :
      B2V3ActualDirectionCertificate Center Parameter T paired.ab truth referenceAB)
    (baCertificate :
      B2V3ActualDirectionCertificate Center Parameter T paired.ba truth referenceBA)
    (abBranch : B2V3TerminalBranch
      (B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T paired.ab truth referenceAB rThetaAB slot .AB maxSteps) maxSteps)
    (baBranch : B2V3TerminalBranch
      (B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T paired.ba truth referenceBA rThetaBA slot .BA maxSteps) maxSteps) :
    B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext Legal truth
      referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps where
  ab := abCertificate.terminalize abBranch
  ba := baCertificate.terminalize baBranch

/-- Every pair of source branches is exact-or-failure after R1--R3 terminalization. -/
theorem terminalized_exactOrFailure
    (paired : B2V3ActualPairedRun Center Parameter T)
    (abCertificate :
      B2V3ActualDirectionCertificate Center Parameter T paired.ab truth referenceAB)
    (baCertificate :
      B2V3ActualDirectionCertificate Center Parameter T paired.ba truth referenceBA)
    (abBranch : B2V3TerminalBranch
      (B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T paired.ab truth referenceAB rThetaAB slot .AB maxSteps) maxSteps)
    (baBranch : B2V3TerminalBranch
      (B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T paired.ba truth referenceBA rThetaBA slot .BA maxSteps) maxSteps) :
    (B2V3PairedSourceOutcome.toPairedCandidateState
      (paired.terminalize abCertificate baCertificate abBranch baBranch :
        B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext Legal truth
          referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps)).ExactOrFailure truth :=
  B2V3PairedSourceOutcome.exactOrFailure _

end B2V3ActualPairedRun

/--
On the coarse route, both derived terminal successes plus the actual paired guard certificate put
an exact post pair into the guarded pool.  A fallback to a coarse pre is deliberately not declared
safe here.
-/
theorem b2V3ActualGuard_exact_of_terminal_success
    {Center : Type uCenter} {Parameter : Type uParameter}
    [Fintype Center] [DecidableEq Center] [Nonempty Center]
    [PseudoMetricSpace Parameter]
    {Slot : Type uSlot} {NormalizationContext : Type uNormalization}
    {FrozenContext : Type uFrozen}
    {Legal : RiskRecord Center (Fin 3) Slot Parameter NormalizationContext FrozenContext → Prop}
    {T : Nat} {truth : Partition Center (Fin 3)}
    {referenceAB referenceBA : Parameter} {rThetaAB rThetaBA : ℝ}
    {slot : Slot} {maxSteps : Nat} {RejectReason : Type uReject}
    (paired : B2V3ActualPairedRun Center Parameter T)
    (abCertificate :
      B2V3ActualDirectionCertificate Center Parameter T paired.ab truth referenceAB)
    (baCertificate :
      B2V3ActualDirectionCertificate Center Parameter T paired.ba truth referenceBA)
    (abSuccess :
      B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T paired.ab truth referenceAB rThetaAB slot .AB maxSteps)
    (baSuccess :
      B2V3ActualTerminalSuccess Center Parameter Slot NormalizationContext FrozenContext Legal
        T paired.ba truth referenceBA rThetaBA slot .BA maxSteps)
    (postOK : PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext
      FrozenContext Legal slot .post → Prop)
    (preOK : PairedRiskCandidates Center (Fin 3) Slot Parameter NormalizationContext
      FrozenContext Legal slot .pre → Prop)
    (rejectReason : RejectReason)
    (preState : PairedCandidateState Center (Fin 3) Slot Parameter NormalizationContext
      FrozenContext Legal (B2V3AdapterFailure RejectReason) slot .pre)
    (hPostOK : ∀ postPair,
      ((B2V3PairedSourceOutcome.toPairedCandidateState
        (paired.terminalize abCertificate baCertificate (.success abSuccess) (.success baSuccess) :
          B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext Legal truth
            referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps)).toOption =
        some postPair) → postOK postPair) :
    (atomicGuard postOK preOK rejectReason preState
      (B2V3PairedSourceOutcome.toPairedCandidateState
        (paired.terminalize abCertificate baCertificate (.success abSuccess) (.success baSuccess) :
          B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext Legal truth
            referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps))).Exact truth := by
  let outcome :=
    (paired.terminalize abCertificate baCertificate (.success abSuccess) (.success baSuccess) :
      B2V3PairedSourceOutcome Center Parameter Slot NormalizationContext FrozenContext Legal truth
        referenceAB referenceBA rThetaAB rThetaBA RejectReason slot maxSteps)
  have hCertified : outcome.BothCertified := by
    refine ⟨?_, ?_⟩
    · refine ⟨abSuccess.toCertifiedPayload abCertificate, ?_⟩
      rfl
    · refine ⟨baSuccess.toCertifiedPayload baCertificate, ?_⟩
      rfl
  exact b2V3CoarseGuard_exact_of_certified postOK preOK rejectReason preState outcome
    hCertified hPostOK

end BiostatFormal.HighDimensional.Multiview
