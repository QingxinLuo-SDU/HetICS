import BiostatFormal.HighDimensional.Multiview.B2Refinement
import Mathlib.Algebra.Order.Field.GeomSum

/-!
# B2-v3 coarse-attraction refinement

This file starts the source-faithful coarse-attraction layer.  It fixes one label alignment for a
whole directional run, exposes the four Python parameter blocks and the complete Bernoulli hard
score, and keeps label contamination, statistical score error, and finite-step optimization error
separate.

The existing exact-start A3 results are not changed by this module.
-/

namespace BiostatFormal.HighDimensional.Multiview

open scoped InnerProductSpace

universe uObservation uCenter uFeature uParameter

/-- The four parameter blocks used by frozen v3: `pi`, `gamma`, intercepts, and class betas. -/
abbrev B2V3ParameterState (Feature : Type uFeature) :=
  (Fin 3 → ℝ) × ℝ × (Fin 3 → ℝ) × (Fin 3 → Feature → ℝ)

/-- Finite Bernoulli observations nested inside label-bearing centers. -/
structure B2V3BernoulliData
    (Observation : Type uObservation) (Center : Type uCenter) (Feature : Type uFeature) where
  center : Observation → Center
  treatment : Observation → ℝ
  feature : Observation → Feature → ℝ
  outcome : Observation → ℝ

/-- The actual v3 class-specific linear predictor. -/
noncomputable def b2V3LinearPredictor
    {Observation : Type uObservation} {Center : Type uCenter} {Feature : Type uFeature}
    [Fintype Feature]
    (data : B2V3BernoulliData Observation Center Feature)
    (theta : B2V3ParameterState Feature) (observation : Observation) (component : Fin 3) : ℝ :=
  theta.2.2.1 component + theta.2.1 * data.treatment observation +
    ∑ feature, theta.2.2.2 component feature * data.feature observation feature

/--
The complete-data center/class score used by the constrained assignment: one center-level
`log pi` term plus the Bernoulli log likelihood of all observations in that center.
-/
noncomputable def b2V3CenterScore
    {Observation : Type uObservation} {Center : Type uCenter} {Feature : Type uFeature}
    [Fintype Observation] [Fintype Feature] [DecidableEq Center]
    (data : B2V3BernoulliData Observation Center Feature)
    (theta : B2V3ParameterState Feature) (center : Center) (component : Fin 3) : ℝ :=
  Real.log (theta.1 component) +
    ∑ observation,
      if data.center observation = center then
        data.outcome observation * b2V3LinearPredictor data theta observation component -
          Real.log (1 + Real.exp (b2V3LinearPredictor data theta observation component))
      else 0

/-- Source stages represented in the mathematical shadow state. -/
inductive B2V3RefinementStage where
  | initial
  | ordinaryFit
  | ordinaryAssignment
  | terminalRefit
  | guarded
  deriving DecidableEq, Repr

/-- One source-shaped directional refinement state with a fixed global label alignment. -/
structure B2V3RefinementState (Center : Type uCenter) (Feature : Type uFeature) where
  theta : B2V3ParameterState Feature
  labels : Partition Center (Fin 3)
  alignment : Equiv.Perm (Fin 3)
  stage : B2V3RefinementStage
  outerStep : Nat

/-- Apply the once-for-the-run permutation to the truth partition. -/
def alignedTruth
    {Center : Type uCenter}
    (alignment : Equiv.Perm (Fin 3)) (truth : Partition Center (Fin 3)) :
    Partition Center (Fin 3) :=
  fun center => alignment (truth center)

/-- Centers mismatched against one fixed global alignment. -/
def alignedMismatchCenters
    {Center : Type uCenter} [Fintype Center] [DecidableEq Center]
    (alignment : Equiv.Perm (Fin 3))
    (candidate truth : Partition Center (Fin 3)) : Finset Center :=
  Finset.univ.filter fun center => candidate center ≠ alignedTruth alignment truth center

/--
Mismatch fraction against one fixed alignment.  Unlike `labelDistance`, this definition does not
minimize over a fresh permutation at every iteration.
-/
noncomputable def alignedLabelError
    {Center : Type uCenter} [Fintype Center] [DecidableEq Center]
    (alignment : Equiv.Perm (Fin 3))
    (candidate truth : Partition Center (Fin 3)) : ℝ :=
  (alignedMismatchCenters alignment candidate truth).card /
    Fintype.card Center

section FitPrimitives

/--
Low-level RSC/score/fit reduction for one realized fixed-label return.  The three upper bounds are
separate so the statistical score term cannot be relabelled as finite-iteration error.
-/
structure B2V3FitPrimitiveCertificate
    (parameterError labelError scoreMagnitude fitResidual
      kappa bLabel gStat epsilonOpt : ℝ) : Prop where
  kappa_positive : 0 < kappa
  parameterError_nonnegative : 0 ≤ parameterError
  labelError_nonnegative : 0 ≤ labelError
  bLabel_nonnegative : 0 ≤ bLabel
  gStat_nonnegative : 0 ≤ gStat
  epsilonOpt_nonnegative : 0 ≤ epsilonOpt
  rsc_to_error : kappa * parameterError ≤ scoreMagnitude + fitResidual
  score_decomposition : scoreMagnitude ≤ bLabel * labelError + gStat
  fit_residual : fitResidual ≤ epsilonOpt

/-- Coefficient multiplying the aligned label-contamination error. -/
noncomputable def v3ContaminationCoefficient (bLabel kappa : ℝ) : ℝ :=
  bLabel / kappa

/-- Statistical score radius, kept separate from optimization error. -/
noncomputable def v3StatisticalRadius (gStat kappa : ℝ) : ℝ :=
  gStat / kappa

/-- Finite-step fit/optimization radius. -/
noncomputable def v3OptimizationRadius (epsilonOpt kappa : ℝ) : ℝ :=
  epsilonOpt / kappa

/--
R1 for a coarse fixed-label call.  RSC plus separate score-contamination and fit-residual bounds
give the declared linear parameter error radius.
-/
theorem parameterError_le_of_v3FitPrimitives
    {parameterError labelError scoreMagnitude fitResidual
      kappa bLabel gStat epsilonOpt : ℝ}
    (h : B2V3FitPrimitiveCertificate parameterError labelError scoreMagnitude fitResidual
      kappa bLabel gStat epsilonOpt) :
    parameterError ≤
      v3ContaminationCoefficient bLabel kappa * labelError +
      v3StatisticalRadius gStat kappa +
      v3OptimizationRadius epsilonOpt kappa := by
  have hReduced :
      kappa * parameterError ≤ bLabel * labelError + gStat + epsilonOpt := by
    linarith [h.rsc_to_error, h.score_decomposition, h.fit_residual]
  have hDiv :
      parameterError ≤ (bLabel * labelError + gStat + epsilonOpt) / kappa := by
    apply (le_div_iff₀ h.kappa_positive).2
    simpa [mul_comm] using hReduced
  rw [v3ContaminationCoefficient, v3StatisticalRadius, v3OptimizationRadius]
  calc
    parameterError ≤ (bLabel * labelError + gStat + epsilonOpt) / kappa := hDiv
    _ = bLabel / kappa * labelError + gStat / kappa + epsilonOpt / kappa := by
      field_simp [ne_of_gt h.kappa_positive]

/-- Gradient-step factor obtained from restricted curvature and restricted smoothness constants. -/
noncomputable def v3GradientFactor (eta kappa rss : ℝ) : ℝ :=
  Real.sqrt (1 - 2 * eta * kappa + eta ^ 2 * rss ^ 2)

/-- The hard-threshold projection factor multiplied by the gradient-step factor. -/
noncomputable def v3IHTFactor (thresholdFactor eta kappa rss : ℝ) : ℝ :=
  thresholdFactor * v3GradientFactor eta kappa rss

/--
Certificate for one actual accepted finite-grid, preconditioned IHT step.  It records RSC/RSS and
the hard-threshold projection inequality; it does not contain a label-contraction conclusion.
-/
structure B2V3AcceptedFitStep
    (Parameter : Type uParameter) [NormedAddCommGroup Parameter] [InnerProductSpace ℝ Parameter]
    (direction : Parameter → Parameter) (reference current next : Parameter)
    (alignedError bLabel gStat epsilonFinite initialStep backtrackingFactor
      kappa rss thresholdFactor : ℝ) (maxBacktracking : Nat) where
  backtrackingIndex : Fin (maxBacktracking + 1)
  eta : ℝ
  eta_eq_grid : eta = initialStep * backtrackingFactor ^ backtrackingIndex.val
  eta_positive : 0 < eta
  kappa_positive : 0 < kappa
  rss_nonnegative : 0 ≤ rss
  factorSquare_nonnegative : 0 ≤ 1 - 2 * eta * kappa + eta ^ 2 * rss ^ 2
  rsc :
    kappa * ‖current - reference‖ ^ 2 ≤
      ⟪direction current - direction reference, current - reference⟫_ℝ
  rss_bound :
    ‖direction current - direction reference‖ ≤ rss * ‖current - reference‖
  truthDirection_bound : ‖direction reference‖ ≤ bLabel * alignedError + gStat
  thresholdFactor_nonnegative : 0 ≤ thresholdFactor
  epsilonFinite_nonnegative : 0 ≤ epsilonFinite
  threshold_bound :
    ‖next - reference‖ ≤
      thresholdFactor * ‖current - eta • direction current - reference‖ + epsilonFinite

namespace B2V3AcceptedFitStep

variable {Parameter : Type uParameter}
  [NormedAddCommGroup Parameter] [InnerProductSpace ℝ Parameter]
  {direction : Parameter → Parameter} {reference current next : Parameter}
  {alignedError bLabel gStat epsilonFinite initialStep backtrackingFactor
    kappa rss thresholdFactor : ℝ} {maxBacktracking : Nat}

/-- RSC and RSS give the finite accepted gradient-step factor before hard thresholding. -/
theorem gradientDifference_step_bound
    (step : B2V3AcceptedFitStep Parameter direction reference current next
      alignedError bLabel gStat epsilonFinite initialStep backtrackingFactor
      kappa rss thresholdFactor maxBacktracking) :
    ‖(current - reference) - step.eta • (direction current - direction reference)‖ ≤
      v3GradientFactor step.eta kappa rss * ‖current - reference‖ := by
  let error := current - reference
  let gradientDifference := direction current - direction reference
  have hExpansion :
      ‖error - step.eta • gradientDifference‖ ^ 2 =
        ‖error‖ ^ 2 -
          2 * step.eta * ⟪gradientDifference, error⟫_ℝ +
          step.eta ^ 2 * ‖gradientDifference‖ ^ 2 := by
    calc
      ‖error - step.eta • gradientDifference‖ ^ 2 =
          ⟪error - step.eta • gradientDifference,
            error - step.eta • gradientDifference⟫_ℝ := by
        symm
        exact real_inner_self_eq_norm_sq _
      _ = ⟪error, error⟫_ℝ -
            2 * ⟪error, step.eta • gradientDifference⟫_ℝ +
            ⟪step.eta • gradientDifference, step.eta • gradientDifference⟫_ℝ :=
        real_inner_sub_sub_self _ _
      _ = ‖error‖ ^ 2 -
            2 * step.eta * ⟪gradientDifference, error⟫_ℝ +
            step.eta ^ 2 * ‖gradientDifference‖ ^ 2 := by
        rw [real_inner_self_eq_norm_sq]
        rw [real_inner_smul_right]
        rw [real_inner_comm error gradientDifference]
        rw [real_inner_self_eq_norm_sq]
        simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg step.eta_positive.le]
        ring
  have hRssSquare :
      ‖gradientDifference‖ ^ 2 ≤ (rss * ‖error‖) ^ 2 := by
    apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg step.rss_nonnegative (norm_nonneg _))).2
    simpa [gradientDifference, error] using step.rss_bound
  have hRscScaled :
      2 * step.eta * (kappa * ‖error‖ ^ 2) ≤
        2 * step.eta * ⟪gradientDifference, error⟫_ℝ := by
    apply mul_le_mul_of_nonneg_left
    · simpa [gradientDifference, error] using step.rsc
    · exact mul_nonneg (by norm_num) step.eta_positive.le
  have hRssScaled :
      step.eta ^ 2 * ‖gradientDifference‖ ^ 2 ≤
        step.eta ^ 2 * (rss * ‖error‖) ^ 2 :=
    mul_le_mul_of_nonneg_left hRssSquare (sq_nonneg _)
  have hSquareBound :
      ‖error - step.eta • gradientDifference‖ ^ 2 ≤
        (1 - 2 * step.eta * kappa + step.eta ^ 2 * rss ^ 2) * ‖error‖ ^ 2 := by
    rw [hExpansion]
    nlinarith
  have hTargetNonnegative :
      0 ≤ v3GradientFactor step.eta kappa rss * ‖error‖ :=
    mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _)
  apply (sq_le_sq₀ (norm_nonneg _) hTargetNonnegative).mp
  rw [mul_pow, v3GradientFactor, Real.sq_sqrt step.factorSquare_nonnegative]
  exact hSquareBound

/-- One accepted preconditioned IHT step with contamination, statistical, and finite-step terms. -/
theorem one_step_parameter_error
    (step : B2V3AcceptedFitStep Parameter direction reference current next
      alignedError bLabel gStat epsilonFinite initialStep backtrackingFactor
      kappa rss thresholdFactor maxBacktracking) :
    ‖next - reference‖ ≤
      v3IHTFactor thresholdFactor step.eta kappa rss * ‖current - reference‖ +
        thresholdFactor * step.eta * (bLabel * alignedError + gStat) + epsilonFinite := by
  have hDecomposition :
      current - step.eta • direction current - reference =
        ((current - reference) - step.eta • (direction current - direction reference)) -
          step.eta • direction reference := by
    rw [smul_sub]
    abel
  have hTruthDirection :
      ‖step.eta • direction reference‖ ≤
        step.eta * (bLabel * alignedError + gStat) := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg step.eta_positive.le]
    exact mul_le_mul_of_nonneg_left step.truthDirection_bound step.eta_positive.le
  have hRawStep :
      ‖current - step.eta • direction current - reference‖ ≤
        v3GradientFactor step.eta kappa rss * ‖current - reference‖ +
          step.eta * (bLabel * alignedError + gStat) := by
    rw [hDecomposition]
    exact (norm_sub_le _ _).trans
      (add_le_add step.gradientDifference_step_bound hTruthDirection)
  calc
    ‖next - reference‖ ≤
        thresholdFactor * ‖current - step.eta • direction current - reference‖ +
          epsilonFinite := step.threshold_bound
    _ ≤ thresholdFactor *
          (v3GradientFactor step.eta kappa rss * ‖current - reference‖ +
            step.eta * (bLabel * alignedError + gStat)) + epsilonFinite := by
      exact add_le_add
        (mul_le_mul_of_nonneg_left hRawStep step.thresholdFactor_nonnegative)
        (le_refl epsilonFinite)
    _ = v3IHTFactor thresholdFactor step.eta kappa rss * ‖current - reference‖ +
          thresholdFactor * step.eta * (bLabel * alignedError + gStat) +
          epsilonFinite := by
      rw [v3IHTFactor]
      ring

end B2V3AcceptedFitStep

end FitPrimitives

section Fin3AssignmentRecurrence

variable {Center : Type uCenter} {Parameter : Type uParameter}
  [Fintype Center] [DecidableEq Center]
  [PseudoMetricSpace Parameter]

/--
Complete source-facing `Fin 3` assignment evidence for one common returned parameter and one fixed
alignment.  Pairwise Lipschitz control is present for every center and both wrong components.
-/
structure B2V3Fin3AssignmentCertificate
    (alignment : Equiv.Perm (Fin 3))
    (truth returned : Partition Center (Fin 3))
    (reference estimate : Parameter) where
  feasible : Partition Center (Fin 3) → Prop
  score : Center → Fin 3 → Parameter → ℝ
  gamma0 : ℝ
  margin : Fin3AssignmentMargin Center Parameter (alignedTruth alignment truth)
    score reference gamma0
  truth_feasible : feasible (alignedTruth alignment truth)
  pairwiseLipschitz : Center → Fin 3 → ℝ
  pairwiseLipschitz_nonnegative : ∀ center component,
    0 ≤ pairwiseLipschitz center component
  pairwise_bound : ∀ center wrongComponent,
    wrongComponent ≠ alignedTruth alignment truth center →
      |(score center (alignedTruth alignment truth center) estimate -
          score center wrongComponent estimate) -
        (score center (alignedTruth alignment truth center) reference -
          score center wrongComponent reference)| ≤
        pairwiseLipschitz center wrongComponent * dist estimate reference
  etaAsn : ℝ
  regret : Fin3AssignmentRegretCertificate Center feasible
    (fin3AssignmentObjective score estimate) returned etaAsn

/-- Sum of the actual pairwise Lipschitz constants over newly mismatched centers. -/
noncomputable def v3AssignmentLipschitzSum
    {alignment : Equiv.Perm (Fin 3)}
    {truth returned : Partition Center (Fin 3)}
    {reference estimate : Parameter}
    (certificate : B2V3Fin3AssignmentCertificate alignment truth returned reference estimate) :
    ℝ :=
  ∑ center ∈ alignedMismatchCenters alignment returned truth,
    certificate.pairwiseLipschitz center (returned center)

theorem v3AssignmentLipschitzSum_nonnegative
    {alignment : Equiv.Perm (Fin 3)}
    {truth returned : Partition Center (Fin 3)}
    {reference estimate : Parameter}
    (certificate : B2V3Fin3AssignmentCertificate alignment truth returned reference estimate) :
    0 ≤ v3AssignmentLipschitzSum certificate := by
  classical
  rw [v3AssignmentLipschitzSum]
  exact Finset.sum_nonneg fun center _ =>
    certificate.pairwiseLipschitz_nonnegative center (returned center)

/--
The complete multiclass margin and total assignment regret control the full fixed-alignment
mismatch mass.  The theorem sums over all wrong returned components rather than reducing to a
binary contrast.
-/
theorem fin3_mismatch_mass_le_of_margin_regret
    {alignment : Equiv.Perm (Fin 3)}
    {truth returned : Partition Center (Fin 3)}
    {reference estimate : Parameter}
    (certificate : B2V3Fin3AssignmentCertificate alignment truth returned reference estimate) :
    ((alignedMismatchCenters alignment returned truth).card : ℝ) * certificate.gamma0 ≤
      v3AssignmentLipschitzSum certificate * dist estimate reference + certificate.etaAsn := by
  classical
  let aligned : Partition Center (Fin 3) := alignedTruth alignment truth
  let bad : Finset Center := alignedMismatchCenters alignment returned truth
  have hPoint : ∀ center ∈ bad,
      certificate.gamma0 ≤
        (certificate.score center (aligned center) estimate -
          certificate.score center (returned center) estimate) +
        certificate.pairwiseLipschitz center (returned center) * dist estimate reference := by
    intro center hCenter
    have hWrong : returned center ≠ aligned center := by
      simpa [bad, aligned, alignedMismatchCenters] using (Finset.mem_filter.mp hCenter).2
    have hMargin := certificate.margin.true_minus_wrong center (returned center) hWrong
    have hPerturbation := certificate.pairwise_bound center (returned center) hWrong
    have hLower := neg_le_of_abs_le hPerturbation
    have hMargin' :
        certificate.gamma0 ≤
          certificate.score center (aligned center) reference -
            certificate.score center (returned center) reference := by
      simpa [aligned] using hMargin
    have hLower' :
        -(certificate.pairwiseLipschitz center (returned center) *
            dist estimate reference) ≤
          (certificate.score center (aligned center) estimate -
              certificate.score center (returned center) estimate) -
            (certificate.score center (aligned center) reference -
              certificate.score center (returned center) reference) := by
      simpa [aligned] using hLower
    linarith
  have hSummed := Finset.sum_le_sum fun center hCenter => hPoint center hCenter
  have hObjectiveDifference :
      fin3AssignmentObjective certificate.score estimate aligned -
          fin3AssignmentObjective certificate.score estimate returned =
        ∑ center ∈ bad,
          (certificate.score center (aligned center) estimate -
            certificate.score center (returned center) estimate) := by
    rw [fin3AssignmentObjective, fin3AssignmentObjective, ← Finset.sum_sub_distrib]
    symm
    rw [show bad = Finset.univ.filter (fun center => returned center ≠ aligned center) by
      ext center
      simp [bad, aligned, alignedMismatchCenters]]
    rw [Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro center _
    by_cases hWrong : returned center ≠ aligned center
    · simp [hWrong]
    · simp [not_ne_iff.mp hWrong]
  have hRegretAtTruth :
      fin3AssignmentObjective certificate.score estimate aligned -
          fin3AssignmentObjective certificate.score estimate returned ≤ certificate.etaAsn := by
    exact certificate.regret.global_regret aligned certificate.truth_feasible
  rw [hObjectiveDifference] at hRegretAtTruth
  have hLipschitzSum :
      ∑ center ∈ bad,
          certificate.pairwiseLipschitz center (returned center) * dist estimate reference =
        v3AssignmentLipschitzSum certificate * dist estimate reference := by
    rw [← Finset.sum_mul]
    rfl
  rw [Finset.sum_add_distrib, hLipschitzSum] at hSummed
  have hFinal :
      ∑ center ∈ bad, certificate.gamma0 ≤
        v3AssignmentLipschitzSum certificate * dist estimate reference + certificate.etaAsn := by
    linarith
  simpa [bad, mul_comm] using hFinal

variable [Nonempty Center]

/-- Multiplying the fixed-alignment mismatch fraction by the center count recovers its numerator. -/
theorem card_mul_alignedLabelError
    (alignment : Equiv.Perm (Fin 3))
    (candidate truth : Partition Center (Fin 3)) :
    (Fintype.card Center : ℝ) * alignedLabelError alignment candidate truth =
      (alignedMismatchCenters alignment candidate truth).card := by
  rw [alignedLabelError]
  field_simp [Nat.cast_ne_zero.mpr Fintype.card_ne_zero]

/-- Normalized form of the complete multiclass mismatch-mass inequality. -/
theorem fin3_alignedLabelError_mass_le_of_margin_regret
    {alignment : Equiv.Perm (Fin 3)}
    {truth returned : Partition Center (Fin 3)}
    {reference estimate : Parameter}
    (certificate : B2V3Fin3AssignmentCertificate alignment truth returned reference estimate) :
    (Fintype.card Center : ℝ) * certificate.gamma0 *
        alignedLabelError alignment returned truth ≤
      v3AssignmentLipschitzSum certificate * dist estimate reference + certificate.etaAsn := by
  calc
    (Fintype.card Center : ℝ) * certificate.gamma0 *
        alignedLabelError alignment returned truth =
        ((Fintype.card Center : ℝ) * alignedLabelError alignment returned truth) *
          certificate.gamma0 := by ring
    _ = ((alignedMismatchCenters alignment returned truth).card : ℝ) *
          certificate.gamma0 := by
      rw [card_mul_alignedLabelError]
    _ ≤ v3AssignmentLipschitzSum certificate * dist estimate reference +
          certificate.etaAsn := fin3_mismatch_mass_le_of_margin_regret certificate

/-- Label-contraction coefficient derived from fit contamination and assignment constants. -/
noncomputable def v3Rho
    (centerCount gamma0 lipschitzSum contaminationCoefficient : ℝ) : ℝ :=
  lipschitzSum * contaminationCoefficient / (centerCount * gamma0)

/-- Statistical, optimization, and assignment-regret floor in the label recurrence. -/
noncomputable def v3Xi
    (centerCount gamma0 lipschitzSum bStat bOpt etaAsn : ℝ) : ℝ :=
  (lipschitzSum * (bStat + bOpt) + etaAsn) / (centerCount * gamma0)

/--
The actual parameter-radius contract and complete `Fin 3` assignment inequality derive the affine
label recurrence.  `rho` and `xi` are definitions from the displayed low-level constants.
-/
theorem labelError_le_rho_mul_add_xi
    {alignment : Equiv.Perm (Fin 3)}
    {truth currentLabels returned : Partition Center (Fin 3)}
    {reference estimate : Parameter}
    (certificate : B2V3Fin3AssignmentCertificate alignment truth returned reference estimate)
    (contaminationCoefficient bStat bOpt : ℝ)
    (hParameter : dist estimate reference ≤
      contaminationCoefficient * alignedLabelError alignment currentLabels truth +
        bStat + bOpt) :
    alignedLabelError alignment returned truth ≤
      v3Rho (Fintype.card Center) certificate.gamma0
          (v3AssignmentLipschitzSum certificate) contaminationCoefficient *
        alignedLabelError alignment currentLabels truth +
      v3Xi (Fintype.card Center) certificate.gamma0
        (v3AssignmentLipschitzSum certificate) bStat bOpt certificate.etaAsn := by
  have hCenterCountPositive : 0 < (Fintype.card Center : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hDenominatorPositive :
      0 < (Fintype.card Center : ℝ) * certificate.gamma0 :=
    mul_pos hCenterCountPositive certificate.margin.gamma0_positive
  have hScaledParameter :
      v3AssignmentLipschitzSum certificate * dist estimate reference ≤
        v3AssignmentLipschitzSum certificate *
          (contaminationCoefficient * alignedLabelError alignment currentLabels truth +
            bStat + bOpt) :=
    mul_le_mul_of_nonneg_left hParameter
      (v3AssignmentLipschitzSum_nonnegative certificate)
  have hMass := fin3_alignedLabelError_mass_le_of_margin_regret certificate
  have hNumerator :
      (Fintype.card Center : ℝ) * certificate.gamma0 *
          alignedLabelError alignment returned truth ≤
        v3AssignmentLipschitzSum certificate *
            (contaminationCoefficient * alignedLabelError alignment currentLabels truth +
              bStat + bOpt) +
          certificate.etaAsn :=
    hMass.trans (add_le_add hScaledParameter (le_refl certificate.etaAsn))
  have hDivide :
      alignedLabelError alignment returned truth ≤
        (v3AssignmentLipschitzSum certificate *
            (contaminationCoefficient * alignedLabelError alignment currentLabels truth +
              bStat + bOpt) + certificate.etaAsn) /
          ((Fintype.card Center : ℝ) * certificate.gamma0) := by
    apply (le_div_iff₀ hDenominatorPositive).2
    calc
      alignedLabelError alignment returned truth *
          ((Fintype.card Center : ℝ) * certificate.gamma0) =
          (Fintype.card Center : ℝ) * certificate.gamma0 *
            alignedLabelError alignment returned truth := by ring
      _ ≤ v3AssignmentLipschitzSum certificate *
            (contaminationCoefficient * alignedLabelError alignment currentLabels truth +
              bStat + bOpt) + certificate.etaAsn := hNumerator
  rw [v3Rho, v3Xi]
  calc
    alignedLabelError alignment returned truth ≤
        (v3AssignmentLipschitzSum certificate *
            (contaminationCoefficient * alignedLabelError alignment currentLabels truth +
              bStat + bOpt) + certificate.etaAsn) /
          ((Fintype.card Center : ℝ) * certificate.gamma0) := hDivide
    _ = v3AssignmentLipschitzSum certificate * contaminationCoefficient /
            ((Fintype.card Center : ℝ) * certificate.gamma0) *
          alignedLabelError alignment currentLabels truth +
        (v3AssignmentLipschitzSum certificate * (bStat + bOpt) + certificate.etaAsn) /
          ((Fintype.card Center : ℝ) * certificate.gamma0) := by
      field_simp [ne_of_gt hDenominatorPositive]
      ring

/-- The exact-label radius is strictly inside the coarse attraction radius. -/
structure B2V3AttractionRadii where
  rExact : ℝ
  rAttr : ℝ
  exact_positive : 0 < rExact
  exact_lt_attraction : rExact < rAttr

/-- Iteration of the affine label-error recurrence, without any independence assumption. -/
theorem iterate_affine_error_bound
    (error : ℕ → ℝ) (rho xi : ℝ) (T : ℕ)
    (hRhoNonnegative : 0 ≤ rho)
    (hStep : ∀ t, error (t + 1) ≤ rho * error t + xi) :
    error T ≤ rho ^ T * error 0 +
      (∑ j ∈ Finset.range T, rho ^ j) * xi := by
  induction T with
  | zero => simp
  | succ T hInduction =>
      calc
        error (T + 1) ≤ rho * error T + xi := hStep T
        _ ≤ rho *
              (rho ^ T * error 0 +
                (∑ j ∈ Finset.range T, rho ^ j) * xi) + xi :=
          add_le_add (mul_le_mul_of_nonneg_left hInduction hRhoNonnegative) (le_refl xi)
        _ = rho ^ (T + 1) * error 0 +
              (∑ j ∈ Finset.range (T + 1), rho ^ j) * xi := by
          rw [geom_sum_succ]
          ring

/-- Closed geometric-series form of the finite-step recurrence. -/
theorem iterate_affine_closed_form
    (error : ℕ → ℝ) (rho xi : ℝ) (T : ℕ)
    (hRhoNonnegative : 0 ≤ rho)
    (hRhoLessThanOne : rho < 1)
    (hStep : ∀ t, error (t + 1) ≤ rho * error t + xi) :
    error T ≤ rho ^ T * error 0 +
      ((1 - rho ^ T) / (1 - rho)) * xi := by
  calc
    error T ≤ rho ^ T * error 0 +
        ((rho ^ T - 1) / (rho - 1)) * xi := by
      rw [← geom_sum_eq (ne_of_lt hRhoLessThanOne)]
      exact iterate_affine_error_bound error rho xi T hRhoNonnegative hStep
    _ = rho ^ T * error 0 + ((1 - rho ^ T) / (1 - rho)) * xi := by
      rw [show rho ^ T - 1 = -(1 - rho ^ T) by ring,
        show rho - 1 = -(1 - rho) by ring, neg_div_neg_eq]

/-- The displayed attraction inequality makes the coarse basin invariant at every finite step. -/
theorem attraction_invariant
    (error : ℕ → ℝ) (rho xi rAttr : ℝ)
    (hRhoNonnegative : 0 ≤ rho)
    (hStep : ∀ t, error (t + 1) ≤ rho * error t + xi)
    (hStart : error 0 ≤ rAttr)
    (hInvariant : rho * rAttr + xi ≤ rAttr) :
    ∀ t, error t ≤ rAttr := by
  intro t
  induction t with
  | zero => exact hStart
  | succ t hInduction =>
      exact (hStep t).trans <|
        (add_le_add (mul_le_mul_of_nonneg_left hInduction hRhoNonnegative)
          (le_refl xi)).trans hInvariant

/-- A nonzero fixed-alignment label error has mass at least one center. -/
theorem labelError_ge_inv_card_of_ne
    (alignment : Equiv.Perm (Fin 3))
    (candidate truth : Partition Center (Fin 3))
    (hNe : candidate ≠ alignedTruth alignment truth) :
    1 / (Fintype.card Center : ℝ) ≤ alignedLabelError alignment candidate truth := by
  classical
  have hCenterCountPositive : 0 < (Fintype.card Center : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hExists : ∃ center, candidate center ≠ alignedTruth alignment truth center := by
    by_contra hNoCenter
    apply hNe
    funext center
    by_contra hWrong
    exact hNoCenter ⟨center, hWrong⟩
  rcases hExists with ⟨center, hWrong⟩
  have hMembership : center ∈ alignedMismatchCenters alignment candidate truth := by
    simp [alignedMismatchCenters, hWrong]
  have hCardPositive : 0 < (alignedMismatchCenters alignment candidate truth).card :=
    Finset.card_pos.mpr ⟨center, hMembership⟩
  have hOneLeCard : (1 : ℝ) ≤ (alignedMismatchCenters alignment candidate truth).card := by
    exact_mod_cast hCardPositive
  rw [alignedLabelError]
  exact (div_le_div_iff_of_pos_right hCenterCountPositive).2 hOneLeCard

/-- Falling below one-center mass forces exact recovery modulo the fixed global alignment. -/
theorem exact_of_labelError_lt_inv_card
    (alignment : Equiv.Perm (Fin 3))
    (candidate truth : Partition Center (Fin 3))
    (hSmall : alignedLabelError alignment candidate truth <
      1 / (Fintype.card Center : ℝ)) :
    Exact candidate truth := by
  have hEqual : candidate = alignedTruth alignment truth := by
    by_contra hNe
    exact (not_le_of_gt hSmall) (labelError_ge_inv_card_of_ne alignment candidate truth hNe)
  refine ⟨alignment.symm, ?_⟩
  funext center
  simp [relabel, hEqual, alignedTruth]

/--
The finite source iteration enters the exact region when its derived affine bound is below the
declared exact radius.  This is a finite-step conclusion, not a stationarity claim.
-/
theorem finite_step_enters_exact_region
    (alignment : Equiv.Perm (Fin 3))
    (truth : Partition Center (Fin 3))
    (labels : ℕ → Partition Center (Fin 3))
    (rho xi : ℝ) (radii : B2V3AttractionRadii) (T : ℕ)
    (hRhoNonnegative : 0 ≤ rho)
    (hRhoLessThanOne : rho < 1)
    (hStep : ∀ t,
      alignedLabelError alignment (labels (t + 1)) truth ≤
        rho * alignedLabelError alignment (labels t) truth + xi)
    (hStart : alignedLabelError alignment (labels 0) truth ≤ radii.rAttr)
    (hExactGrid : radii.rExact ≤ 1 / (Fintype.card Center : ℝ))
    (hFinite :
      rho ^ T * radii.rAttr + ((1 - rho ^ T) / (1 - rho)) * xi < radii.rExact) :
    Exact (labels T) truth := by
  let error : ℕ → ℝ := fun t => alignedLabelError alignment (labels t) truth
  have hClosed := iterate_affine_closed_form error rho xi T
    hRhoNonnegative hRhoLessThanOne hStep
  have hPowNonnegative : 0 ≤ rho ^ T := pow_nonneg hRhoNonnegative T
  have hAttractionBound :
      error T ≤ rho ^ T * radii.rAttr + ((1 - rho ^ T) / (1 - rho)) * xi :=
    hClosed.trans <| add_le_add
      (mul_le_mul_of_nonneg_left hStart hPowNonnegative)
      (le_refl (((1 - rho ^ T) / (1 - rho)) * xi))
  have hInsideExact : error T < radii.rExact := hAttractionBound.trans_lt hFinite
  have hBelowGrid : error T < 1 / (Fintype.card Center : ℝ) :=
    hInsideExact.trans_le hExactGrid
  exact exact_of_labelError_lt_inv_card alignment (labels T) truth hBelowGrid

end Fin3AssignmentRecurrence

end BiostatFormal.HighDimensional.Multiview
