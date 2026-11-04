import BiostatFormal.HighDimensional.Multiview.UniformRiskDeviation
import BiostatFormal.Probability.WeightedHoeffding
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# B2 constants and conditioning layer for Appendix A2

This file specializes the deterministic parts of Appendix A2 to B2's fixed 68-slot envelope.  It
proves the clipped Bernoulli and capped F-loss envelopes, the replacement constant for a frozen
equal-center stability average, the exact weighted-Hoeffding pointwise tail for a frozen affine
normalizer, and the measure-theoretic tower/union layer for the two directions.

The weighted-Hoeffding theorem applies to the revised preregistered P0 contract.  It still requires
the source-facing dependency facts: candidates, predictors, weights, and affine normalizer must be
frozen outside the current evaluation half, and the evaluation center blocks must be conditionally
independent.  It does not apply to the superseded validation-selected nonlinear normalizer or rerun
global clustering statistic.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uCenter uContextAB uContextBA uOmega uCandidate uSample

/-- The two cross-validation directions used by B2. -/
inductive B2Direction where
  | ab
  | ba
  deriving DecidableEq, Fintype, Repr

/-- The two candidate states that are charged inside each generator slot. -/
inductive B2Stage where
  | pre
  | post
  deriving DecidableEq, Fintype, Repr

/-- Fixed generator-slot indices, before any deduplication or validation guard. -/
abbrev B2Slot := Fin 17

/-- One direction contains every generator slot at both pre/post stages. -/
abbrev B2DirectionIndex := B2Slot × B2Stage

/-- The full B2 A2 envelope additionally includes both cross-validation directions. -/
abbrev B2EnvelopeIndex := B2Slot × B2Direction × B2Stage

/-- Seventeen generator slots and two pre/post states occur in each fixed direction. -/
def b2SlotsPerDirection : ℕ := 17 * 2

/-- The full fixed envelope also contains both AB and BA directions. -/
def b2EnvelopeCard : ℕ := b2SlotsPerDirection * 2

@[simp] theorem b2SlotsPerDirection_eq : b2SlotsPerDirection = 34 := by
  norm_num [b2SlotsPerDirection]

@[simp] theorem b2EnvelopeCard_eq : b2EnvelopeCard = 68 := by
  norm_num [b2EnvelopeCard, b2SlotsPerDirection]

@[simp] theorem b2Stage_card : Fintype.card B2Stage = 2 := by
  classical
  change (Finset.univ : Finset B2Stage).card = 2
  have hUniv : (Finset.univ : Finset B2Stage) = {.pre, .post} := by
    ext stage
    cases stage <;> simp
  rw [hUniv]
  simp

@[simp] theorem b2Direction_card : Fintype.card B2Direction = 2 := by
  classical
  change (Finset.univ : Finset B2Direction).card = 2
  have hUniv : (Finset.univ : Finset B2Direction) = {.ab, .ba} := by
    ext direction
    cases direction <;> simp
  rw [hUniv]
  simp

@[simp] theorem b2DirectionIndex_card : Fintype.card B2DirectionIndex = 34 := by
  norm_num [B2DirectionIndex, B2Slot, Fintype.card_prod]

@[simp] theorem b2EnvelopeIndex_card : Fintype.card B2EnvelopeIndex = 68 := by
  norm_num [B2EnvelopeIndex, B2Slot, Fintype.card_prod]

/-- Probability clipping used by the revised B2-A2 contract. -/
def clipProbability (ε probability : ℝ) : ℝ :=
  winsorize ε (1 - ε) probability

/-- Binary negative log loss, evaluated after probability clipping in B2. -/
noncomputable def bernoulliNegLogLoss (outcome : Bool) (probability : ℝ) : ℝ :=
  if outcome then -Real.log probability else -Real.log (1 - probability)

theorem clipProbability_mem_Icc
    (ε probability : ℝ) (hε : ε ≤ 1 - ε) :
    clipProbability ε probability ∈ Set.Icc ε (1 - ε) := by
  constructor
  · exact le_max_left _ _
  · exact max_le hε (min_le_left _ _)

/-- A Bernoulli log loss with probability in `[ε,1-ε]` lies in `[0,-log ε]`. -/
theorem bernoulliNegLogLoss_mem_Icc
    (ε probability : ℝ) (outcome : Bool)
    (hε : 0 < ε)
    (hProbability : probability ∈ Set.Icc ε (1 - ε)) :
    bernoulliNegLogLoss outcome probability ∈ Set.Icc 0 (-Real.log ε) := by
  have hProbabilityPos : 0 < probability := hε.trans_le hProbability.1
  have hProbabilityLeOne : probability ≤ 1 := by linarith [hProbability.2]
  have hComplementPos : 0 < 1 - probability := by linarith [hProbability.2]
  have hComplementLeOne : 1 - probability ≤ 1 := by linarith [hProbability.1]
  cases outcome with
  | false =>
      simp only [bernoulliNegLogLoss, Bool.false_eq_true, ↓reduceIte]
      constructor
      · exact neg_nonneg.mpr (Real.log_nonpos hComplementPos.le hComplementLeOne)
      · exact neg_le_neg (Real.log_le_log hε (by linarith [hProbability.2]))
  | true =>
      simp only [bernoulliNegLogLoss, ↓reduceIte]
      constructor
      · exact neg_nonneg.mpr (Real.log_nonpos hProbabilityPos.le hProbabilityLeOne)
      · exact neg_le_neg (Real.log_le_log hε hProbability.1)

/-- The revised clipped Y/X loss has the deterministic envelope `-log ε`. -/
theorem clippedBernoulliNegLogLoss_mem_Icc
    (ε probability : ℝ) (outcome : Bool)
    (hε : 0 < ε) (hεHalf : ε ≤ 1 / 2) :
    bernoulliNegLogLoss outcome (clipProbability ε probability) ∈
      Set.Icc 0 (-Real.log ε) := by
  apply bernoulliNegLogLoss_mem_Icc ε (clipProbability ε probability) outcome hε
  exact clipProbability_mem_Icc ε probability (by linarith)

/-- Explicit capping of the nonnegative per-center F squared-distance loss. -/
def cappedFLoss (envelope rawLoss : ℝ) : ℝ :=
  winsorize 0 envelope rawLoss

/-- The revised F loss lies in its registered finite envelope. -/
theorem cappedFLoss_mem_Icc
    (envelope rawLoss : ℝ) (hEnvelope : 0 ≤ envelope) :
    cappedFLoss envelope rawLoss ∈ Set.Icc 0 envelope := by
  constructor
  · exact le_max_left _ _
  · exact max_le hEnvelope (min_le_left _ _)

/-- The robust score scale after applying a registered deterministic positive floor. -/
def robustScaleWithFloor (scaleFloor madScale iqrScale : ℝ) : ℝ :=
  max scaleFloor (max madScale iqrScale)

theorem scaleFloor_le_robustScaleWithFloor
    (scaleFloor madScale iqrScale : ℝ) :
    scaleFloor ≤ robustScaleWithFloor scaleFloor madScale iqrScale := by
  exact le_max_left _ _

theorem robustScaleWithFloor_pos
    (scaleFloor madScale iqrScale : ℝ) (hScaleFloor : 0 < scaleFloor) :
    0 < robustScaleWithFloor scaleFloor madScale iqrScale :=
  hScaleFloor.trans_le (scaleFloor_le_robustScaleWithFloor scaleFloor madScale iqrScale)

/-- The existing normalization Lipschitz theorem instantiated with the algorithmic scale floor. -/
theorem abs_robustNormalizeWithFloor_sub_le
    (lower upper location scaleFloor madScale iqrScale x y : ℝ)
    (hScaleFloor : 0 < scaleFloor) :
    |robustNormalize lower upper location
        (robustScaleWithFloor scaleFloor madScale iqrScale) x -
      robustNormalize lower upper location
        (robustScaleWithFloor scaleFloor madScale iqrScale) y| ≤
      |x - y| / scaleFloor := by
  exact abs_robustNormalize_sub_robustNormalize_le
    lower upper location (robustScaleWithFloor scaleFloor madScale iqrScale)
      scaleFloor x y hScaleFloor
      (scaleFloor_le_robustScaleWithFloor scaleFloor madScale iqrScale)

/-- A finite weighted center average with weights fixed before evaluation. -/
def weightedCenterRisk
    {Center : Type uCenter} [Fintype Center]
    (weight loss : Center → ℝ) : ℝ :=
  ∑ center, weight center * loss center

/--
If replacing one evaluation center changes only that center's bounded summand, the whole frozen
weighted average changes by at most `weight changed * envelope`.
-/
theorem abs_weightedCenterRisk_sub_le
    {Center : Type uCenter} [Fintype Center] [DecidableEq Center]
    (weight loss replacementLoss : Center → ℝ) (changed : Center) (envelope : ℝ)
    (hWeight : ∀ center, 0 ≤ weight center)
    (hLoss : ∀ center, loss center ∈ Set.Icc 0 envelope)
    (hReplacementLoss : ∀ center, replacementLoss center ∈ Set.Icc 0 envelope)
    (hSame : ∀ center, center ≠ changed → loss center = replacementLoss center) :
    |weightedCenterRisk weight loss - weightedCenterRisk weight replacementLoss| ≤
      weight changed * envelope := by
  classical
  have hSum :
      (∑ center, (weight center * loss center - weight center * replacementLoss center)) =
        weight changed * loss changed - weight changed * replacementLoss changed := by
    apply Finset.sum_eq_single changed
    · intro center _ hCenter
      rw [hSame center hCenter]
      ring
    · simp
  have hDifference : |loss changed - replacementLoss changed| ≤ envelope := by
    rw [abs_sub_le_iff]
    constructor
    · linarith [hLoss changed |>.2, hReplacementLoss changed |>.1]
    · linarith [hLoss changed |>.1, hReplacementLoss changed |>.2]
  rw [weightedCenterRisk, weightedCenterRisk, ← Finset.sum_sub_distrib, hSum]
  calc
    |weight changed * loss changed - weight changed * replacementLoss changed|
        = weight changed * |loss changed - replacementLoss changed| := by
          rw [← mul_sub, abs_mul, abs_of_nonneg (hWeight changed)]
    _ ≤ weight changed * envelope :=
      mul_le_mul_of_nonneg_left hDifference (hWeight changed)

/-- Equal-center stability loss for two predictors frozen outside the evaluation center blocks. -/
noncomputable def equalCenterStabilityRisk
    {numberOfCenters : ℕ} (loss : Fin numberOfCenters → ℝ) : ℝ :=
  ∑ center, (1 / (numberOfCenters : ℝ)) * loss center

/-- The revised evaluation-only stability risk has replacement constant exactly `1 / E`. -/
theorem abs_equalCenterStabilityRisk_sub_le
    {numberOfCenters : ℕ}
    (loss replacementLoss : Fin numberOfCenters → ℝ) (changed : Fin numberOfCenters)
    (hLoss : ∀ center, loss center ∈ Set.Icc (0 : ℝ) 1)
    (hReplacementLoss : ∀ center, replacementLoss center ∈ Set.Icc (0 : ℝ) 1)
    (hSame : ∀ center, center ≠ changed → loss center = replacementLoss center) :
    |equalCenterStabilityRisk loss - equalCenterStabilityRisk replacementLoss| ≤
      1 / (numberOfCenters : ℝ) := by
  simpa [equalCenterStabilityRisk, weightedCenterRisk] using
    (abs_weightedCenterRisk_sub_le
      (fun _ : Fin numberOfCenters ↦ 1 / (numberOfCenters : ℝ))
      loss replacementLoss changed 1
      (fun _ ↦ one_div_nonneg.mpr (Nat.cast_nonneg numberOfCenters))
      hLoss hReplacementLoss hSame)

/-- One revised B2 record's empirical risk in a frozen affine component-risk coordinate system. -/
noncomputable def b2FrozenAffineEmpiricalRisk
    {numberOfCenters : ℕ} {Sample : Type uSample}
    (weight : Fin numberOfCenters → ℝ) (loss : Fin numberOfCenters → Sample → ℝ)
    (location scale : ℝ) (sample : Sample) : ℝ :=
  BiostatFormal.Probability.frozenAffineRisk location scale
    (BiostatFormal.Probability.weightedCenterRisk Finset.univ weight loss sample)

/-- The matching population risk: center expectations first, then the identical frozen transform. -/
noncomputable def b2FrozenAffinePopulationRisk
    {numberOfCenters : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (weight : Fin numberOfCenters → ℝ)
    (loss : Fin numberOfCenters → Sample → ℝ) (location scale : ℝ) : ℝ :=
  BiostatFormal.Probability.frozenAffineRisk location scale
    (BiostatFormal.Probability.weightedCenterPopulationRisk
      μ Finset.univ weight loss)

/-- Exact pointwise Hoeffding proxy `B²/4 * sum_e weight_e²` for one revised B2 record. -/
noncomputable def b2PointwiseHoeffdingProxy
    {numberOfCenters : ℕ} (weight : Fin numberOfCenters → ℝ) (lossEnvelope : ℝ) : NNReal :=
  BiostatFormal.Probability.weightedHoeffdingProxy Finset.univ weight lossEnvelope

theorem b2PointwiseHoeffdingProxy_coe
    {numberOfCenters : ℕ} (weight : Fin numberOfCenters → ℝ) (lossEnvelope : ℝ)
    (hEnvelope : 0 ≤ lossEnvelope) :
    (b2PointwiseHoeffdingProxy weight lossEnvelope : ℝ) =
      lossEnvelope ^ 2 / 4 * ∑ center, weight center ^ 2 := by
  exact BiostatFormal.Probability.weightedHoeffdingProxy_coe
    Finset.univ weight lossEnvelope hEnvelope

/--
The missing B2 candidate-specific tail, now derived rather than assumed.  Conditional on one frozen
training/normalization context, independent center-block losses in `[0,B]` satisfy the exact
two-sided weighted Hoeffding bound after positive affine scaling.
-/
theorem measure_b2FrozenAffinePointwiseDeviation_le
    {numberOfCenters : ℕ} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (weight : Fin numberOfCenters → ℝ) (loss : Fin numberOfCenters → Sample → ℝ)
    (lossEnvelope location scale threshold : ℝ)
    (hIndep : ProbabilityTheory.iIndepFun loss μ)
    (hMeasurable : ∀ center, AEMeasurable (loss center) μ)
    (hEnvelope : ∀ center, ∀ᵐ sample ∂μ,
      loss center sample ∈ Set.Icc 0 lossEnvelope)
    (hScale : 0 < scale) (hThreshold : 0 ≤ threshold) :
    μ {sample |
      threshold < |b2FrozenAffineEmpiricalRisk weight loss location scale sample -
        b2FrozenAffinePopulationRisk μ weight loss location scale|} ≤
      ENNReal.ofReal
        (2 * Real.exp
          (-(scale * threshold) ^ 2 /
            (2 * (b2PointwiseHoeffdingProxy weight lossEnvelope : ℝ)))) := by
  simpa [b2FrozenAffineEmpiricalRisk, b2FrozenAffinePopulationRisk,
    b2PointwiseHoeffdingProxy] using
    (BiostatFormal.Probability.measure_frozenAffineWeightedCenterRisk_sub_population_gt_le
      μ Finset.univ weight loss lossEnvelope location scale hIndep
      (fun center _ ↦ hMeasurable center) (fun center _ ↦ hEnvelope center)
      hScale hThreshold)

/--
Candidate-specific weighted-Hoeffding tails feed the existing finite-envelope union theorem
without an additional concentration assumption.  The pool is still fixed before evaluation.
-/
theorem measure_b2DirectionFrozenAffineUniformDeviation_le_sum
    {Candidate : Type uCandidate} {numberOfCenters : ℕ}
    {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (pool : Finset Candidate)
    (weight : Candidate → Fin numberOfCenters → ℝ)
    (loss : Candidate → Fin numberOfCenters → Sample → ℝ)
    (lossEnvelope location scale threshold : Candidate → ℝ)
    (hIndep : ∀ candidate ∈ pool, ProbabilityTheory.iIndepFun (loss candidate) μ)
    (hMeasurable : ∀ candidate ∈ pool, ∀ center,
      AEMeasurable (loss candidate center) μ)
    (hEnvelope : ∀ candidate ∈ pool, ∀ center, ∀ᵐ sample ∂μ,
      loss candidate center sample ∈ Set.Icc 0 (lossEnvelope candidate))
    (hScale : ∀ candidate ∈ pool, 0 < scale candidate)
    (hThreshold : ∀ candidate ∈ pool, 0 ≤ threshold candidate) :
    μ (uniformDeviationFailureEvent pool
        (fun candidate ↦ b2FrozenAffineEmpiricalRisk
          (weight candidate) (loss candidate) (location candidate) (scale candidate))
        (fun candidate ↦ b2FrozenAffinePopulationRisk
          μ (weight candidate) (loss candidate) (location candidate) (scale candidate))
        threshold) ≤
      ∑ candidate ∈ pool,
        ENNReal.ofReal
          (2 * Real.exp
            (-(scale candidate * threshold candidate) ^ 2 /
              (2 * (b2PointwiseHoeffdingProxy
                (weight candidate) (lossEnvelope candidate) : ℝ)))) := by
  apply measure_uniformDeviationFailureEvent_le_sum
  intro candidate hCandidate
  exact measure_b2FrozenAffinePointwiseDeviation_le
    μ (weight candidate) (loss candidate) (lossEnvelope candidate)
      (location candidate) (scale candidate) (threshold candidate)
      (hIndep candidate hCandidate)
      (hMeasurable candidate hCandidate)
      (hEnvelope candidate hCandidate)
      (hScale candidate hCandidate)
      (hThreshold candidate hCandidate)

/--
Tower-law upper bound.  A pointwise conditional failure budget under an a.e.-measurable family of
conditional laws remains the same after integrating over a probability law for the frozen context.
-/
theorem measure_bind_failure_le
    {Context : Type uContextAB} {Omega : Type uOmega}
    [MeasurableSpace Context] [MeasurableSpace Omega]
    (contextLaw : Measure Context) [IsProbabilityMeasure contextLaw]
    (conditionalLaw : Context → Measure Omega)
    (failure : Set Omega) (budget : ENNReal)
    (hFailure : MeasurableSet failure)
    (hConditionalLaw : AEMeasurable conditionalLaw contextLaw)
    (hConditional : ∀ context, conditionalLaw context failure ≤ budget) :
    contextLaw.bind conditionalLaw failure ≤ budget := by
  rw [Measure.bind_apply hFailure hConditionalLaw]
  calc
    (∫⁻ context, conditionalLaw context failure ∂contextLaw)
        ≤ ∫⁻ _context, budget ∂contextLaw :=
      lintegral_mono fun context ↦ hConditional context
    _ = budget := by simp

/--
AB and BA may use different conditioning contexts, but both must factor the same joint data law.
Each direction is deconditioned first; only then is the direction union charged.
-/
theorem measure_twoDirectionFailure_le
    {ContextAB : Type uContextAB} {ContextBA : Type uContextBA} {Omega : Type uOmega}
    [MeasurableSpace ContextAB] [MeasurableSpace ContextBA] [MeasurableSpace Omega]
    (jointLaw : Measure Omega)
    (contextLawAB : Measure ContextAB) [IsProbabilityMeasure contextLawAB]
    (contextLawBA : Measure ContextBA) [IsProbabilityMeasure contextLawBA]
    (conditionalLawAB : ContextAB → Measure Omega)
    (conditionalLawBA : ContextBA → Measure Omega)
    (failureAB failureBA : Set Omega) (budgetAB budgetBA : ENNReal)
    (hFailureAB : MeasurableSet failureAB) (hFailureBA : MeasurableSet failureBA)
    (hConditionalLawAB : AEMeasurable conditionalLawAB contextLawAB)
    (hConditionalLawBA : AEMeasurable conditionalLawBA contextLawBA)
    (hJointAB : jointLaw = contextLawAB.bind conditionalLawAB)
    (hJointBA : jointLaw = contextLawBA.bind conditionalLawBA)
    (hConditionalAB : ∀ context, conditionalLawAB context failureAB ≤ budgetAB)
    (hConditionalBA : ∀ context, conditionalLawBA context failureBA ≤ budgetBA) :
    jointLaw (failureAB ∪ failureBA) ≤ budgetAB + budgetBA := by
  have hAB : jointLaw failureAB ≤ budgetAB := by
    rw [hJointAB]
    exact measure_bind_failure_le contextLawAB conditionalLawAB failureAB budgetAB
      hFailureAB hConditionalLawAB hConditionalAB
  have hBA : jointLaw failureBA ≤ budgetBA := by
    rw [hJointBA]
    exact measure_bind_failure_le contextLawBA conditionalLawBA failureBA budgetBA
      hFailureBA hConditionalLawBA hConditionalBA
  exact (measure_union_le failureAB failureBA).trans (add_le_add hAB hBA)

/-- Union of the four component failures in both B2 directions. -/
def b2DeviationFailureEvent
    {Omega : Type uOmega}
    (failure : RiskComponent → B2Direction → Set Omega) : Set Omega :=
  ⋃ component, ⋃ direction, failure component direction

/-- Candidate/direction-specific budgets; component budgets are obtained by the inner sum. -/
noncomputable def b2TotalDeviationBudget
    (budget : RiskComponent → B2Direction → ENNReal) : ENNReal :=
  ∑ component, ∑ direction, budget component direction

/-- The final A2 failure probability is the sum of the eight component-direction budgets. -/
theorem measure_b2DeviationFailureEvent_le
    {Omega : Type uOmega} [MeasurableSpace Omega]
    (μ : Measure Omega)
    (failure : RiskComponent → B2Direction → Set Omega)
    (budget : RiskComponent → B2Direction → ENNReal)
    (hFailure : ∀ component direction,
      μ (failure component direction) ≤ budget component direction) :
    μ (b2DeviationFailureEvent failure) ≤ b2TotalDeviationBudget budget := by
  unfold b2DeviationFailureEvent b2TotalDeviationBudget
  calc
    μ (⋃ component, ⋃ direction, failure component direction)
        ≤ ∑ component, μ (⋃ direction, failure component direction) :=
      measure_iUnion_fintype_le μ _
    _ ≤ ∑ component, ∑ direction, budget component direction := by
      apply Finset.sum_le_sum
      intro component _
      exact (measure_iUnion_fintype_le μ _).trans
        (Finset.sum_le_sum fun direction _ ↦ hFailure component direction)

/-- A common pointwise tail over a 34-slot direction envelope costs exactly `34 * tail`. -/
theorem measure_b2DirectionUniformFailure_le
    {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (pool : Finset Candidate)
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ)
    (commonTail : ENNReal)
    (hCard : pool.card = b2SlotsPerDirection)
    (hTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalRisk populationRisk threshold candidate) ≤ commonTail) :
    μ (uniformDeviationFailureEvent pool empiricalRisk populationRisk threshold) ≤
      34 * commonTail := by
  simpa [hCard] using
    (measure_uniformDeviationFailureEvent_le_card_mul
      μ pool empiricalRisk populationRisk threshold commonTail hTail)

/-- Weighted-Hoeffding calibration for 34 records: `2 exp(-2u²/C) = δ/34`. -/
noncomputable def b2WeightedHoeffdingThreshold
    (sensitivitySquare directionBudget : ℝ) : ℝ :=
  Real.sqrt (sensitivitySquare / 2 * Real.log (68 / directionBudget))

/-- Backward-compatible name from the earlier bounded-difference interface. -/
noncomputable def b2McDiarmidThreshold
    (sensitivitySquare directionBudget : ℝ) : ℝ :=
  b2WeightedHoeffdingThreshold sensitivitySquare directionBudget

/-- Y/X/F calibration after a frozen normalization with a positive scale floor. -/
noncomputable def b2NormalizedComponentThreshold
    (lossEnvelope scaleFloor centerWeightSquare directionBudget : ℝ) : ℝ :=
  (lossEnvelope / scaleFloor) *
    Real.sqrt (centerWeightSquare / 2 * Real.log (68 / directionBudget))

/-- Revised equal-center stability calibration, with sensitivity-square sum `1/E`. -/
noncomputable def b2StabilityThreshold
    (numberOfCenters : ℕ) (directionBudget : ℝ) : ℝ :=
  Real.sqrt ((1 / (numberOfCenters : ℝ)) / 2 * Real.log (68 / directionBudget))

/-- Y/X center-block weight induced by an individual-weighted validation mean. -/
noncomputable def b2ObservationCenterWeight
    {numberOfCenters : ℕ} (validationCount : Fin numberOfCenters → ℕ)
    (center : Fin numberOfCenters) : ℝ :=
  validationCount center / ∑ otherCenter, validationCount otherCenter

/-- Squared-weight sum used in the Y/X bounded-difference constant. -/
noncomputable def b2ObservationCenterWeightSquare
    {numberOfCenters : ℕ} (validationCount : Fin numberOfCenters → ℕ) : ℝ :=
  ∑ center, (b2ObservationCenterWeight validationCount center) ^ 2

/-- F and revised stability use equal center weights, whose squared-weight sum is `1/E`. -/
noncomputable def b2EqualCenterWeightSquare (numberOfCenters : ℕ) : ℝ :=
  1 / (numberOfCenters : ℝ)

/-- The registered P0 component weights. -/
noncomputable def b2P0Weight : RiskComponent → ℝ
  | .outcome => 1 / 3
  | .treatment => 1 / 3
  | .covariate => 1 / 3
  | .stability => 1

/-- Explicit B2-A2 composite threshold under the P0 weights. -/
noncomputable def b2P0CompositeThreshold
    (outcome treatment covariate stability : ℝ) : ℝ :=
  (outcome + treatment + covariate) / 3 + stability

/-- A component threshold valid for both directions. -/
def b2UniformDirectionThreshold (threshold : B2Direction → ℝ) : ℝ :=
  max (threshold .ab) (threshold .ba)

/--
The explicit B2 `u_A2`: take the larger AB/BA threshold componentwise, then apply the P0 weights.
-/
noncomputable def b2A2Threshold
    (threshold : RiskComponent → B2Direction → ℝ) : ℝ :=
  b2P0CompositeThreshold
    (b2UniformDirectionThreshold (threshold .outcome))
    (b2UniformDirectionThreshold (threshold .treatment))
    (b2UniformDirectionThreshold (threshold .covariate))
    (b2UniformDirectionThreshold (threshold .stability))

/-- The general composite-risk theorem specializes to the displayed B2 P0 threshold. -/
theorem abs_b2P0CompositeRisk_sub_le
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (risk : CompositeRiskPair Candidate) (candidate : Candidate)
    (componentThreshold : RiskComponent → ℝ)
    (hWeight : risk.weight = b2P0Weight)
    (hDeviation : ∀ component,
      |risk.empiricalComponent component candidate -
        risk.populationComponent component candidate| ≤ componentThreshold component) :
    |empiricalCompositeRisk risk candidate - populationCompositeRisk risk candidate| ≤
      b2P0CompositeThreshold
        (componentThreshold .outcome) (componentThreshold .treatment)
        (componentThreshold .covariate) (componentThreshold .stability) := by
  have hNonnegative : ∀ component, 0 ≤ risk.weight component := by
    intro component
    rw [hWeight]
    cases component <;> norm_num [b2P0Weight]
  have hBound := abs_empiricalCompositeRisk_sub_populationCompositeRisk_le
    risk candidate componentThreshold hNonnegative hDeviation
  rw [hWeight] at hBound
  calc
    |empiricalCompositeRisk risk candidate - populationCompositeRisk risk candidate| ≤
        ∑ component : RiskComponent,
          b2P0Weight component * componentThreshold component := hBound
    _ = b2P0CompositeThreshold
          (componentThreshold .outcome) (componentThreshold .treatment)
          (componentThreshold .covariate) (componentThreshold .stability) := by
      classical
      have hUniv : (Finset.univ : Finset RiskComponent) =
          {.outcome, .treatment, .covariate, .stability} := by
        ext component
        cases component <;> simp
      rw [hUniv]
      simp [Finset.sum_insert, b2P0CompositeThreshold, b2P0Weight]
      ring

end BiostatFormal.HighDimensional.Multiview
