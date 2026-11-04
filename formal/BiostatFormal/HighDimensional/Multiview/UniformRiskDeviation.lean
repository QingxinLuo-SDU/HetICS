import BiostatFormal.HighDimensional.Multiview.Risks
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Group.MinMax
import Mathlib.MeasureTheory.Measure.Real

/-!
# Uniform risk deviation over a finite candidate envelope

This file proves the finite-union and deterministic composition layers needed by Appendix A2.
It deliberately does not assert a McDiarmid bound: each pointwise probability bound remains an
explicit input until the corresponding bounded-loss, independence, and replacement-sensitivity
theorem has been proved for the actual algorithm.

The finite set is an evaluation-sample-independent envelope (or a finite set frozen after
conditioning on training data).  A pool selected from the evaluation sample is covered only via an
explicit pointwise subset hypothesis.
-/

namespace BiostatFormal.HighDimensional.Multiview

open MeasureTheory

universe uCandidate uSample uTraining

/-- Failure of the deviation bound for one fixed candidate index. -/
def deviationFailureEvent
    {Candidate : Type uCandidate} {Sample : Type uSample}
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ)
    (candidate : Candidate) : Set Sample :=
  {sample | threshold candidate <
    |empiricalRisk candidate sample - populationRisk candidate|}

/-- Failure of a deviation bound somewhere in a fixed finite candidate envelope. -/
def uniformDeviationFailureEvent
    {Candidate : Type uCandidate} {Sample : Type uSample}
    (pool : Finset Candidate)
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ) : Set Sample :=
  ⋃ candidate ∈ pool,
    deviationFailureEvent empiricalRisk populationRisk threshold candidate

/--
Finite-pool union bound with candidate-specific tail budgets.  No independence between candidate
risks is required at this layer; independence is needed upstream to prove each pointwise tail.
-/
theorem measure_uniformDeviationFailureEvent_le_sum
    {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (pool : Finset Candidate)
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ)
    (tailBudget : Candidate → ENNReal)
    (hTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalRisk populationRisk threshold candidate) ≤
        tailBudget candidate) :
    μ (uniformDeviationFailureEvent pool empiricalRisk populationRisk threshold) ≤
      ∑ candidate ∈ pool, tailBudget candidate := by
  calc
    μ (uniformDeviationFailureEvent pool empiricalRisk populationRisk threshold)
        ≤ ∑ candidate ∈ pool,
            μ (deviationFailureEvent empiricalRisk populationRisk threshold candidate) := by
          exact measure_biUnion_finset_le pool _
    _ ≤ ∑ candidate ∈ pool, tailBudget candidate := by
      exact Finset.sum_le_sum fun candidate hCandidate ↦ hTail candidate hCandidate

/-- Common-tail specialization: the union-bound cardinality is exactly the envelope cardinality. -/
theorem measure_uniformDeviationFailureEvent_le_card_mul
    {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (pool : Finset Candidate)
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ)
    (commonTail : ENNReal)
    (hTail : ∀ candidate ∈ pool,
      μ (deviationFailureEvent empiricalRisk populationRisk threshold candidate) ≤ commonTail) :
    μ (uniformDeviationFailureEvent pool empiricalRisk populationRisk threshold) ≤
      pool.card * commonTail := by
  calc
    μ (uniformDeviationFailureEvent pool empiricalRisk populationRisk threshold)
        ≤ ∑ candidate ∈ pool, commonTail :=
          measure_uniformDeviationFailureEvent_le_sum μ pool empiricalRisk populationRisk threshold
            (fun _ ↦ commonTail) hTail
    _ = pool.card * commonTail := by simp

/--
Conditional form of the finite-pool theorem.  For each frozen training realization, `conditionalLaw
training` is the validation law, `pool training` is fixed, and `threshold training` therefore does
not vary with the validation sample.
-/
theorem conditionalMeasure_uniformDeviationFailureEvent_le_sum
    {Training : Type uTraining} {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (conditionalLaw : Training → Measure Sample)
    (pool : Training → Finset Candidate)
    (empiricalRisk : Training → Candidate → Sample → ℝ)
    (populationRisk threshold : Training → Candidate → ℝ)
    (tailBudget : Training → Candidate → ENNReal)
    (hTail : ∀ training candidate, candidate ∈ pool training →
      conditionalLaw training
          (deviationFailureEvent (empiricalRisk training) (populationRisk training)
            (threshold training) candidate) ≤ tailBudget training candidate)
    (training : Training) :
    conditionalLaw training
        (uniformDeviationFailureEvent (pool training) (empiricalRisk training)
          (populationRisk training) (threshold training)) ≤
      ∑ candidate ∈ pool training, tailBudget training candidate := by
  exact measure_uniformDeviationFailureEvent_le_sum
    (conditionalLaw training) (pool training) (empiricalRisk training)
    (populationRisk training) (threshold training) (tailBudget training)
    (fun candidate hCandidate ↦ hTail training candidate hCandidate)

/--
Failure among a sample-dependent pool.  The pool may use the same validation sample, but its
indices must be a subset of a fixed envelope whose pointwise tails have already been controlled.
-/
def dataDependentPoolFailureEvent
    {Candidate : Type uCandidate} {Sample : Type uSample}
    (pool : Sample → Finset Candidate)
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ) : Set Sample :=
  {sample | ∃ candidate,
    candidate ∈ pool sample ∧
      sample ∈ deviationFailureEvent empiricalRisk populationRisk threshold candidate}

/--
A validation-dependent guard or eligibility filter incurs no additional multiplicity cost when it
selects only from the fixed envelope.  This theorem does not cover candidates newly generated from
the validation sample outside that envelope.
-/
theorem measure_dataDependentPoolFailureEvent_le_sum
    {Candidate : Type uCandidate} {Sample : Type uSample}
    [MeasurableSpace Sample]
    (μ : Measure Sample)
    (envelope : Finset Candidate)
    (pool : Sample → Finset Candidate)
    (empiricalRisk : Candidate → Sample → ℝ)
    (populationRisk threshold : Candidate → ℝ)
    (tailBudget : Candidate → ENNReal)
    (hSubset : ∀ sample, pool sample ⊆ envelope)
    (hTail : ∀ candidate ∈ envelope,
      μ (deviationFailureEvent empiricalRisk populationRisk threshold candidate) ≤
        tailBudget candidate) :
    μ (dataDependentPoolFailureEvent pool empiricalRisk populationRisk threshold) ≤
      ∑ candidate ∈ envelope, tailBudget candidate := by
  have hEventSubset :
      dataDependentPoolFailureEvent pool empiricalRisk populationRisk threshold ⊆
        uniformDeviationFailureEvent envelope empiricalRisk populationRisk threshold := by
    intro sample hSample
    rcases hSample with ⟨candidate, hCandidate, hFailure⟩
    apply Set.mem_iUnion.2
    refine ⟨candidate, ?_⟩
    apply Set.mem_iUnion.2
    exact ⟨hSubset sample hCandidate, hFailure⟩
  exact (measure_mono hEventSubset).trans
    (measure_uniformDeviationFailureEvent_le_sum μ envelope empiricalRisk populationRisk threshold
      tailBudget hTail)

/-- Clipping to a closed real interval, used by the robust normalization map in Appendix A2. -/
def winsorize (lower upper x : ℝ) : ℝ :=
  max lower (min upper x)

/-- A location-scale normalization followed by clipping. -/
noncomputable def robustNormalize (lower upper location scale x : ℝ) : ℝ :=
  winsorize lower upper ((x - location) / scale)

private theorem abs_winsorize_sub_winsorize_le
    (lower upper x y : ℝ) :
    |winsorize lower upper x - winsorize lower upper y| ≤ |x - y| := by
  unfold winsorize
  calc
    |max lower (min upper x) - max lower (min upper y)|
        = |max (min upper x) lower - max (min upper y) lower| := by
            rw [max_comm lower (min upper x), max_comm lower (min upper y)]
    _ ≤ |min upper x - min upper y| := abs_max_sub_max_le_abs _ _ _
    _ = |min x upper - min y upper| := by
      rw [min_comm upper x, min_comm upper y]
    _ ≤ |x - y| := by
      simpa using abs_min_sub_min_le_max x upper y upper

/--
The robust scale lower bound is explicit: clipping cannot increase distances, and division by a
scale `s ≥ ε > 0` has Lipschitz constant at most `1 / ε`.
-/
theorem abs_robustNormalize_sub_robustNormalize_le
    (lower upper location scale ε x y : ℝ)
    (hε : 0 < ε) (hScale : ε ≤ scale) :
    |robustNormalize lower upper location scale x -
        robustNormalize lower upper location scale y| ≤ |x - y| / ε := by
  have hScalePos : 0 < scale := hε.trans_le hScale
  calc
    |robustNormalize lower upper location scale x -
        robustNormalize lower upper location scale y|
        ≤ |(x - location) / scale - (y - location) / scale| :=
          abs_winsorize_sub_winsorize_le _ _ _ _
    _ = |x - y| / scale := by
      rw [← sub_div, abs_div, abs_of_pos hScalePos]
      congr 2
      ring
    _ ≤ |x - y| / ε := by
      exact (div_le_div_iff₀ hScalePos hε).2
        (mul_le_mul_of_nonneg_left hScale (abs_nonneg (x - y)))

/-- A raw deviation bound propagates through robust normalization with factor `1 / ε`. -/
theorem robustNormalize_deviation_le
    (lower upper location scale ε x y t : ℝ)
    (hε : 0 < ε) (hScale : ε ≤ scale) (hDeviation : |x - y| ≤ t) :
    |robustNormalize lower upper location scale x -
        robustNormalize lower upper location scale y| ≤ t / ε := by
  exact (abs_robustNormalize_sub_robustNormalize_le
    lower upper location scale ε x y hε hScale).trans
      (div_le_div_of_nonneg_right hDeviation hε.le)

/--
Componentwise deviations combine with the same fixed nonnegative weights used by population and
empirical composite risk.  If negative weights were allowed, the right side would require absolute
weights instead.
-/
theorem abs_empiricalCompositeRisk_sub_populationCompositeRisk_le
    {Candidate : Type uCandidate} [RiskInput Candidate]
    (risk : CompositeRiskPair Candidate)
    (candidate : Candidate)
    (componentThreshold : RiskComponent → ℝ)
    (hWeight : ∀ component, 0 ≤ risk.weight component)
    (hDeviation : ∀ component,
      |risk.empiricalComponent component candidate -
        risk.populationComponent component candidate| ≤ componentThreshold component) :
    |empiricalCompositeRisk risk candidate - populationCompositeRisk risk candidate| ≤
      ∑ component : RiskComponent, risk.weight component * componentThreshold component := by
  rw [empiricalCompositeRisk, populationCompositeRisk, ← Finset.sum_sub_distrib]
  calc
    |∑ component : RiskComponent,
        (risk.weight component * risk.empiricalComponent component candidate -
          risk.weight component * risk.populationComponent component candidate)|
        = |∑ component : RiskComponent,
            risk.weight component *
              (risk.empiricalComponent component candidate -
                risk.populationComponent component candidate)| := by
            congr 1
            apply Finset.sum_congr rfl
            intro component _
            ring
    _ ≤ ∑ component : RiskComponent,
        |risk.weight component *
          (risk.empiricalComponent component candidate -
            risk.populationComponent component candidate)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ component : RiskComponent,
        risk.weight component *
          |risk.empiricalComponent component candidate -
            risk.populationComponent component candidate| := by
      apply Finset.sum_congr rfl
      intro component _
      rw [abs_mul, abs_of_nonneg (hWeight component)]
    _ ≤ ∑ component : RiskComponent,
        risk.weight component * componentThreshold component := by
      exact Finset.sum_le_sum fun component _ ↦
        mul_le_mul_of_nonneg_left (hDeviation component) (hWeight component)

end BiostatFormal.HighDimensional.Multiview
