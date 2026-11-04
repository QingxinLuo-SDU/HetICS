import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# Selection and aggregation for regression adjustment

The invariance threshold is a parameter, not a fixed numerical constant. Retention of an
identifying core and convergence of reference estimators are statistical premises. The finite-family
transfer below does not require learning and estimation samples to be independent. Its good event
can include label-aligned recovery, admissibility and numerical availability.
-/

open Filter MeasureTheory
open scoped BigOperators Topology

namespace HetICS.RA

set_option autoImplicit false
noncomputable section

section Selection

variable {Feature : Type*}

/-- Both exposure-side and outcome-side candidates enter the next stage. -/
def dualCandidates [DecidableEq Feature] (exposure outcome : Finset Feature) : Finset Feature :=
  exposure ∪ outcome

/-- The strict continuous rule, with arbitrary threshold and adjusted p-values. -/
def invarianceKept (candidates : Finset Feature) (p : Feature → ℝ) (alpha : ℝ) :
    Finset Feature := candidates.filter fun j => alpha ≤ p j

theorem core_subset_dualCandidates [DecidableEq Feature] (core exposure outcome : Finset Feature)
    (hCoverage : ∀ j ∈ core, j ∈ exposure ∨ j ∈ outcome) :
    core ⊆ dualCandidates exposure outcome := by
  intro j hj
  exact Finset.mem_union.mpr (hCoverage j hj)

/-- Candidate coverage and actual nonrejection of the core imply final retention. -/
theorem core_subset_invarianceKept (core candidates : Finset Feature)
    (p : Feature → ℝ) (alpha : ℝ) (hCoverage : core ⊆ candidates)
    (hRetention : ∀ j ∈ core, alpha ≤ p j) :
    core ⊆ invarianceKept candidates p alpha := by
  intro j hj
  exact Finset.mem_filter.mpr ⟨hCoverage hj, hRetention j hj⟩

/-- Decreasing the threshold can only enlarge the retained set. -/
theorem invarianceKept_antitone (candidates : Finset Feature) (p : Feature → ℝ)
    {alpha beta : ℝ} (hab : alpha ≤ beta) :
    invarianceKept candidates p beta ⊆ invarianceKept candidates p alpha := by
  intro j hj
  rcases Finset.mem_filter.mp hj with ⟨hc, hp⟩
  exact Finset.mem_filter.mpr ⟨hc, hab.trans hp⟩

end Selection

section FiniteReferences

variable {K Ω : Type*} [Fintype K] [MeasurableSpace Ω]

/-- A randomly chosen member of a finite admissible family is bounded by the sum of its errors. -/
theorem selected_error_le_sum (reference : K → ℝ) (actual theta : ℝ)
    (hMember : ∃ k, actual = reference k) :
    |actual - theta| ≤ ∑ k, |reference k - theta| := by
  classical
  obtain ⟨k, hk⟩ := hMember
  rw [hk]
  exact Finset.single_le_sum (fun j _ => abs_nonneg (reference j - theta)) (Finset.mem_univ k)

/-- A finite-family union bound transfers consistency through same-data learning. -/
theorem learned_consistent_of_finite_references (μ : Measure Ω)
    (reference : K → ℕ → Ω → ℝ) (actual : ℕ → Ω → ℝ)
    (theta : ℝ) (good : ℕ → Set Ω)
    (hMember : ∀ n ω, ω ∈ good n → ∃ k, actual n ω = reference k n ω)
    (hReference : ∀ k, TendstoInMeasure μ (reference k) atTop (fun _ => theta))
    (hGood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ actual atTop (fun _ => theta) := by
  classical
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  have hRef : Tendsto (fun n => ∑ k, μ {ω | ε ≤ ‖reference k n ω - theta‖})
      atTop (𝓝 0) := by
    simpa using tendsto_finsetSum Finset.univ
      (fun k _ => (tendstoInMeasure_iff_norm.mp (hReference k)) ε hε)
  have hTotal := hGood.add hRef
  simp only [add_zero] at hTotal
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hTotal
  · intro n
    exact bot_le
  · intro n
    calc
      μ {ω | ε ≤ ‖actual n ω - theta‖}
          ≤ μ ((good n)ᶜ ∪ ⋃ k, {ω | ε ≤ ‖reference k n ω - theta‖}) := by
            apply measure_mono
            intro ω hω
            by_cases hg : ω ∈ good n
            · obtain ⟨k, hk⟩ := hMember n ω hg
              exact Or.inr (Set.mem_iUnion.mpr ⟨k, by simpa [← hk] using hω⟩)
            · exact Or.inl hg
      _ ≤ μ (good n)ᶜ + μ (⋃ k, {ω | ε ≤ ‖reference k n ω - theta‖}) :=
        measure_union_le _ _
      _ ≤ μ (good n)ᶜ + ∑ k, μ {ω | ε ≤ ‖reference k n ω - theta‖} :=
        add_le_add_right (measure_iUnion_fintype_le μ _) _

end FiniteReferences

section Aggregation

variable {K : Type*} [Fintype K]

/-- Normalize precision weights; a nonzero total precision is the availability condition. -/
def normalizedWeights (precision : K → ℝ) : K → ℝ :=
  fun k => precision k / ∑ j, precision j

theorem normalizedWeights_nonnegative (precision : K → ℝ)
    (hNonnegative : ∀ k, 0 ≤ precision k) :
    ∀ k, 0 ≤ normalizedWeights precision k := by
  intro k
  exact div_nonneg (hNonnegative k) (Finset.sum_nonneg fun j _ => hNonnegative j)

theorem sum_normalizedWeights (precision : K → ℝ)
    (hAvailable : (∑ k, precision k) ≠ 0) :
    ∑ k, normalizedWeights precision k = 1 := by
  simp only [normalizedWeights, ← Finset.sum_div, div_self hAvailable]

/-- Convex aggregation of a common-effect family; weights may depend on the same observations. -/
theorem convex_error_le_sum (weight estimate : K → ℝ) (theta : ℝ)
    (hNonnegative : ∀ k, 0 ≤ weight k) (hSum : ∑ k, weight k = 1) :
    |(∑ k, weight k * estimate k) - theta| ≤ ∑ k, |estimate k - theta| := by
  classical
  have hBound (k : K) : weight k ≤ 1 := by
    calc
      weight k ≤ ∑ j, weight j :=
        Finset.single_le_sum (fun j _ => hNonnegative j) (Finset.mem_univ k)
      _ = 1 := hSum
  have hIdentity : (∑ k, weight k * estimate k) - theta =
      ∑ k, weight k * (estimate k - theta) := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hSum, one_mul]
  rw [hIdentity]
  calc
    |∑ k, weight k * (estimate k - theta)|
        ≤ ∑ k, |weight k * (estimate k - theta)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ k, weight k * |estimate k - theta| := by
      apply Finset.sum_congr rfl
      intro k _
      rw [abs_mul, abs_of_nonneg (hNonnegative k)]
    _ ≤ ∑ k, |estimate k - theta| := by
      apply Finset.sum_le_sum
      intro k _
      exact mul_le_of_le_one_left (abs_nonneg _) (hBound k)

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Integrability of the aggregate error follows from the finite reference-error envelope. -/
theorem convex_aggregate_error_integrable (μ : Measure Ω)
    (weight estimate : K → Ω → ℝ) (theta : ℝ)
    (hNonnegative : ∀ ω k, 0 ≤ weight k ω)
    (hSum : ∀ ω, ∑ k, weight k ω = 1)
    (hErrorIntegrable : ∀ k, Integrable (fun ω => |estimate k ω - theta|) μ)
    (hAggregateMeasurable :
      AEStronglyMeasurable (fun ω => (∑ k, weight k ω * estimate k ω) - theta) μ) :
    Integrable (fun ω => (∑ k, weight k ω * estimate k ω) - theta) μ := by
  classical
  have hEnvelope : Integrable (fun ω => ∑ k, |estimate k ω - theta|) μ :=
    integrable_finsetSum Finset.univ (fun k _ => hErrorIntegrable k)
  apply hEnvelope.mono' hAggregateMeasurable
  exact Filter.Eventually.of_forall fun ω => by
    simpa only [Real.norm_eq_abs] using
      convex_error_le_sum (fun k => weight k ω) (fun k => estimate k ω) theta
        (hNonnegative ω) (hSum ω)

/-- Finite convex aggregation preserves actual integral L1 convergence, even for random weights.
Measurability of the aggregate is supplied separately; its integrability follows from domination. -/
theorem convex_aggregate_l1 (μ : Measure Ω)
    (weight estimate : ℕ → K → Ω → ℝ) (theta : ℝ)
    (hNonnegative : ∀ n ω k, 0 ≤ weight n k ω)
    (hSum : ∀ n ω, ∑ k, weight n k ω = 1)
    (hErrorIntegrable : ∀ n k, Integrable (fun ω => |estimate n k ω - theta|) μ)
    (hAggregateMeasurable : ∀ n,
      AEStronglyMeasurable (fun ω => (∑ k, weight n k ω * estimate n k ω) - theta) μ)
    (hReference : ∀ k,
      Tendsto (fun n => ∫ ω, |estimate n k ω - theta| ∂μ) atTop (𝓝 0)) :
    Tendsto (fun n => ∫ ω, |(∑ k, weight n k ω * estimate n k ω) - theta| ∂μ)
      atTop (𝓝 0) := by
  classical
  have hBound n ω : |(∑ k, weight n k ω * estimate n k ω) - theta| ≤
      ∑ k, |estimate n k ω - theta| :=
    convex_error_le_sum (fun k => weight n k ω) (fun k => estimate n k ω) theta
      (hNonnegative n ω) (hSum n ω)
  have hSumIntegrable n : Integrable (fun ω => ∑ k, |estimate n k ω - theta|) μ :=
    integrable_finsetSum Finset.univ (fun k _ => hErrorIntegrable n k)
  have hAggregateIntegrable n :
      Integrable (fun ω => (∑ k, weight n k ω * estimate n k ω) - theta) μ :=
    convex_aggregate_error_integrable μ (weight n) (estimate n) theta
      (hNonnegative n) (hSum n) (hErrorIntegrable n) (hAggregateMeasurable n)
  have hLimit : Tendsto (fun n => ∑ k, ∫ ω, |estimate n k ω - theta| ∂μ)
      atTop (𝓝 0) := by
    simpa using tendsto_finsetSum Finset.univ (fun k _ => hReference k)
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hLimit
  · intro n
    exact integral_nonneg (fun _ => abs_nonneg _)
  · intro n
    calc
      (∫ ω, |(∑ k, weight n k ω * estimate n k ω) - theta| ∂μ)
          ≤ ∫ ω, ∑ k, |estimate n k ω - theta| ∂μ :=
        integral_mono (hAggregateIntegrable n).abs (hSumIntegrable n) (hBound n)
      _ = ∑ k, ∫ ω, |estimate n k ω - theta| ∂μ :=
        integral_finsetSum Finset.univ (fun k _ => hErrorIntegrable n k)

end Aggregation

end

end HetICS.RA
