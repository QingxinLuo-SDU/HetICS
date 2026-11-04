import HetICS.RAProjection
import HetICS.RAMoments
import HetICS.RASelection
import HetICS.RAExpectation

/-!
# Complete conditional RA point-estimation chain

The good event includes label recovery, membership in a finite admissible family, and numerical
availability. Each family's empirical error is the FWL numerator divided by its residual bread.
Moment convergence, a vanishing probability of leaving this event, and uniform integrability
yield actual mean absolute error and expectation convergence for the learned estimator.
The threshold and any tuning rule enter through the final selection and good event.
-/

open Filter MeasureTheory
open scoped Topology BigOperators

namespace HetICS.RA

set_option autoImplicit false
noncomputable section

theorem learned_ra_point_guarantee {J Ω : Type*} [Fintype J] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (numerator denominator : J → ℕ → Ω → ℝ) (q : J → ℝ)
    (actual : ℕ → Ω → ℝ) (theta : ℝ) (good : ℕ → Set Ω)
    (hPositive : ∀ j, 0 < q j)
    (hNumerator : ∀ j, TendstoInMeasure μ (numerator j) atTop (fun _ => 0))
    (hDenominator : ∀ j, TendstoInMeasure μ (denominator j) atTop (fun _ => q j))
    (hMember : ∀ n ω, ω ∈ good n →
      ∃ j, actual n ω = theta + numerator j n ω / denominator j n ω)
    (hGood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0))
    (hUI : UniformIntegrable actual 1 μ) :
    TendstoInMeasure μ actual atTop (fun _ => theta) ∧
      Tendsto (fun n => ∫ ω, |actual n ω - theta| ∂μ) atTop (𝓝 0) ∧
      Tendsto (fun n => ∫ ω, actual n ω ∂μ) atTop (𝓝 theta) := by
  have hConsistent := learned_consistent_of_finite_references μ
    (fun j n ω => theta + numerator j n ω / denominator j n ω)
    actual theta good hMember
    (fun j => ratio_consistent_of_moments μ (numerator j) (denominator j) theta (q j)
      (hPositive j) (hNumerator j) (hDenominator j)) hGood
  exact ⟨hConsistent, ra_l1_consistent theta hUI hConsistent,
    ra_asymptotically_unbiased theta hUI hConsistent⟩

/-- Common-effect convex aggregation needs no independence between weights and estimates. -/
theorem common_effect_aggregate_unbiased {K Ω : Type*} [Fintype K] [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (weight estimate : ℕ → K → Ω → ℝ) (theta : ℝ)
    (hNonnegative : ∀ n ω k, 0 ≤ weight n k ω)
    (hSum : ∀ n ω, ∑ k, weight n k ω = 1)
    (hInt : ∀ n k, Integrable (estimate n k) μ)
    (hMeasurable : ∀ n,
      AEStronglyMeasurable (fun ω => (∑ k, weight n k ω * estimate n k ω) - theta) μ)
    (hL1 : ∀ k, Tendsto (fun n => ∫ ω, |estimate n k ω - theta| ∂μ)
      atTop (𝓝 0)) :
    Tendsto (fun n => ∫ ω, (∑ k, weight n k ω * estimate n k ω) ∂μ)
      atTop (𝓝 theta) := by
  have hErrorInt n k : Integrable (fun ω => |estimate n k ω - theta|) μ :=
    ((hInt n k).sub (integrable_const theta)).abs
  have hAggregateInt n : Integrable (fun ω => ∑ k, weight n k ω * estimate n k ω) μ := by
    have h := (convex_aggregate_error_integrable μ (weight n) (estimate n) theta
      (hNonnegative n) (hSum n) (hErrorInt n) (hMeasurable n)).add (integrable_const theta)
    exact h.congr (Filter.Eventually.of_forall (fun _ => sub_add_cancel _ _))
  exact expectation_tendsto_of_l1_error theta hAggregateInt
    (convex_aggregate_l1 μ weight estimate theta hNonnegative hSum hErrorInt hMeasurable hL1)

end
end HetICS.RA
