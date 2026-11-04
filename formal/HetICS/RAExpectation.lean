import Mathlib.MeasureTheory.Function.UniformIntegrable
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Linarith

/-!
# From consistency to actual expectations

These results apply to measurable, whole-sample-space versions of an estimator, including a
fixed finite extension on unavailable-output events. Convergence in probability and uniform
integrability are explicit statistical premises. Mathlib's Vitali theorem then proves convergence
of the actual absolute-error integrals and actual expectations; no abstract expected-estimate
sequence or vanishing-bias premise is substituted for those integrals.
-/

open Filter MeasureTheory
open scoped Topology ENNReal

namespace HetICS.RA

set_option autoImplicit false
noncomputable section

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Vitali's theorem specialized to real-valued estimation errors and an integrable target.
The target may itself be random, as in a fixed collection of heterogeneous cluster effects. -/
theorem l1_error_tendsto_zero_of_inMeasure_ui [IsFiniteMeasure μ]
    {estimate : ℕ → Ω → ℝ} {target : Ω → ℝ}
    (hTarget : MemLp target 1 μ)
    (hUI : UniformIntegrable estimate 1 μ)
    (hConsistent : TendstoInMeasure μ estimate atTop target) :
    Tendsto (fun n => ∫ ω, |estimate n ω - target ω| ∂μ) atTop (𝓝 0) := by
  have hLp : Tendsto (fun n => eLpNorm (estimate n - target) 1 μ) atTop (𝓝 0) :=
    tendsto_Lp_finite_of_tendstoInMeasure (by simp) (by simp)
      hUI.aestronglyMeasurable hTarget hUI.unifIntegrable hConsistent
  have hReal := (ENNReal.tendsto_toReal ENNReal.zero_ne_top).comp hLp
  have hEq (n : ℕ) : (∫ ω, |estimate n ω - target ω| ∂μ) =
      (eLpNorm (estimate n - target) 1 μ).toReal := by
    rw [eLpNorm_one_eq_lintegral_enorm]
    simpa only [Real.norm_eq_abs, Pi.sub_apply] using
      integral_norm_eq_lintegral_enorm
        ((hUI.aestronglyMeasurable n).sub hTarget.aestronglyMeasurable)
  simpa only [Function.comp_def, ENNReal.toReal_zero, ← hEq] using hReal

/-- The same premises imply convergence of genuine Bochner expectations. -/
theorem expectation_tendsto_of_inMeasure_ui [IsFiniteMeasure μ]
    {estimate : ℕ → Ω → ℝ} {target : Ω → ℝ}
    (hTarget : MemLp target 1 μ)
    (hUI : UniformIntegrable estimate 1 μ)
    (hConsistent : TendstoInMeasure μ estimate atTop target) :
    Tendsto (fun n => ∫ ω, estimate n ω ∂μ) atTop (𝓝 (∫ ω, target ω ∂μ)) := by
  apply tendsto_integral_of_L1' target hTarget.aestronglyMeasurable
    (Filter.Eventually.of_forall fun n => (hUI.memLp n).integrable (by simp))
  exact tendsto_Lp_finite_of_tendstoInMeasure (by simp) (by simp)
    hUI.aestronglyMeasurable hTarget hUI.unifIntegrable hConsistent

/-- Consistency and uniform integrability of the complete RA estimator give mean absolute
error convergence to zero. All clustering, selection and availability rules are included in
`estimate`; the theorem does not require sample splitting or selection-estimation independence. -/
theorem ra_l1_consistent [IsProbabilityMeasure μ]
    {estimate : ℕ → Ω → ℝ} (θ : ℝ)
    (hUI : UniformIntegrable estimate 1 μ)
    (hConsistent : TendstoInMeasure μ estimate atTop (fun _ => θ)) :
    Tendsto (fun n => ∫ ω, |estimate n ω - θ| ∂μ) atTop (𝓝 0) :=
  l1_error_tendsto_zero_of_inMeasure_ui (memLp_const θ) hUI hConsistent

/-- Asymptotic unbiasedness is stated as convergence of the integral of the estimator itself. -/
theorem ra_asymptotically_unbiased [IsProbabilityMeasure μ]
    {estimate : ℕ → Ω → ℝ} (θ : ℝ)
    (hUI : UniformIntegrable estimate 1 μ)
    (hConsistent : TendstoInMeasure μ estimate atTop (fun _ => θ)) :
    Tendsto (fun n => ∫ ω, estimate n ω ∂μ) atTop (𝓝 θ) := by
  simpa using expectation_tendsto_of_inMeasure_ui (memLp_const θ) hUI hConsistent

/-- A real absolute-error integral controls the bias of an integrable estimator. -/
theorem abs_expectation_error_le [IsProbabilityMeasure μ]
    {estimate : Ω → ℝ} (θ : ℝ) (hInt : Integrable estimate μ) :
    |(∫ ω, estimate ω ∂μ) - θ| ≤ ∫ ω, |estimate ω - θ| ∂μ := by
  have hEq : (∫ ω, estimate ω ∂μ) - θ = ∫ ω, (estimate ω - θ) ∂μ := by
    rw [integral_sub hInt (integrable_const θ)]
    simp
  rw [hEq]
  exact abs_integral_le_integral_abs

/-- This route is useful after proving the convex-weight absolute-error bound: the aggregate
requires no separate uniform-integrability premise once its mean absolute error tends to zero. -/
theorem expectation_tendsto_of_l1_error [IsProbabilityMeasure μ]
    {estimate : ℕ → Ω → ℝ} (θ : ℝ)
    (hInt : ∀ n, Integrable (estimate n) μ)
    (hL1 : Tendsto (fun n => ∫ ω, |estimate n ω - θ| ∂μ) atTop (𝓝 0)) :
    Tendsto (fun n => ∫ ω, estimate n ω ∂μ) atTop (𝓝 θ) := by
  have hLower : Tendsto (fun n => θ - ∫ ω, |estimate n ω - θ| ∂μ)
      atTop (𝓝 θ) := by simpa using tendsto_const_nhds.sub hL1
  have hUpper : Tendsto (fun n => θ + ∫ ω, |estimate n ω - θ| ∂μ)
      atTop (𝓝 θ) := by simpa using tendsto_const_nhds.add hL1
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le hLower hUpper ?_ ?_
  · intro n
    have h := (abs_le.mp (abs_expectation_error_le θ (hInt n))).1
    linarith
  · intro n
    have h := (abs_le.mp (abs_expectation_error_le θ (hInt n))).2
    linarith

end
end HetICS.RA
