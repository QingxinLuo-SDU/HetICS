import AIPWJacobian
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Moments.Variance
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-! Exact target accounting and a conditional probability-law composition.
No theorem asserts that the actual DGP satisfies the supplied CLT or consistency assumptions. -/
noncomputable section
open Filter MeasureTheory ProbabilityTheory Matrix
open scoped Topology BigOperators
namespace ErrorTarget

section Targets
variable {I : Type*} [Fintype I]
def rowMean (v : I → ℝ) : ℝ := (Fintype.card I : ℝ)⁻¹ * ∑ i, v i
def thetaStar (b : I → BinaryCore.BinaryLaw) (e m0 m1 : I → ℝ) : ℝ :=
  rowMean (fun i => (b i).fourStateMean (e i) (m0 i) (m1 i))
def doTarget (b : I → BinaryCore.BinaryLaw) : ℝ := rowMean (fun i => (b i).mu1 - (b i).mu0)
def rootBias (b : I → BinaryCore.BinaryLaw) (e m0 m1 : I → ℝ) : ℝ :=
  rowMean (fun i => ((b i).e0 - e i) *
    (((b i).mu1 - m1 i) / e i + ((b i).mu0 - m0 i) / (1 - e i)))

theorem conditional_root_bias (b : I → BinaryCore.BinaryLaw) (e m0 m1 : I → ℝ)
    (he : ∀ i, 0 < e i ∧ e i < 1) : thetaStar b e m0 m1 - doTarget b = rootBias b e m0 m1 := by
  unfold thetaStar doTarget rootBias rowMean
  simp_rw [(b _).four_state_bias _ _ _ (he _).1 (he _).2]
  rw [Finset.sum_add_distrib]
  ring

theorem conditional_root_propensity_correct (b : I → BinaryCore.BinaryLaw) (m0 m1 : I → ℝ)
    (he : ∀ i, 0 < (b i).e0 ∧ (b i).e0 < 1) :
    thetaStar b (fun i => (b i).e0) m0 m1 = doTarget b := by
  have h := conditional_root_bias b (fun i => (b i).e0) m0 m1 he
  simp [rootBias, rowMean] at h
  exact sub_eq_zero.mp h

theorem conditional_root_outcome_correct (b : I → BinaryCore.BinaryLaw) (e : I → ℝ)
    (he : ∀ i, 0 < e i ∧ e i < 1) :
    thetaStar b e (fun i => (b i).mu0) (fun i => (b i).mu1) = doTarget b := by
  have h := conditional_root_bias b e (fun i => (b i).mu0) (fun i => (b i).mu1) he
  simp [rootBias, rowMean] at h
  exact sub_eq_zero.mp h

/-- The last averaged conditional equation identifies the root target. -/
theorem root_last_equation (b : I → BinaryCore.BinaryLaw) (e m0 m1 : I → ℝ)
    (theta : ℝ) [Nonempty I]
    (hroot : rowMean (fun i => (b i).fourStateMean (e i) (m0 i) (m1 i) - theta) = 0) :
    theta = thetaStar b e m0 m1 := by
  have hn : (Fintype.card I : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp [rowMean, Finset.sum_sub_distrib, hn] at hroot
  unfold thetaStar rowMean
  field_simp
  nlinarith [hroot]
end Targets

structure ErrorParts where
  actual : ℝ
  reference : ℝ
  root : ℝ
  target : ℝ
  linear : ℝ
def ErrorParts.bias (v : ErrorParts) : ℝ := v.root - v.target
def ErrorParts.mRemainder (v : ErrorParts) : ℝ := v.reference - v.root - v.linear
def ErrorParts.selection (v : ErrorParts) : ℝ := v.actual - v.reference
theorem error_decomposition (v : ErrorParts) :
    v.actual - v.target = v.bias + v.linear + v.mRemainder + v.selection := by
  unfold ErrorParts.bias ErrorParts.mRemainder ErrorParts.selection
  ring

/-- The numerical equation residual Uhat remains in the exact inverse expansion. -/
theorem inverse_taylor_expansion {d : Type*} [Fintype d] [DecidableEq d]
    (A R : Matrix d d ℝ) (hRA : R * A = 1) (Ustar Uhat delta : d → ℝ) :
    delta = R *ᵥ Ustar + R *ᵥ ((Uhat - Ustar + A *ᵥ delta) - Uhat) := by
  rw [Matrix.mulVec_sub, Matrix.mulVec_add, Matrix.mulVec_sub,
    Matrix.mulVec_mulVec, hRA, Matrix.one_mulVec]
  abel

theorem error_targets_identity (estimate thetaP TD : ℝ) :
    estimate - TD = (estimate - thetaP) - (TD - thetaP) := by ring

section Probability
variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

/-- The good event may depend on the full data. No independence is required.
Outer-measure decay suffices; measurability is needed separately for distribution limits. -/
theorem scaled_difference_of_high_probability_eq
    (actual reference : ℕ → Ω → ℝ) (scale : ℕ → ℝ) (good : ℕ → Set Ω)
    (heq : ∀ n ω, ω ∈ good n → actual n ω = reference n ω)
    (hgood : Tendsto (fun n => μ (good n)ᶜ) atTop (𝓝 0)) :
    TendstoInMeasure μ (fun n ω => scale n * (actual n ω - reference n ω)) atTop 0 := by
  rw [tendstoInMeasure_iff_norm]
  intro ε hε
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hgood
  · intro n; exact bot_le
  · intro n
    apply measure_mono
    intro ω hω
    by_contra hnot
    have hg : ω ∈ good n := by simpa using hnot
    simp [heq n ω hg] at hω
    exact (not_le_of_gt hε) hω

private theorem add_zero (X Y : ℕ → Ω → ℝ)
    (hX : TendstoInMeasure μ X atTop 0) (hY : TendstoInMeasure μ Y atTop 0) :
    TendstoInMeasure μ (fun n ω => X n ω + Y n ω) atTop 0 := by
  rw [tendstoInMeasure_iff_norm] at hX hY ⊢
  intro ε hε
  simp only [Pi.zero_apply, sub_zero] at hX hY ⊢
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by simpa using (hX (ε / 2) (half_pos hε)).add (hY (ε / 2) (half_pos hε)))
  · intro n; exact bot_le
  · intro n
    refine (measure_mono (fun ω hω => ?_)).trans (measure_union_le _ _)
    change ε / 2 ≤ ‖X n ω‖ ∨ ε / 2 ≤ ‖Y n ω‖
    by_contra h
    push Not at h
    have := norm_add_le (X n ω) (Y n ω)
    change ε ≤ ‖X n ω + Y n ω‖ at hω
    linarith [h.1, h.2]

/-- Transfer a matched variance-consistency premise using a scaled variance difference. -/
theorem variance_consistency_transfer (reference actual : ℕ → Ω → ℝ) (v : ℝ)
    (href : TendstoInMeasure μ reference atTop (fun _ => v))
    (hdiff : TendstoInMeasure μ (fun n ω => actual n ω - reference n ω) atTop 0) :
    TendstoInMeasure μ actual atTop (fun _ => v) := by
  have hr : TendstoInMeasure μ (fun n ω => reference n ω - v) atTop 0 := by
    simpa [tendstoInMeasure_iff_norm] using href
  have hsum := add_zero μ _ _ hr hdiff
  have heq : (fun n ω => reference n ω - v + (actual n ω - reference n ω)) =
      (fun n ω => actual n ω - v) := by ext n ω; ring
  rw [heq] at hsum
  simpa [tendstoInMeasure_iff_norm] using hsum

variable [IsProbabilityMeasure μ]

/-- External CLT applies only to the explicitly supplied centered row-sum term L.
The two remainders, including a numerical residual and same-data selection, are separate. -/
theorem linear_plus_remainders_clt (L rM rSel : ℕ → Ω → ℝ)
    (hCLT : TendstoInDistribution L atTop id (fun _ => μ) (gaussianReal 0 1))
    (hM : TendstoInMeasure μ rM atTop 0) (hSel : TendstoInMeasure μ rSel atTop 0)
    (hm : ∀ n, AEMeasurable (rM n) μ) (hs : ∀ n, AEMeasurable (rSel n) μ) :
    TendstoInDistribution (fun n ω => L n ω + rM n ω + rSel n ω)
      atTop id (fun _ => μ) (gaussianReal 0 1) := by
  have h1 := hCLT.add_of_tendstoInMeasure_const (c := (0 : ℝ)) hM hm
  have h2 := h1.add_of_tendstoInMeasure_const (c := (0 : ℝ)) hSel hs
  convert! h2 using 1
  simp
  rfl

/-- Positive-limit variance gives a continuous, proof-only truncation that coincides
with the original studentized statistic with probability tending to one. -/
theorem studentize (X V : ℕ → Ω → ℝ) (σ : ℝ) (hσ : 0 < σ)
    (hX : TendstoInDistribution X atTop id (fun _ => μ) (gaussianReal 0 1))
    (hV : TendstoInMeasure μ V atTop (fun _ => σ ^ 2))
    (hmV : ∀ n, AEMeasurable (V n) μ) :
    TendstoInDistribution (fun n ω => σ * X n ω / Real.sqrt (V n ω))
      atTop id (fun _ => μ) (gaussianReal 0 1) := by
  let g : ℝ × ℝ → ℝ := fun p => σ * p.1 / Real.sqrt (max p.2 (σ ^ 2 / 2))
  have hpos : 0 < σ ^ 2 / 2 := by positivity
  have hg : Continuous g := by
    apply Continuous.div (by fun_prop) (by fun_prop)
    intro p
    exact ne_of_gt (Real.sqrt_pos.2 (hpos.trans_le (le_max_right _ _)))
  have htrunc := hX.continuous_comp_prodMk_of_tendstoInMeasure_const hg hV hmV
  have hlim : (fun ω : ℝ => g (id ω, σ ^ 2)) = id := by
    funext ω
    simp [g, max_eq_left (by nlinarith : σ ^ 2 / 2 ≤ σ ^ 2),
      Real.sqrt_sq (le_of_lt hσ), ne_of_gt hσ]
  rw [hlim] at htrunc
  have hbad : Tendsto (fun n => μ {ω | ¬ σ ^ 2 / 2 ≤ V n ω}) atTop (𝓝 0) := by
    have hv := (tendstoInMeasure_iff_norm.mp hV) (σ ^ 2 / 2) hpos
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hv
    · intro n; exact bot_le
    · intro n
      apply measure_mono
      intro ω hω
      simp only [Set.mem_setOf_eq, Real.norm_eq_abs]
      have : V n ω < σ ^ 2 / 2 := lt_of_not_ge hω
      rw [abs_sub_comm]
      exact (le_abs_self _).trans' (by linarith)
  have heq := scaled_difference_of_high_probability_eq μ
    (fun n ω => σ * X n ω / Real.sqrt (V n ω)) (fun n ω => g (X n ω, V n ω))
    (fun _ => 1) (fun n => {ω | σ ^ 2 / 2 ≤ V n ω})
    (fun n ω hω => by change σ ^ 2 / 2 ≤ V n ω at hω; simp [g, max_eq_left hω]) hbad
  apply tendstoInDistribution_of_tendstoInMeasure_sub _ id htrunc
  · convert! heq using 1
    simp
    rfl
  · intro n
    exact (aemeasurable_const.mul (hX.forall_aemeasurable n)).div
      (Real.continuous_sqrt.measurable.comp_aemeasurable (hmV n))

end Probability

/-- The row contribution includes the inverse bread and the target extraction row.
Here `meanScore` is the row's expectation under the stated reference law. -/
def centeredContribution {d : Type*} [Fintype d]
    (h : d → ℝ) (R : Matrix d d ℝ) (score meanScore : d → ℝ) : ℝ :=
  h ⬝ᵥ (R *ᵥ (score - meanScore))

section ATE
variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

/-- This is the explicit N-row mean to which the external row-array CLT applies. -/
def linearRowMean (ξ : (n : ℕ) → Ω → Fin (n + 1) → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ((n + 1 : ℕ) : ℝ)⁻¹ * ∑ i, ξ n ω i

/-- The actual estimate, reference CLT, two explicit remainders, and reference
variance consistency are composed. `thetaP n` allows the exact fixed-center weights.
Inputs refer to one specified law μ; no same-data independence is inferred. -/
theorem actual_ate_studentized
    (actual : ℕ → Ω → ℝ) (thetaP : ℕ → ℝ)
    (ξ : (n : ℕ) → Ω → Fin (n + 1) → ℝ) (rM rSel Vref Vactual : ℕ → Ω → ℝ)
    (σ : ℝ) (hσ : 0 < σ)
    (hexpansion : ∀ n ω, actual n ω - thetaP n = linearRowMean ξ n ω + rM n ω + rSel n ω)
    (hCLT : TendstoInDistribution
      (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) * linearRowMean ξ n ω / σ)
      atTop id (fun _ => μ) (gaussianReal 0 1))
    (hM : TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) * rM n ω / σ) atTop 0)
    (hSel : TendstoInMeasure μ (fun (n : ℕ) ω => Real.sqrt ((n : ℝ) + 1) * rSel n ω / σ) atTop 0)
    (hmM : ∀ n, AEMeasurable (rM n) μ) (hmSel : ∀ n, AEMeasurable (rSel n) μ)
    (hVref : TendstoInMeasure μ (fun n ω => (n + 1) * Vref n ω) atTop (fun _ => σ ^ 2))
    (hVdiff : TendstoInMeasure μ (fun n ω => (n + 1) * (Vactual n ω - Vref n ω)) atTop 0)
    (hmV : ∀ n, AEMeasurable (Vactual n) μ) :
    TendstoInDistribution (fun n ω => (actual n ω - thetaP n) / Real.sqrt (Vactual n ω))
      atTop id (fun _ => μ) (gaussianReal 0 1) := by
  have hvd : TendstoInMeasure μ
      (fun n ω => (n + 1) * Vactual n ω - (n + 1) * Vref n ω) atTop 0 := by
    simpa only [mul_sub] using hVdiff
  have hv := variance_consistency_transfer μ _ _ (σ ^ 2) hVref hvd
  have hnum := linear_plus_remainders_clt μ _ _ _ hCLT hM hSel
    (fun n => (aemeasurable_const.mul (hmM n)).div_const σ)
    (fun n => (aemeasurable_const.mul (hmSel n)).div_const σ)
  have hstudent := studentize μ _ _ σ hσ hnum hv
    (fun n => aemeasurable_const.mul (hmV n))
  apply hstudent.congr (fun n => Filter.Eventually.of_forall (fun ω => ?_)) Filter.EventuallyEq.rfl
  simp only [hexpansion]
  have hn : (0 : ℝ) < n + 1 := by positivity
  rw [Real.sqrt_mul (le_of_lt hn)]
  have hsqrt : Real.sqrt (n + 1) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hn)
  calc
    _ = (Real.sqrt (n + 1) * (linearRowMean ξ n ω + rM n ω + rSel n ω)) /
        (Real.sqrt (n + 1) * Real.sqrt (Vactual n ω)) := by
      congr 1
      field_simp [ne_of_gt hσ]
    _ = _ := mul_div_mul_left _ _ hsqrt

/-- Portmanteau applied to the standard-normal interval. -/
theorem studentized_interval_probability (T : ℕ → Ω → ℝ) (z : ℝ) (hz : 0 ≤ z)
    (hT : TendstoInDistribution T atTop id (fun _ => μ) (gaussianReal 0 1)) :
    Tendsto (fun n => μ {ω | |T n ω| ≤ z}) atTop
      (𝓝 ((gaussianReal 0 1) (Set.Icc (-z) z))) := by
  haveI : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal (by norm_num)
  have hb : ((gaussianReal 0 1).map id) (frontier (Set.Icc (-z) z)) = 0 := by
    rw [Measure.map_id, frontier_Icc (by linarith)]
    exact ((Set.finite_singleton z).insert (-z)).measure_zero _
  have h := ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto' hT.tendsto hb
  simp only [ProbabilityMeasure.coe_mk, Measure.map_id] at h
  convert h using 1
  funext n
  rw [Measure.map_apply_of_aemeasurable (hT.forall_aemeasurable n) measurableSet_Icc]
  congr 1
  ext ω
  simp [abs_le]
end ATE

theorem target_error_variance {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (estimate TD : Ω → ℝ) (thetaP : ℝ)
    (he : MemLp (fun ω => estimate ω - thetaP) 2 μ)
    (ht : MemLp (fun ω => TD ω - thetaP) 2 μ) :
    Var[fun ω => estimate ω - TD ω; μ] =
      Var[fun ω => estimate ω - thetaP; μ] + Var[fun ω => TD ω - thetaP; μ] -
        2 * cov[fun ω => estimate ω - thetaP, fun ω => TD ω - thetaP; μ] := by
  have h := variance_fun_sub he ht
  have heq : (fun ω => (estimate ω - thetaP) - (TD ω - thetaP)) =
      (fun ω => estimate ω - TD ω) := by ext ω; ring
  rw [heq] at h
  linarith

theorem interval_iff_studentized (estimate theta se z : ℝ) (hse : 0 < se) (_hz : 0 ≤ z) :
    (estimate - z * se ≤ theta ∧ theta ≤ estimate + z * se) ↔
      |(estimate - theta) / se| ≤ z := by
  rw [abs_le]
  constructor
  · rintro ⟨h1, h2⟩
    constructor
    · apply (le_div_iff₀ hse).2; nlinarith
    · apply (div_le_iff₀ hse).2; nlinarith
  · rintro ⟨h1, h2⟩
    have h1' := (le_div_iff₀ hse).1 h1
    have h2' := (div_le_iff₀ hse).1 h2
    constructor <;> nlinarith

section Reporting
variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

omit [IsProbabilityMeasure μ] in
theorem nonpositive_variance_probability (V : ℕ → Ω → ℝ) (σ : ℝ) (hσ : 0 < σ)
    (hV : TendstoInMeasure μ (fun n ω => (n + 1) * V n ω) atTop (fun _ => σ ^ 2)) :
    Tendsto (fun n => μ {ω | V n ω ≤ 0}) atTop (𝓝 0) := by
  have h := (tendstoInMeasure_iff_norm.mp hV) (σ ^ 2) (by positivity)
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds h
  · intro n; exact bot_le
  · intro n
    apply measure_mono
    intro ω hω
    simp only [Set.mem_setOf_eq, Real.norm_eq_abs]
    rw [abs_sub_comm]
    apply (le_abs_self _).trans' ?_
    have hn : (0 : ℝ) ≤ n + 1 := by positivity
    have : (n + 1 : ℝ) * V n ω ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hn hω
    linarith

/-- Reported coverage counts unavailability as noncoverage; it is not a conditional
coverage statement obtained by deleting unsuccessful samples. -/
theorem reported_interval_coverage
    (estimate se : ℕ → Ω → ℝ) (theta : ℕ → ℝ) (z : ℝ) (hz : 0 ≤ z)
    (available : ℕ → Set Ω) (hm : ∀ n, MeasurableSet (available n))
    (hpos : ∀ n ω, ω ∈ available n → 0 < se n ω)
    (hAvailable : Tendsto (fun n => μ (available n)ᶜ) atTop (𝓝 0))
    (hT : TendstoInDistribution (fun n ω => (estimate n ω - theta n) / se n ω)
      atTop id (fun _ => μ) (gaussianReal 0 1)) :
    Tendsto (fun n => μ.real {ω | ω ∈ available n ∧
      estimate n ω - z * se n ω ≤ theta n ∧ theta n ≤ estimate n ω + z * se n ω})
      atTop (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) := by
  let E : ℕ → Set Ω := fun n => {ω | |(estimate n ω - theta n) / se n ω| ≤ z}
  have ht := studentized_interval_probability μ _ z hz hT
  have htReal : Tendsto (fun n => μ.real (E n)) atTop
      (𝓝 ((gaussianReal 0 1).real (Set.Icc (-z) z))) :=
    (ENNReal.continuousAt_toReal (measure_ne_top _ _)).tendsto.comp ht
  have havReal : Tendsto (fun n => μ.real (available n)ᶜ) atTop (𝓝 0) := by
    simpa [measureReal_def, Function.comp_def] using!
      ENNReal.continuousAt_toReal ENNReal.zero_ne_top |>.tendsto.comp hAvailable
  have hevent : ∀ n, {ω | ω ∈ available n ∧
      estimate n ω - z * se n ω ≤ theta n ∧ theta n ≤ estimate n ω + z * se n ω} =
      E n ∩ available n := by
    intro n
    ext ω
    constructor
    · rintro ⟨ha, hci⟩
      exact ⟨(interval_iff_studentized _ _ _ _ (hpos n ω ha) hz).1 hci, ha⟩
    · rintro ⟨ht, ha⟩
      exact ⟨ha, (interval_iff_studentized _ _ _ _ (hpos n ω ha) hz).2 ht⟩
  simp_rw [hevent]
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le
    (by simpa using htReal.sub havReal) htReal
  · intro n
    have hsplit := measureReal_inter_add_sdiff (μ := μ) (s := E n) (hm n)
    have hle : μ.real (E n \ available n) ≤ μ.real (available n)ᶜ :=
      measureReal_mono (Set.sdiff_subset_compl _ _)
    linarith
  · intro n
    exact measureReal_mono Set.inter_subset_left
end Reporting

#print axioms conditional_root_bias
#print axioms conditional_root_propensity_correct
#print axioms conditional_root_outcome_correct
#print axioms root_last_equation
#print axioms error_decomposition
#print axioms inverse_taylor_expansion
#print axioms error_targets_identity
#print axioms scaled_difference_of_high_probability_eq
#print axioms variance_consistency_transfer
#print axioms linear_plus_remainders_clt
#print axioms studentize
#print axioms actual_ate_studentized
#print axioms studentized_interval_probability
#print axioms target_error_variance
#print axioms interval_iff_studentized
#print axioms nonpositive_variance_probability
#print axioms reported_interval_coverage
end ErrorTarget
