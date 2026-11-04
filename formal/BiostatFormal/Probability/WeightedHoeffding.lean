import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.Ring

/-!
# Weighted Hoeffding bounds for frozen center-level risks

This file proves the pointwise concentration layer used by the revised multiview P0 contract.
The random coordinates are independent center blocks.  A center-level loss may summarize dependent
observations inside that center; no row-level independence is assumed here.

The normalizer specialization is deliberately affine and frozen outside the current evaluation
sample.  Data-selected nonlinear normalizers and post-aggregation clipping are outside this file.
-/

namespace BiostatFormal.Probability

open MeasureTheory ProbabilityTheory Real Set

noncomputable section

universe uIndex uSample

/-- A frozen finite weighted sum of center-level losses. -/
def weightedCenterRisk
    {Index : Type uIndex} {Sample : Type uSample}
    (centers : Finset Index) (weight : Index → ℝ) (loss : Index → Sample → ℝ)
    (sample : Sample) : ℝ :=
  ∑ center ∈ centers, weight center * loss center sample

/-- The corresponding weighted sum of center-level population means. -/
def weightedCenterPopulationRisk
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (centers : Finset Index) (weight : Index → ℝ)
    (loss : Index → Sample → ℝ) : ℝ :=
  ∑ center ∈ centers, weight center * ∫ sample, loss center sample ∂μ

/-- The centered weighted sum to which the sub-Gaussian addition theorem is applied. -/
def centeredWeightedCenterRisk
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (centers : Finset Index) (weight : Index → ℝ)
    (loss : Index → Sample → ℝ) (sample : Sample) : ℝ :=
  ∑ center ∈ centers,
    weight center * (loss center sample - ∫ sample', loss center sample' ∂μ)

/--
The exact Hoeffding MGF proxy for weights `weight` and a common loss envelope of length `B`.
Its real value is `(B^2 / 4) * ∑ center, weight center ^ 2` when `B ≥ 0`.
-/
def weightedHoeffdingProxy
    {Index : Type uIndex} (centers : Finset Index) (weight : Index → ℝ) (B : ℝ) : NNReal :=
  ∑ center ∈ centers,
    Real.toNNReal (weight center ^ 2) * (‖B‖₊ / 2) ^ 2

theorem toNNReal_sq_eq_mk (x : ℝ) :
    Real.toNNReal (x ^ 2) = ⟨x ^ 2, sq_nonneg x⟩ := by
  exact Real.toNNReal_of_nonneg (sq_nonneg x)

/-- The NNReal proxy has the familiar exact real-valued weighted Hoeffding form. -/
theorem weightedHoeffdingProxy_coe
    {Index : Type uIndex} (centers : Finset Index) (weight : Index → ℝ) (B : ℝ)
    (hB : 0 ≤ B) :
    (weightedHoeffdingProxy centers weight B : ℝ) =
      B ^ 2 / 4 * ∑ center ∈ centers, weight center ^ 2 := by
  simp only [weightedHoeffdingProxy, NNReal.coe_sum, NNReal.coe_mul, NNReal.coe_pow,
    NNReal.coe_div, NNReal.coe_ofNat, Real.coe_toNNReal _ (sq_nonneg _), coe_nnnorm,
    Real.norm_eq_abs]
  rw [abs_of_nonneg hB]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro center hCenter
  ring

theorem centeredWeightedCenterRisk_eq_sub
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) (centers : Finset Index) (weight : Index → ℝ)
    (loss : Index → Sample → ℝ) (sample : Sample) :
    centeredWeightedCenterRisk μ centers weight loss sample =
      weightedCenterRisk centers weight loss sample -
        weightedCenterPopulationRisk μ centers weight loss := by
  simp_rw [centeredWeightedCenterRisk, weightedCenterRisk, weightedCenterPopulationRisk, mul_sub]
  rw [Finset.sum_sub_distrib]

/-- A frozen affine normalization of a scalar risk. -/
def frozenAffineRisk (location scale value : ℝ) : ℝ :=
  (value - location) / scale

/--
Hoeffding's lemma plus independence: a weighted sum of independent center losses in `[0,B]`,
centered at its weighted population mean, has the exact weighted Hoeffding MGF proxy.
-/
theorem hasSubgaussianMGF_centeredWeightedCenterRisk
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (centers : Finset Index) (weight : Index → ℝ) (loss : Index → Sample → ℝ)
    (B : ℝ)
    (hIndep : iIndepFun loss μ)
    (hMeasurable : ∀ center ∈ centers, AEMeasurable (loss center) μ)
    (hEnvelope : ∀ center ∈ centers, ∀ᵐ sample ∂μ, loss center sample ∈ Set.Icc 0 B) :
    HasSubgaussianMGF
      (centeredWeightedCenterRisk μ centers weight loss)
      (weightedHoeffdingProxy centers weight B) μ := by
  let centeredScaled : Index → Sample → ℝ := fun center sample =>
    weight center * (loss center sample - ∫ sample', loss center sample' ∂μ)
  have hCenteredScaledIndep : iIndepFun centeredScaled μ := by
    exact hIndep.comp
      (fun center value => weight center * (value - ∫ sample, loss center sample ∂μ))
      (fun _ => by fun_prop)
  have hCenteredScaledSubgaussian :
      ∀ center ∈ centers,
        HasSubgaussianMGF (centeredScaled center)
          (Real.toNNReal (weight center ^ 2) *
            (‖B‖₊ / 2) ^ 2) μ := by
    intro center hCenter
    have hBase := hasSubgaussianMGF_of_mem_Icc
      (hMeasurable center hCenter) (hEnvelope center hCenter)
    simpa [centeredScaled, toNNReal_sq_eq_mk] using hBase.const_mul (weight center)
  change HasSubgaussianMGF
    (fun sample => ∑ center ∈ centers,
      weight center * (loss center sample - ∫ sample', loss center sample' ∂μ))
    (∑ center ∈ centers,
      Real.toNNReal (weight center ^ 2) * (‖B‖₊ / 2) ^ 2) μ
  exact HasSubgaussianMGF.sum_of_iIndepFun
    hCenteredScaledIndep hCenteredScaledSubgaussian

/-- A reusable two-sided tail consequence of a sub-Gaussian MGF certificate. -/
theorem measureReal_abs_ge_le_of_hasSubgaussianMGF
    {Sample : Type uSample} [MeasurableSpace Sample]
    {μ : Measure Sample} [IsProbabilityMeasure μ]
    {X : Sample → ℝ} {c : NNReal} (hX : HasSubgaussianMGF X c μ)
    {t : ℝ} (ht : 0 ≤ t) :
    μ.real {sample | t ≤ |X sample|} ≤
      2 * Real.exp (-t ^ 2 / (2 * (c : ℝ))) := by
  have hPositive :
      μ.real {sample | t ≤ X sample} ≤ Real.exp (-t ^ 2 / (2 * (c : ℝ))) :=
    hX.measure_ge_le ht
  have hNegative :
      μ.real {sample | t ≤ -X sample} ≤ Real.exp (-t ^ 2 / (2 * (c : ℝ))) := by
    simpa only [Pi.neg_apply] using hX.neg.measure_ge_le ht
  have hSubset :
      {sample | t ≤ |X sample|} ⊆
        {sample | t ≤ X sample} ∪ {sample | t ≤ -X sample} := by
    intro sample hSample
    simp only [Set.mem_setOf_eq, Set.mem_union] at hSample ⊢
    rcases le_or_gt 0 (X sample) with hNonnegative | hNegativeValue
    · left
      simpa [abs_of_nonneg hNonnegative] using hSample
    · right
      simpa [abs_of_neg hNegativeValue] using hSample
  calc
    μ.real {sample | t ≤ |X sample|}
        ≤ μ.real ({sample | t ≤ X sample} ∪ {sample | t ≤ -X sample}) := by
          exact ENNReal.toReal_mono (measure_ne_top μ _) (measure_mono hSubset)
    _ ≤ μ.real {sample | t ≤ X sample} + μ.real {sample | t ≤ -X sample} := by
          exact measureReal_union_le _ _
    _ ≤ Real.exp (-t ^ 2 / (2 * (c : ℝ))) +
          Real.exp (-t ^ 2 / (2 * (c : ℝ))) :=
      add_le_add hPositive hNegative
    _ = 2 * Real.exp (-t ^ 2 / (2 * (c : ℝ))) := by ring

/-- ENNReal, strict-event form used directly by finite-envelope union bounds. -/
theorem measure_abs_gt_le_of_hasSubgaussianMGF
    {Sample : Type uSample} [MeasurableSpace Sample]
    {μ : Measure Sample} [IsProbabilityMeasure μ]
    {X : Sample → ℝ} {c : NNReal} (hX : HasSubgaussianMGF X c μ)
    {t : ℝ} (ht : 0 ≤ t) :
    μ {sample | t < |X sample|} ≤
      ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * (c : ℝ)))) := by
  have hSubset : {sample | t < |X sample|} ⊆ {sample | t ≤ |X sample|} := by
    intro sample hSample
    exact (show t < |X sample| from hSample).le
  calc
    μ {sample | t < |X sample|} ≤ μ {sample | t ≤ |X sample|} := measure_mono hSubset
    _ = ENNReal.ofReal (μ.real {sample | t ≤ |X sample|}) := by
      rw [Measure.real, ENNReal.ofReal_toReal (measure_ne_top μ _)]
    _ ≤ ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * (c : ℝ)))) := by
      exact ENNReal.ofReal_le_ofReal
        (measureReal_abs_ge_le_of_hasSubgaussianMGF hX ht)

/-- The exact two-sided weighted Hoeffding tail for one fixed record and frozen context. -/
theorem measureReal_weightedCenterRisk_sub_population_ge_le
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (centers : Finset Index) (weight : Index → ℝ) (loss : Index → Sample → ℝ)
    (B : ℝ)
    (hIndep : iIndepFun loss μ)
    (hMeasurable : ∀ center ∈ centers, AEMeasurable (loss center) μ)
    (hEnvelope : ∀ center ∈ centers, ∀ᵐ sample ∂μ, loss center sample ∈ Set.Icc 0 B)
    {t : ℝ} (ht : 0 ≤ t) :
    μ.real {sample |
      t ≤ |weightedCenterRisk centers weight loss sample -
        weightedCenterPopulationRisk μ centers weight loss|} ≤
      2 * Real.exp
        (-t ^ 2 / (2 * (weightedHoeffdingProxy centers weight B : ℝ))) := by
  have hSubgaussian := hasSubgaussianMGF_centeredWeightedCenterRisk
    μ centers weight loss B hIndep hMeasurable hEnvelope
  simpa only [centeredWeightedCenterRisk_eq_sub] using
    measureReal_abs_ge_le_of_hasSubgaussianMGF hSubgaussian ht

/--
Weighted Hoeffding after a positive frozen affine normalization.  The location cancels, while the
scale changes the raw deviation threshold from `t` to `scale * t`.
-/
theorem measureReal_frozenAffineWeightedCenterRisk_sub_population_ge_le
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (centers : Finset Index) (weight : Index → ℝ) (loss : Index → Sample → ℝ)
    (B location scale : ℝ)
    (hIndep : iIndepFun loss μ)
    (hMeasurable : ∀ center ∈ centers, AEMeasurable (loss center) μ)
    (hEnvelope : ∀ center ∈ centers, ∀ᵐ sample ∂μ, loss center sample ∈ Set.Icc 0 B)
    (hScale : 0 < scale) {t : ℝ} (ht : 0 ≤ t) :
    μ.real {sample |
      t ≤ |frozenAffineRisk location scale (weightedCenterRisk centers weight loss sample) -
        frozenAffineRisk location scale
          (weightedCenterPopulationRisk μ centers weight loss)|} ≤
      2 * Real.exp
        (-(scale * t) ^ 2 / (2 * (weightedHoeffdingProxy centers weight B : ℝ))) := by
  have hEvent :
      {sample |
        t ≤ |frozenAffineRisk location scale (weightedCenterRisk centers weight loss sample) -
          frozenAffineRisk location scale
            (weightedCenterPopulationRisk μ centers weight loss)|} =
      {sample |
        scale * t ≤ |weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss|} := by
    ext sample
    simp only [Set.mem_setOf_eq, frozenAffineRisk]
    rw [show
      (weightedCenterRisk centers weight loss sample - location) / scale -
          (weightedCenterPopulationRisk μ centers weight loss - location) / scale =
        (weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss) / scale by ring]
    rw [abs_div, abs_of_pos hScale]
    simpa [mul_comm] using
      (le_div_iff₀ hScale :
        t ≤ |weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss| / scale ↔
        t * scale ≤ |weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss|)
  rw [hEvent]
  exact measureReal_weightedCenterRisk_sub_population_ge_le
    μ centers weight loss B hIndep hMeasurable hEnvelope (mul_nonneg hScale.le ht)

/-- ENNReal strict-event specialization for one frozen affine weighted record risk. -/
theorem measure_frozenAffineWeightedCenterRisk_sub_population_gt_le
    {Index : Type uIndex} {Sample : Type uSample} [MeasurableSpace Sample]
    (μ : Measure Sample) [IsProbabilityMeasure μ]
    (centers : Finset Index) (weight : Index → ℝ) (loss : Index → Sample → ℝ)
    (B location scale : ℝ)
    (hIndep : iIndepFun loss μ)
    (hMeasurable : ∀ center ∈ centers, AEMeasurable (loss center) μ)
    (hEnvelope : ∀ center ∈ centers, ∀ᵐ sample ∂μ, loss center sample ∈ Set.Icc 0 B)
    (hScale : 0 < scale) {t : ℝ} (ht : 0 ≤ t) :
    μ {sample |
      t < |frozenAffineRisk location scale (weightedCenterRisk centers weight loss sample) -
        frozenAffineRisk location scale
          (weightedCenterPopulationRisk μ centers weight loss)|} ≤
      ENNReal.ofReal
        (2 * Real.exp
          (-(scale * t) ^ 2 / (2 * (weightedHoeffdingProxy centers weight B : ℝ)))) := by
  have hEvent :
      {sample |
        t < |frozenAffineRisk location scale (weightedCenterRisk centers weight loss sample) -
          frozenAffineRisk location scale
            (weightedCenterPopulationRisk μ centers weight loss)|} =
      {sample |
        scale * t < |centeredWeightedCenterRisk μ centers weight loss sample|} := by
    ext sample
    simp only [Set.mem_setOf_eq, frozenAffineRisk, centeredWeightedCenterRisk_eq_sub]
    rw [show
      (weightedCenterRisk centers weight loss sample - location) / scale -
          (weightedCenterPopulationRisk μ centers weight loss - location) / scale =
        (weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss) / scale by ring]
    rw [abs_div, abs_of_pos hScale]
    simpa [mul_comm] using
      (lt_div_iff₀ hScale :
        t < |weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss| / scale ↔
        t * scale < |weightedCenterRisk centers weight loss sample -
          weightedCenterPopulationRisk μ centers weight loss|)
  rw [hEvent]
  exact measure_abs_gt_le_of_hasSubgaussianMGF
    (hasSubgaussianMGF_centeredWeightedCenterRisk
      μ centers weight loss B hIndep hMeasurable hEnvelope)
    (mul_nonneg hScale.le ht)

end

end BiostatFormal.Probability
