import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.Tactic.Linarith

/-!
# Moment and separation bridges for regression adjustment

The empirical RA error is a numerator divided by residual exposure variation.
The theorem below derives consistency from convergence of these two moments, without requiring
independence between them. Laws of large numbers for the chosen sampling regime supply the
moment premises. The second result records the deterministic center-score separation argument.
-/

open Filter MeasureTheory
open scoped Topology

namespace HetICS.RA

set_option autoImplicit false

theorem ratio_consistent_of_moments {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (numerator denominator : ℕ → Ω → ℝ) (theta q : ℝ)
    (hq : 0 < q)
    (hNumerator : TendstoInMeasure μ numerator atTop (fun _ => 0))
    (hDenominator : TendstoInMeasure μ denominator atTop (fun _ => q)) :
    TendstoInMeasure μ (fun n ω => theta + numerator n ω / denominator n ω)
      atTop (fun _ => theta) := by
  rw [tendstoInMeasure_iff_norm] at hNumerator hDenominator ⊢
  intro ε hε
  have hq2 : 0 < q / 2 := half_pos hq
  have hεq : 0 < ε * (q / 2) := mul_pos hε hq2
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (by simpa using (hDenominator (q / 2) hq2).add (hNumerator (ε * (q / 2)) hεq))
  · intro n
    exact bot_le
  · intro n
    refine (measure_mono (fun ω hω => ?_)).trans (measure_union_le _ _)
    change q / 2 ≤ |denominator n ω - q| ∨ ε * (q / 2) ≤ |numerator n ω|
    by_contra h
    push_neg at h
    have hd : q / 2 < denominator n ω := by
      have hh := (abs_lt.mp (show |denominator n ω - q| < q / 2 from h.1)).1
      linarith
    have hdpos : 0 < denominator n ω := lt_trans hq2 hd
    have hn : |numerator n ω| < ε * (q / 2) := by simpa using h.2
    have hratio : |numerator n ω / denominator n ω| < ε := by
      rw [abs_div, abs_of_pos hdpos, div_lt_iff₀ hdpos]
      exact hn.trans (mul_lt_mul_of_pos_left hd hε)
    have hlarge : ε ≤ |numerator n ω / denominator n ω| := by
      simpa only [Set.mem_setOf_eq, add_sub_cancel_left, Real.norm_eq_abs] using hω
    exact (not_le_of_gt hratio) hlarge

/-- A positive population score gap survives controlled score and log-prior errors. -/
theorem center_score_gap (n gap truth wrong fittedTruth fittedWrong priorTruth priorWrong
    priorBound : ℝ)
    (hn : 0 ≤ n) (hgap : gap ≤ truth - wrong)
    (hTruth : |fittedTruth - truth| ≤ gap / 4)
    (hWrong : |fittedWrong - wrong| ≤ gap / 4)
    (hPrior : -priorBound ≤ priorTruth - priorWrong) :
    n * (gap / 2) - priorBound ≤
      (priorTruth + n * fittedTruth) - (priorWrong + n * fittedWrong) := by
  have ht := (abs_le.mp hTruth).1
  have hw := (abs_le.mp hWrong).2
  have hs : gap / 2 ≤ fittedTruth - fittedWrong := by linarith
  have hm := mul_le_mul_of_nonneg_left hs hn
  nlinarith

end HetICS.RA
