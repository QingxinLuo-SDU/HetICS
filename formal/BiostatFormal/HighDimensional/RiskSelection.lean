import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Deterministic risk-gap selection

This file contains the deterministic core of the risk-to-label implication used in the
multiview label-recovery case study. It deliberately assumes, rather than proves, candidate
coverage, a uniform risk-deviation bound, and a positive population risk gap.
-/

namespace BiostatFormal.HighDimensional

/--
If empirical and population risks are uniformly within `u`, every incorrect candidate has a
strict population-risk gap larger than `2 * u + η`, and the selected candidate is empirically
within `η` of a known exact candidate, then the selected candidate is exact.

`hExactWitness` is retained to record the intended meaning of `zGood` and to connect this
deterministic lemma to an upstream candidate-coverage theorem.
-/
theorem exact_of_uniform_deviation_and_margin
    {Candidate : Type*}
    (Exact : Candidate → Prop)
    (R Rhat : Candidate → ℝ)
    (zGood zSelected : Candidate)
    (u η : ℝ)
    (hExactWitness : Exact zGood)
    (hDeviation : ∀ z, |Rhat z - R z| ≤ u)
    (hMargin : ∀ z, Exact zGood → ¬ Exact z → R zGood + 2 * u + η < R z)
    (hSelected : Rhat zSelected ≤ Rhat zGood + η) :
    Exact zSelected := by
  by_contra hNotExact
  have hSelectedLower : R zSelected - u ≤ Rhat zSelected := by
    have h := (abs_le.mp (hDeviation zSelected)).1
    linarith
  have hGoodUpper : Rhat zGood ≤ R zGood + u := by
    have h := (abs_le.mp (hDeviation zGood)).2
    linarith
  have hGap := hMargin zSelected hExactWitness hNotExact
  linarith

end BiostatFormal.HighDimensional
