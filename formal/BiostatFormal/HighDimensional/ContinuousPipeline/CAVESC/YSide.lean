import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Objects
import Mathlib.Data.Fintype.Perm

/-!
# CAVES-C Y-side candidates

The current ridge inverse, eigenvalue truncation, projection, and QR numerical construction is not
proved exact here. Retention of a required core is instead exposed as the named ideal condition
`CoreDetectable`. This module proves only deterministic finite-set consequences and no FDR theorem.
-/

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

variable {Feature : Type*}

/-- Difference between the absolute original-feature and knockoff Y-side statistics. -/
def ySideStatistic (original knockoff : Feature → ℝ) (j : Feature) : ℝ :=
  |original j| - |knockoff j|

/-- Swapping an original feature statistic with its knockoff negates the Y-side statistic. -/
theorem ySideStatistic_swap (original knockoff : Feature → ℝ) (j : Feature) :
    ySideStatistic knockoff original j = -ySideStatistic original knockoff j := by
  exact (neg_sub (|original j|) (|knockoff j|)).symm

/-- Features whose supplied statistic reaches the candidate threshold. -/
def ySideCandidates [Fintype Feature]
    (statistic : Feature → ℝ) (threshold : ℝ) : Finset Feature :=
  Finset.univ.filter fun j => threshold ≤ statistic j

/-- Named ideal detectability condition: every required core feature reaches the threshold. -/
def CoreDetectable (core : Finset Feature)
    (statistic : Feature → ℝ) (threshold : ℝ) : Prop :=
  ∀ j ∈ core, threshold ≤ statistic j

/-- Ideal core detectability entails coverage by the Y-side candidate set. -/
theorem core_subset_ySideCandidates [Fintype Feature]
    (core : Finset Feature) (statistic : Feature → ℝ) (threshold : ℝ)
    (hDetectable : CoreDetectable core statistic threshold) :
    core ⊆ ySideCandidates statistic threshold := by
  intro j hj
  simp only [ySideCandidates, Finset.mem_filter, Finset.mem_univ, true_and]
  exact hDetectable j hj

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
