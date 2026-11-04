import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.YSide
import BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC.Stability

/-!
# CAVES-C adjustment bridge

Safe finite-set inclusion alone does not prove identification, ignorability, positivity, or causal
stability. Those scientific conditions are represented below by explicit caller-supplied predicates,
whose substantive content must be established outside this finite-set algebra.
-/

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false
noncomputable section

variable {Feature Cluster : Type*}

/-- Strictly stable members of the Y-side candidate set. -/
def finalSelected [Fintype Feature]
    (statistic : Feature → ℝ) (threshold : ℝ)
    (adjustedP : Feature → ℝ) (alpha : ℝ) : Finset Feature :=
  strictKept (ySideCandidates statistic threshold) adjustedP alpha

/-- Ideal detectability and simultaneous non-rejection imply safe cluster-specific adjustment. -/
theorem safeAdjustment_of_detectable_and_notRejected [DecidableEq Feature] [Fintype Feature]
    (Z shared : Finset Feature) (specific : Cluster → Finset Feature) (k : Cluster)
    (statistic : Feature → ℝ) (threshold : ℝ)
    (adjustedP : Feature → ℝ) (alpha : ℝ)
    (hDetectable : CoreDetectable (clusterCore Z shared specific k) statistic threshold)
    (hNotRejected : CoreNotRejected (clusterCore Z shared specific k) adjustedP alpha) :
    SafeAdjustment (clusterCore Z shared specific k)
      (finalSelected statistic threshold adjustedP alpha) := by
  exact core_subset_strictKept
    (clusterCore Z shared specific k)
    (ySideCandidates statistic threshold) adjustedP alpha
    (core_subset_ySideCandidates (clusterCore Z shared specific k) statistic threshold hDetectable)
    hNotRejected

/-- Caller-supplied scientific identification conditions for latent cluster `k`; safe inclusion alone
does not establish identification or ignorability. -/
abbrev IdentificationConditions
    (condition : Cluster → Prop) (k : Cluster) : Prop := condition k

/-- Caller-supplied positivity condition for latent cluster `k`; safe inclusion alone does not prove
positivity. -/
abbrev PositivityCondition
    (condition : Cluster → Prop) (k : Cluster) : Prop := condition k

/-- Caller-supplied causal-stability condition for latent cluster `k`; safe inclusion alone does not
prove causal stability. -/
abbrev StableCoreMechanism
    (condition : Cluster → Prop) (k : Cluster) : Prop := condition k

end

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
