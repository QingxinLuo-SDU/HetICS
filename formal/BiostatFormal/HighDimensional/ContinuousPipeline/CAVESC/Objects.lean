import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic

/-!
# CAVES-C objects

Current cluster-specific adjustment cores, their coverage relation, and the latent-cluster effect
decomposition used by the active CAVES-C theory package.
-/

namespace BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC

set_option autoImplicit false

section AdjustmentCore

variable {Feature Cluster : Type*} [DecidableEq Feature]

/-- The required adjustment core for latent cluster `k`. -/
def clusterCore (Z shared : Finset Feature)
    (specific : Cluster → Finset Feature) (k : Cluster) : Finset Feature :=
  Z ∪ shared ∪ specific k

/-- The selected variables cover every member of the required adjustment core. -/
def SafeAdjustment (core selected : Finset Feature) : Prop :=
  core ⊆ selected

/-- Prespecified variables are contained in every cluster-specific core. -/
theorem clusterCore_Z_subset
    (Z shared : Finset Feature) (specific : Cluster → Finset Feature) (k : Cluster) :
    Z ⊆ clusterCore Z shared specific k := by
  intro feature hFeature
  exact Finset.mem_union_left (specific k) (Finset.mem_union_left shared hFeature)

/-- Shared variables are contained in every cluster-specific core. -/
theorem clusterCore_shared_subset
    (Z shared : Finset Feature) (specific : Cluster → Finset Feature) (k : Cluster) :
    shared ⊆ clusterCore Z shared specific k := by
  intro feature hFeature
  exact Finset.mem_union_left (specific k) (Finset.mem_union_right Z hFeature)

/-- Cluster-specific variables are contained in their corresponding cluster core. -/
theorem clusterCore_specific_subset
    (Z shared : Finset Feature) (specific : Cluster → Finset Feature) (k : Cluster) :
    specific k ⊆ clusterCore Z shared specific k := by
  intro feature hFeature
  exact Finset.mem_union_right (Z ∪ shared) hFeature

/-- Adding precision variables preserves coverage of the adjustment core. -/
theorem safeAdjustment_mono
    (core selected precision : Finset Feature)
    (hSafe : SafeAdjustment core selected) :
    SafeAdjustment core (selected ∪ precision) := by
  intro feature hCore
  exact Finset.mem_union_left precision (hSafe hCore)

end AdjustmentCore

/--
A latent-cluster effect decomposed around a designated mean parameter for an externally specified
random-effects model. Distributional and centering conditions are separate source assumptions.
-/
structure LatentClusterEffectModel (Cluster : Type*) where
  effect : Cluster → ℝ
  mean : ℝ
  deviation : Cluster → ℝ
  decompose : ∀ k, effect k = mean + deviation k

end BiostatFormal.HighDimensional.ContinuousPipeline.CAVESC
