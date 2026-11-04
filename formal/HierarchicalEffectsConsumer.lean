import HierarchicalEffects

noncomputable section
open scoped BigOperators
open BinaryFormal.HierarchicalEffects

namespace BinaryFormal.HierarchicalEffectsConsumer

variable {I K : Type*} [DecidableEq I] [DecidableEq K] [Fintype K]

/-- Consumer for the within-cluster-stability bridge used by the manuscript. -/
theorem stable_cluster_targets_aggregate (p : BinaryCore.SamplePartition I K)
    (rowEffect : I → ℝ) (clusterEffect : K → ℝ)
    (h : WithinClusterStable p rowEffect clusterEffect) :
    ∑ k, p.weight k * clusterEffect k = p.fullMean rowEffect :=
  h.personWeighted_eq_fullMean p rowEffect clusterEffect

/-- Consumer for the common-upper-level decomposition. -/
theorem centered_cluster_deviations_aggregate (p : BinaryCore.SamplePartition I K)
    (mu : ℝ) (deviation : K → ℝ)
    (hzero : ∑ k, p.weight k * deviation k = 0) :
    ∑ k, p.weight k * (mu + deviation k) = mu :=
  weighted_eq_upperMean p mu deviation hzero

#print axioms stable_cluster_targets_aggregate
#print axioms centered_cluster_deviations_aggregate

end BinaryFormal.HierarchicalEffectsConsumer

